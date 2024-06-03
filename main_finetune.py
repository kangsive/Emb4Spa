import os
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.prepare_dataset import get_finetune_dataset_mnist
from utils.lr_schedule import adjust_learning_rate
from torch.utils.data import TensorDataset, DataLoader

from pot import Pot
from evaluate import downstream_evaluate
from utils.early_stop import EarlyStopper


def get_args_parser():
    """
    Get arguments parser for pre-training
    """
    parser = argparse.ArgumentParser('PoT fine tuning', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='training batch size, (default: 256)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='the number of training iteration over the whole dataset (default: 50)')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--runs', default=1, type=int,
                        help='number of runs for an hyperparameter setting')
    parser.add_argument('--group_name', default="", type=str,
                        help='group name for wandb runs')
    parser.add_argument('--fine_tune', default=False, action="store_true",
                        help='fine tune from pre-trained model or train from scratch')
    
    # Model parameters
    parser.add_argument('--fea_dim', default=7, type=int,
                        help='the number of features, or the feature dimension, this must match the dataset')
    parser.add_argument('--d_model', default=36, type=int,
                        help='model dimension in transformer layers')
    parser.add_argument('--num_heads', default=4, type=int,
                        help='the number of heads in transformer layers')
    parser.add_argument('--hidden_dim', default=64, type=int,
                        help='hidden dimension, or embedding size')
    parser.add_argument('--layer_repeat', default=1, type=int,
                        help='the repeated number of transformer encoder layer at each hierachical stage')
    parser.add_argument('--ffn_dim', default=32, type=int,
                        help='the feed-forward network dimension in transformer layers (default: 32)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout ratio during training (default: 0.1)')
    parser.add_argument('--max_seq_len', default=64, type=int,
                        help='the maximum sequence length of input tokens (default: 64)')
    parser.add_argument('--pretrain_model', default="weights/potae_pretrain_bs256_epoch100_runname-iconic-durian-30.pth", type=str,
                        help='path to pre-trained model (stat_dict)')
    parser.add_argument('--early_stop', default=False, action="store_true",
                        help='apply early stop')
    
    # Optimizer parameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate (absolute lr)')
    parser.add_argument('--no_schedule', default=False, action="store_true")
    
    parser.add_argument('--weight_decay', default=0.0001, type=float, 
                        help='weight decay (default: 0.05)')
    
    # Dataset parameters
    parser.add_argument('--train_data', default='./dataset/mnist_polygon_train_10k.npz', type=str,
                        help='training data path')
    parser.add_argument('--eval_data', default="./dataset/mnist_polygon_test_2k.npz", type=str,
                        help='if eval_data is given, evaluate after training')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing, only if cuda is availale')
    
    return parser


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    if args.fine_tune:
        model_name = f"pot_fine_tune_bs{args.batch_size}_epoch{args.epochs}"
    else:
        model_name = f"pot_bs{args.batch_size}_epoch{args.epochs}"

    if not args.no_wandb and args.group_name == '':
        args.group_name = model_name

    if args.device == "cpu":
        print("### Using cpu since cuda is not available ###")

    train_tokens, train_labels, val_tokens, val_labels  = get_finetune_dataset_mnist(args.train_data, dataset_size=None, train=True)
    test_tokens, test_labels = get_finetune_dataset_mnist(file=args.eval_data, train=False)
    
    num_class= train_labels.unique().shape[0]

    best_model = None
    best_acc = 0
    test_accs = []
    for run in range(1, args.runs+1):
        if not args.no_wandb:
            wandb.init(
                mode="online" if not args.no_wandb else "disabled",
                project="Emb4Spa",
                entity="kangsive",
                group=args.group_name,
                config=args
            )
            wandb.run.name = f"run_{run}"

        model, test_acc = train_and_test(args,
                                        train_tokens, train_labels,
                                        val_tokens, val_labels, 
                                        test_tokens, test_labels,
                                        num_class,
                                        device)

        if not args.no_wandb:
            wandb.log({"test acc": test_acc})
            wandb.finish()

        if test_acc > best_acc:
            best_model = model
            best_acc = test_acc
        
        test_accs.append(test_acc)
        print(f"Model: {model_name}, Run: {run}, Test Acc: {test_acc}")

    print(f"Model: {model_name}, Avg Test Acc: {sum(test_accs)/len(test_accs)}")
    # torch.save(best_model.state_dict(), f"./weights/{model_name}.pth")


def train_and_test(args, train_tokens, train_labels, val_tokens, val_labels, test_tokens, test_labels, num_class, device):

    model = Pot(fea_dim=args.fea_dim, d_model=args.d_model, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                  ffn_dim=args.ffn_dim, layer_repeat=args.layer_repeat, dropout=args.dropout, max_seq_len=args.max_seq_len).to(device)
    
    if args.fine_tune:
        # Load pre-trained weights
        pretrain_state_dict = torch.load(args.pretrain_model, map_location=device)
        pot_state_dict = model.state_dict()

        for name, param in pretrain_state_dict.items():
            # Create a new state_dict for `pot` based on compatible keys from `potae`
            if name in pot_state_dict:
                if pot_state_dict[name].size() == param.size():
                    pot_state_dict[name].copy_(param)
                else:
                    print(f"key {name} have unmatch size {param.size()}")
            else:
                # print(f"key {name} missing")
                pass

        model.load_state_dict(pot_state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    loss_func = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=20, min_delta=0)

    train_loader = DataLoader(TensorDataset(train_tokens, train_labels), batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, dim=-1)
            correct += (predicted == labels).sum().item()
        
        train_loss = total_loss/len(train_loader)
        train_acc = correct / train_tokens.shape[0]

        model.eval()
        with torch.no_grad():
            val_tokens, val_labels = val_tokens.to(device), val_labels.to(device)
            val_outputs = model(val_tokens)
            val_loss = loss_func(val_outputs, val_labels).item()
            _, val_predicted = torch.max(val_outputs, dim=-1)
            val_correct = (val_predicted == val_labels).sum().item()
            val_acc = val_correct / val_tokens.shape[0]

        if not args.no_schedule:
            current_lr = adjust_learning_rate(optimizer, epoch+1, args.lr, 0, args.epochs)
        else:
            current_lr = args.lr
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}, Lr: {current_lr}")

        if args.early_stop:
            if early_stopper.early_stop(val_loss):
                print("Early Stop")
                break
        
        if not args.no_wandb:
            wandb.log({
                "train loss": train_loss,
                "val loss": val_loss,
                "train acc": train_acc,
                "val acc": val_acc,
                "lr": current_lr,
            },
            step = epoch+1)

    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    test_acc = downstream_evaluate(model, test_tokens, test_labels)
    return model, test_acc



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)

