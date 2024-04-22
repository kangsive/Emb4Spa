import os
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.prepare_dataset import prepare_dataset_mnist
from torch.utils.data import TensorDataset, DataLoader

from pot import Pot


def get_args_parser():
    """
    Get arguments parser for pre-training
    """
    parser = argparse.ArgumentParser('PoTAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='training batch size, (default: 256)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='the number of training iteration over the whole dataset (default: 50)')
    
    # Model parameters
    parser.add_argument('--fea_dim', default=7, type=int,
                        help='the number of features, or the feature dimension, this must match the dataset')
    parser.add_argument('--ffn_dim', default=32, type=int,
                        help='the feed-forward network dimension in transformer layers (default: 32)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout ratio during training (default: 0.1)')
    parser.add_argument('--max_seq_len', default=64, type=int,
                        help='the maximum sequence length of input tokens (default: 64)')
    parser.add_argument('--pretrain_model', default="weights/synthetic_simple_200k_50epoch_transf.pth", type=str,
                        help='path to pre-trained model (stat_dict)')
    
    # Optimizer parameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--weight_decay', default=0.0001, type=float, 
                        help='weight decay (default: 0.05)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='dataset/mnist_polygon_train_10k_subsize2000.npz', type=str,
                        help='dataset path')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing, only if cuda is availale')
    
    return parser


def get_finetune_dataset_mnist(file, dataset_size=None, train=True, max_points=64, save_path=None):
    """
    Get pretrain dataset, a dataset of random polygons

    Args:
        file (str): a numpy zip file end with '.npz'
        dataset_size (int): if file is not specified, this is used for polygon generation.
        train (bool): if the dataset for training, False for testset.
        max_points: if file is not specified, this is used for polygon generation.
        save_path: save vecterized dataset, this can save time for loading dataset next time.

    """
    if ".npz" in file:
        loaded = np.load(file)
        train_tokens, train_labels = loaded["train_tokens"], loaded["train_labels"]
        val_tokens, val_labels = loaded["val_tokens"], loaded["val_labels"]

        train_tokens = torch.tensor(train_tokens, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_tokens = torch.tensor(val_tokens, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        return train_tokens, train_labels, val_tokens, val_labels

    elif ".csv" in file:
        if train:
            train_tokens, train_labels, train_mask, val_tokens, val_labels, val_mask = prepare_dataset_mnist(file=file,
                                                                                                    with_mask=False,
                                                                                                    split_ratio=0.2,
                                                                                                    dataset_size=dataset_size,
                                                                                                    max_seq_len=max_points,
                                                                                                    train=train)
            save_name = file.replace(".csv", f"_subsize{dataset_size}.npz")
            print(f"saving {save_name}")
            np.savez(save_name, train_tokens=train_tokens, train_labels=train_labels, val_tokens=val_tokens, val_labels=val_labels)

            return train_tokens, train_labels, val_tokens, val_labels
        
        else:
            test_tokens, test_labels, test_mask = prepare_dataset_mnist(file=file,
                                                                with_mask=False,
                                                                max_seq_len=64,
                                                                dataset_size=None,
                                                                train=train)
            return test_tokens, test_labels


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Emb4Spa",
    #     # entity="kangsive",
    #     # track hyperparameters and run metadata
    #     config=args
    # )

    if args.device == "cpu":
        print("Using cpu since cuda is not available")

    # model_name = f"pot_finetune_bs{args.batch_size}_epoch{args.epochs}_runname-{wandb.run.name}"

    train_tokens, train_labels, val_tokens, val_labels  = get_finetune_dataset_mnist(args.data_path, dataset_size=2000, train=True)
    train_loader= DataLoader(TensorDataset(train_tokens, train_labels), batch_size=args.batch_size, shuffle=True)

    num_class= train_labels.unique().shape[0]

    model = Pot(fea_dim=args.fea_dim, d_model=36, ffn_dim=args.ffn_dim, dropout=args.dropout,
                max_seq_len=args.max_seq_len, num_class=num_class).to(device)
    
    # Load pre-trained weights
    pretrain_state_dict = torch.load(args.pretrain_model, map_location=torch.device('cpu'))
    pot_state_dict = model.state_dict()

    for name, param in pretrain_state_dict.items():
        # Create a new state_dict for `pot` based on compatible keys from `potae`
        if name in pot_state_dict and pot_state_dict[name].size() == param.size():
            pot_state_dict[name].copy_(param) 

    model.load_state_dict(pot_state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for inputs, labels in train_loader:
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
            val_outputs = model(val_tokens)
            val_loss = loss_func(val_outputs, val_labels).item()
            _, val_predicted = torch.max(val_outputs, dim=-1)
            val_correct = (val_predicted == val_labels).sum().item()
            val_acc = val_correct / val_tokens.shape[0]
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

        # wandb.log({
        #     "training loss": train_loss,
        #     "val loss": val_loss
        # },
        # step = epoch+1)

    # torch.save(model.state_dict(), f"weights/{model_name}.pth")
    # wandb.log_model(path=f"weights/{model_name}.pth", name=model_name)
    # wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)

