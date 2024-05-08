import os
import wandb
import torch
import argparse
import numpy as np
import torch.optim as optim

from pathlib import Path
from potae import PoTAE
from utils.generate_polygon import generate_polygon_and_vectorize
from utils.lr_schedule import adjust_learning_rate



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
    
    # Optimizer parameters
    parser.add_argument('--lr', default=0.004, type=float,
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', default=0.0001, type=float, 
                        help='weight decay (default: 0.05)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./dataset/synthetic_simple_200k.npz', type=str,
                        help='dataset path')
    
    # parser.add_argument('--log_dir', default='./output_dir',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing, only if cuda is availale')
    
    return parser


def get_pretrain_dataset(file=None, dataset_size=200000, max_points=64):
    """
    Get pretrain dataset, a dataset of random polygons

    Args:
        file (str): a numpy zip file end with '.npz'
        dataset_size (int): if file is not specified, this is used for polygon generation.
        max_points: if file is not specified, this is used for polygon generation.

    Returns:
        train_tokens: a torch tensor contains all vectorized polygon shapes.

    """
    if file is None:
        print("Warning: no dataset file is specified, will genearte a random dataset of size 200k, \
               make sure your device have enough space and it will cost about 11 minutes...")
        geoms, wkts = generate_polygon_and_vectorize(num_poly=dataset_size, max_points=max_points)
        np.savez("dataset/synthetic_simple_200k.npz", geoms=geoms, wkts=wkts)
    else:
        loaded = np.load(file)
        train_tokens = loaded["geoms"]
        wkts = loaded["wkts"]

        train_tokens = torch.tensor(train_tokens, dtype=torch.float32)
        return train_tokens
    
    
def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Emb4Spa",
        # entity="kangsive",
        # track hyperparameters and run metadata
        config=args
    )

    if args.device == "cpu":
        print("Using cpu since cuda is not available")

    model_name = f"potae_pretrain_bs{args.batch_size}_epoch{args.epochs}_runname-{wandb.run.name}"

    train_tokens = get_pretrain_dataset(args.data_path)

    model = PoTAE(fea_dim=args.fea_dim, d_model=36, ffn_dim=args.ffn_dim, dropout=args.dropout, max_seq_len=args.max_seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for idx in range(0, len(train_tokens), args.batch_size):
            inputs = train_tokens[idx:idx+args.batch_size, :, :]
            inputs = inputs.to(device)
            optimizer.zero_grad()
            _, _, loss = model(inputs)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_count += 1

        train_loss = total_loss/batch_count

        current_lr = adjust_learning_rate(optimizer, epoch+1, args.lr, 20, args.epochs)

        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Lr: {current_lr}")

        wandb.log({
            "training loss": train_loss,
            "Lr": {current_lr}
        },
        step = epoch+1)

        # with open('training_log.txt', 'a') as f:
        #     f.write(f'Epoch: {epoch+1}, Training Loss: {train_loss}\n')

    torch.save(model.state_dict(), f"./weights/{model_name}.pth")
    wandb.log_model(path=f"./weights/{model_name}.pth", name=model_name)
    wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True

    main(args)