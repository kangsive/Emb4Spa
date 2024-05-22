import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for generating positional embedding for the input of transformer
    """
    def __init__(self, emb_dim, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1] # Avoid dismatch dimension for odd d_model
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class Pot(nn.Module):
    """
    Polygon Transformer AutoEncoder, a hierarchical transformer-based autoencoder.
    
    Args:
        fea_dim: feature dimension, also the number of features.
        d_model: the number of expected features in transformer layers.
        ffn_dim: the dimension of the feedforward network in transformer layers.
        dropout: the dropout value.
        max_seq_len: maximum sequence length, for polygon it's the number of points.
    """
    def __init__(self, fea_dim=7, d_model=512, ffn_dim=32, dropout=0.5, max_seq_len=64, num_class=10, num_layers=1):
        super().__init__()

        num_layers = int(math.log(max_seq_len))
        num_heads = 12
        hidden_dim = 128
        end = max_seq_len // (2*num_layers)
        

        self.enc_layer_head = nn.Sequential(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
                                                                       dropout=dropout, batch_first=True),
                                            nn.Linear(d_model, d_model//2))
        
        self.enc_layer_tail = nn.Sequential(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
                                                                       dropout=dropout, batch_first=True),
                                            nn.Flatten(),
                                            nn.Linear(end * d_model, hidden_dim))


        self.enc_layers = [deepcopy(self.enc_layer_head) for _ in range(num_layers-1)] + [self.enc_layer_tail]


        
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.project = nn.Linear(fea_dim, d_model)

        self.fc_norm = nn.LayerNorm(64, eps=1e-6)
        self.head_drop = nn.Dropout(0.1)
        self.head = nn.Linear(64, num_class)


    def forward(self, x):
        # Project to increase feature dimension for multi-head attension (,64,7) --> (,64,36)
        x = self.project(x)
        x = self.pos_emb(x)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
            if i != len(self.enc_layers)-1:
                # Reshape: decrese seq_len and increase fea dim;
                # layer1: (,64,18) --> (,32,36), layer2: (,32,18) --> (,16,36)
                x = x.view(x.size(0), x.size(1)//2, -1) 

        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)

        return x
    
class Classifier(nn.Module):
    def __init__(self, input_size, dense_size, num_classes, dropout):
        super().__init__()
        self.dense1 = nn.Linear(input_size, dense_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_size, num_classes)

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x
    
