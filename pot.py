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
    
class HalfSqueeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), x.size(1)//2, -1) 

class HalfExpend(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), x.size(1)*2, -1)
    
    
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
    def __init__(self, fea_dim=7, d_model=36, num_heads=4, hidden_dim=64, ffn_dim=64, layer_repeat=1, num_classes=10, dropout=0.5, max_seq_len=64):
        super().__init__()

        num_layers = int(math.log(max_seq_len))
        end = max_seq_len // (2**(num_layers-1))

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim, 
                                                            dropout=dropout, batch_first=True),
                                                            num_layers=layer_repeat)

        self.enc_layer_head = nn.Sequential(self.transformer_encoder,
                                            nn.Linear(d_model, d_model//2),
                                            HalfSqueeze())
        
        self.enc_layer_tail = nn.Sequential(deepcopy(self.transformer_encoder),
                                            nn.Flatten(),
                                            nn.Linear(end * d_model, hidden_dim),
                                            nn.LayerNorm(hidden_dim))
        
        
        self.enc_layers = nn.ModuleList([deepcopy(self.enc_layer_head) for _ in range(num_layers-1)]).append(self.enc_layer_tail)

        
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.project = nn.Linear(fea_dim, d_model)

        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(64, num_classes)


    def forward(self, x):
        # Project to increase feature dimension for multi-head attension (,64,7) --> (,64,36)
        x = self.project(x)
        x = self.pos_emb(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x)
            
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
    
