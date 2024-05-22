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
    

class PoTAE(nn.Module):
    """
    Polygon Transformer AutoEncoder, a hierarchical transformer-based autoencoder.
    
    Args:
        fea_dim: feature dimension, also the number of features.
        d_model: the number of expected features in transformer layers.
        ffn_dim: the dimension of the feedforward network in transformer layers.
        dropout: the dropout value.
        max_seq_len: maximum sequence length, for polygon it's the number of points.
    """
    def __init__(self, fea_dim=7, d_model=36, num_heads=4, hidden_dim=64, ffn_dim=64, layer_repeat=1, dropout=0.5, max_seq_len=64):
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
        
        
        self.dec_layer_head = nn.Sequential(nn.Linear(hidden_dim, end * d_model),
                                            nn.Unflatten(1, (end, d_model)),
                                            deepcopy(self.transformer_encoder))
                                        
        self.dec_layer_tail = nn.Sequential(HalfExpend(),
                                            nn.Linear(d_model//2, d_model),
                                            nn.LayerNorm(d_model),
                                            deepcopy(self.transformer_encoder))
        
        
        self.enc_layers = nn.ModuleList([deepcopy(self.enc_layer_head) for _ in range(num_layers-1)]).append(self.enc_layer_tail)
        self.dec_layers = nn.ModuleList(self.dec_layer_head).extend([deepcopy(self.dec_layer_tail) for _ in range(num_layers-1)])

        
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.project = nn.Linear(fea_dim, d_model)
        self.remap = nn.Linear(d_model, fea_dim)

        self.mse_loss_func = F.mse_loss
        self.meta_loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x):
        # Project to increase feature dimension for multi-head attension
        input = self.project(x)
        # Add positional embeddings
        input = self.pos_emb(input)

        # Encoding
        hidden = input
        for enc_layer in self.enc_layers:
            hidden = enc_layer(hidden)
        
        # Decoding
        decoded = hidden
        for dec_layer in self.dec_layers:
            decoded = dec_layer(decoded)

        # Remap to original feature dimension
        decoded = self.remap(decoded)

        # Separate outputs for calculating loss
        coord_output = decoded[:, :, :2]
        meta_output1 = decoded[:, :, 2:4]
        meta_output2 = decoded[:, :, 4:]
        target_meta1 = torch.argmax(x[:, :, 2:4], dim=-1) # inner or outer points
        target_meta2 = torch.argmax(x[:, :, 4:], dim=-1) # render one-hot code

        # Loss w.r.t coordinates
        coord_loss = self.mse_loss_func(coord_output, x[:, :, :2], reduction="none")
        coord_loss = coord_loss.sum(dim=[1, 2]).mean(dim=[0])

        # Loss w.r.t meta info
        # 2 is inner or outer one-hot vocab size, 3 is render one-hot vocab size
        meta_loss1 = self.meta_loss_func(meta_output1.view(-1, 2), target_meta1.view(-1)) # contiguous()
        meta_loss2 = self.meta_loss_func(meta_output2.view(-1, 3), target_meta2.view(-1))

        # Reconstrut meta info, combine it with the output coordinates to have final output
        meta_indices1 = torch.argmax(meta_output1, dim=-1)
        meta_indices2 = torch.argmax(meta_output2, dim=-1)
        output = torch.cat([coord_output, nn.functional.one_hot(meta_indices1, 2), nn.functional.one_hot(meta_indices2, 3)], dim=-1)
        
        # return hidden, output, coord_loss*0.25 + (meta_loss1 + meta_loss2)*0.75
        return hidden, output, coord_loss + meta_loss1 + meta_loss2