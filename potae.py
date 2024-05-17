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
    def __init__(self, fea_dim=7, d_model=36, ffn_dim=32, dropout=0.5, max_seq_len=64, num_layers=1):
        super().__init__()

        self.enc_layer1 = nn.Sequential(*[deepcopy(nn.TransformerEncoderLayer(d_model=d_model, nhead=d_model//4, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)],
                                        nn.Linear(d_model, 18),
                                        )   # (,64,36) --> (,64,18)
        
        self.enc_layer2 = nn.Sequential(*[deepcopy(nn.TransformerEncoderLayer(d_model=36, nhead=d_model//6, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)],
                                        nn.Linear(36, 18),
                                        )   # (,32,36) --> (,32,18)
        
        self.enc_layer3 = nn.Sequential(*[deepcopy(nn.TransformerEncoderLayer(d_model=36, nhead=d_model//9, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)],
                                        nn.Flatten(),
                                        nn.Linear(16*36, 64),
                                        )   # (,16,36) --> (,32*36) --> (,64)

        
        self.dec_layer1 = nn.Sequential(*[deepcopy(nn.TransformerEncoderLayer(d_model=36, nhead=d_model//9, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)])
                                        
        
        self.dec_layer2 = nn.Sequential(nn.Linear(18, 36),
                                        *[deepcopy(nn.TransformerEncoderLayer(d_model=36, nhead=d_model//6, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)])
                                            # (,32,18) --> (32,36)
        
        self.dec_layer3 = nn.Sequential(nn.Linear(18, d_model),
                                        *[deepcopy(nn.TransformerEncoderLayer(d_model=d_model, nhead=d_model//4, dim_feedforward=ffn_dim,
                                                                    dropout=dropout, batch_first=True)) for _ in range(num_layers)])
                                            # (,64,18) --> (64,36)

        self.enc_layers = [self.enc_layer1, self.enc_layer2, self.enc_layer3]
        self.dec_layers = [self.dec_layer1, self.dec_layer2, self.dec_layer3]

        
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.recover = nn.Linear(64, 16*36)
        self.project = nn.Linear(fea_dim, d_model)
        self.remap = nn.Linear(d_model, fea_dim)

        self.mse_loss_func = F.mse_loss
        self.meta_loss_func = nn.CrossEntropyLoss()

    
    def Repeat_Block(num_layers, module):
        module_list = [deepcopy(module) for _ in range(num_layers)]
        nn.ModuleList(module_list)
        return module_list
    

    def forward(self, x):
        # Project to increase feature dimension for multi-head attension (,64,7) --> (,64,36)
        input = self.project(x)
        input = self.pos_emb(input)

        hidden = input
        for i, enc_layer in enumerate(self.enc_layers):
            hidden = enc_layer(hidden)
            if i != len(self.enc_layers)-1:
                # Reshape: decrese seq_len and increase fea dim;
                # layer1: (,64,18) --> (,32,36), layer2: (,32,18) --> (,16,36)
                hidden = hidden.view(hidden.size(0), hidden.size(1)//2, -1) 

        # Recover 1D vector (embedding) to 2D feature map, (,64) --> (,16*36) --> (,16,36)
        decoded = self.recover(hidden).reshape(input.size(0), 16, 36)
        for i, dec_layer in enumerate(self.dec_layers):
            if i != 0:
                # Reshape: increse seq_len and decrease fea dim;
                # layer2: (,16,36) --> (,32,18), layer2: (,32,36) --> (,64,18)
                decoded = decoded.view(decoded.size(0), decoded.size(1)*2, -1)
            decoded = dec_layer(decoded)
        
        # Remap feature dimension back to original (,64,36) --> (,64,7)
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