"""
Convert dataset to the format used in resnet1D ... refer paper: ...
"""
import numpy as np
import geopandas as gpd
from utils.vector2shape import reverse_vector_polygon
from main_finetune import get_finetune_dataset_mnist
import torch

train_data_path = "dataset/mnist_polygon_train_10k.csv"
train_tokens, train_labels, val_tokens, val_labels  = get_finetune_dataset_mnist(train_data_path, dataset_size=None, train=True)

test_dataset = "dataset/mnist_polygon_test_2k.csv"
test_tokens, test_labels = get_finetune_dataset_mnist(file=test_dataset, train=False)

train_flag = [1] * train_tokens.shape[0]
test_flag = [0] * test_tokens.shape[0]
valid_flag = [-1] * val_tokens.shape[0]

tokens = torch.cat([train_tokens, test_tokens, val_tokens], dim=0)
labels = torch.cat([train_labels, test_labels, val_labels], dim=0)
flags = train_flag + test_flag + valid_flag

polys = [reverse_vector_polygon(one) for one in tokens]
ids = list(range(1, len(polys)+1))
coors = tokens[:, :, :2].contiguous()

pgon_list = []
for pgon in polys:
    ex_coors = np.expand_dims(np.array(pgon.exterior.coords), axis = 0)
    if len(pgon.interiors):
        in_coors = np.concatenate([np.array(interior.coords.xy).T for interior in pgon.interiors], axis = 0)
        in_coors = np.expand_dims(in_coors, axis = 0)
        coors = np.concatenate([in_coors, ex_coors], axis = 1)
    else:
        coors = ex_coors
    pgon_list.append(coors)
pgon_list = np.concatenate(pgon_list, axis = 0)

gdf = gpd.GeoDataFrame({"ID": ids, "TYPEID": labels, "SPLIT_0": flags, "geometry_norm": polys, "geom_coors": coors})
gdf.to_pickle("dataset/mnist_resnet1d_12k.pkl")