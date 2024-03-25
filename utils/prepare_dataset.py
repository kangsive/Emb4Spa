import pandas as pd
import numpy as np
from deep_geometry import vectorizer as gv
from deep_geometry import GeomScaler
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import utils.geom_scaler as geom_scaler


gs = GeomScaler()

def prepare_dataset(max_seq_len=64, batch_size=32, dataset_size=1000):
    types_dict = {'PK':0, 'MR': 1, 'KL':2, 'NV':3, 'WA':4, 'LG':5, 'HO':6, 'GR':7, 'REC':8, 'PGK':9}
    df = pd.read_csv("archaeology.csv")
    df['type'] = df['Aardspoor'].map(types_dict)
    df = df.dropna().reset_index(drop=True)

    def count_points(wkt):
        try:
            num_points = gv.num_points_from_wkt(wkt)
            # gv.vectorize_wkt(wkt)
            return num_points
        except:
            print("Invalid wkt string, skip it")
            return np.inf

    filtered_df = df[df['WKT'].apply(lambda x: count_points(x) <= max_seq_len)]
    df = filtered_df

    df = df[:dataset_size]

    ori_train_data, ori_train_labels, ori_val_data, ori_val_labels, ori_test_data, ori_test_labels = dataset_split(df, 0.1, 0.1)

    train_tokens, train_labels, train_mask = prepare_polygon_dataset(ori_train_data, ori_train_labels, max_seq_len)
    val_tokens, val_labels, val_mask = prepare_polygon_dataset(ori_val_data, ori_val_labels, max_seq_len, train=False)
    test_tokens, test_labels, test_mask = prepare_polygon_dataset(ori_test_data, ori_test_labels, max_seq_len, train=False)

    train_loader = DataLoader(TensorDataset(train_tokens, train_labels, train_mask), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tokens, val_labels, val_mask), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_tokens, test_labels, test_mask), batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_tokens, train_labels, train_mask, test_tokens, test_labels, test_mask

def dataset_split(df, val_split_ratio, test_split_ratio):

    data, labels = np.array(df['WKT'].tolist()), np.array(df['type'].tolist())

    num_val = int(val_split_ratio * len(df))
    num_test = int(test_split_ratio * len(df))

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[num_val+num_test:], indices[:num_val], indices[num_val:num_val+num_test]

    train_data, train_labels = data[train_indices], labels[train_indices]
    val_data, val_labels = data[val_indices], labels[val_indices]
    test_data, test_labels = data[test_indices], labels[test_indices]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def prepare_polygon_dataset(wkts, types, max_seq_len, train=True): # TODO - 1. split into train, validate, test. 2. randomly sample
    geoms, labels, start_points = [], [], []
    for i, wkt in enumerate(wkts):
        num_point = gv.num_points_from_wkt(wkt)
        if  num_point > max_seq_len:
             continue
        geom = gv.vectorize_wkt(wkt, max_points=max_seq_len, fixed_size=True)
        geoms.append(geom)
        labels.append(types[i])
        start_points.append(num_point)

    start_points = torch.tensor(start_points).unsqueeze(1)
    indices = torch.arange(max_seq_len).unsqueeze(0)
    mask = indices >= start_points
    tokens = np.stack(geoms, axis=0)
    if train:
        gs.fit(tokens)
    tokens = gs.transform(tokens)
    tokens = torch.tensor(tokens, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return tokens, labels, mask

def prepare_dataset_fixedsize(file="dataset/buildings_train_v8.npz", batch_size=32, dataset_size=None, geom_scale=None):
    loaded = np.load(file)
    fixed_size_geoms = loaded['fixed_size_geoms']
    geom_types = loaded['building_type']
    if dataset_size is not None:
        fixed_size_geoms = fixed_size_geoms[:dataset_size]
        geom_types = geom_types[:dataset_size]

    if "test" in file:
        return geom_scaler.transform(fixed_size_geoms, geom_scale), geom_types
    else:
        geom_train, geom_test, label_train, label_test = train_test_split(fixed_size_geoms, geom_types, test_size=0.2, random_state=42)
        # Normalize
        geom_scale = geom_scaler.scale(geom_train)
        geom_train = geom_scaler.transform(geom_train, geom_scale)
        geom_test = geom_scaler.transform(geom_test, geom_scale)  # re-use variance from training
        return geom_train, geom_test, label_train, label_test, geom_scale
