import pandas as pd
import numpy as np
from deep_geometry import vectorizer as gv
from deep_geometry import GeomScaler
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.vectorizer import vectorize_wkt
import utils.geom_scaler as geom_scaler


gs = GeomScaler()


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
        if train:
            train_tokens, train_labels = loaded["train_tokens"], loaded["train_labels"]
            val_tokens, val_labels = loaded["val_tokens"], loaded["val_labels"]

            train_tokens = torch.tensor(train_tokens, dtype=torch.float32)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            val_tokens = torch.tensor(val_tokens, dtype=torch.float32)
            val_labels = torch.tensor(val_labels, dtype=torch.long)

            return train_tokens, train_labels, val_tokens, val_labels

        else:
            test_tokens, test_labels = loaded["test_tokens"], loaded["test_labels"]

            test_tokens = torch.tensor(test_tokens, dtype=torch.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.long)

            return test_tokens, test_labels


    elif ".csv" in file:
        if train:
            train_tokens, train_labels, train_mask, val_tokens, val_labels, val_mask = prepare_dataset_mnist(file=file,
                                                                                                    with_mask=False,
                                                                                                    split_ratio=0.2,
                                                                                                    dataset_size=dataset_size,
                                                                                                    max_seq_len=max_points,
                                                                                                    train=train)
            save_name = file.replace(".csv", f"_subsize{dataset_size}.npz") if dataset_size \
                        else file.replace(".csv", ".npz")
            np.savez(save_name, train_tokens=train_tokens, train_labels=train_labels, val_tokens=val_tokens, val_labels=val_labels)

            return train_tokens, train_labels, val_tokens, val_labels
        
        else:
            test_tokens, test_labels, test_mask = prepare_dataset_mnist(file=file,
                                                                with_mask=False,
                                                                max_seq_len=64,
                                                                dataset_size=None,
                                                                train=train)
            save_name = file.replace(".csv", f"_subsize{dataset_size}.npz") if dataset_size \
                        else file.replace(".csv", ".npz")
            np.savez(save_name, test_tokens=test_tokens, test_labels=test_labels)

            return test_tokens, test_labels
        

def prepare_dataset_mnist(file="dataset/mnist_polygon_train_10k.csv", train=True, max_seq_len=64,
                            batch_size=32, split_ratio=0.2, dataset_size=None, with_mask=False):
    """
    Prepare mnist dataset from wkt dataset file
    Args:
        file: dataset file path
        train: get training set (True) or test set (False)
        max_seq_len: max sequence length in transformer inputs, equals to max number of points in geometries
        dataset_size: get only part of the dataset with specified size
        with mask: creating padding mask for transfomer inputs
    Returns:
        X, y, mask (optional)
    """
    df = pd.read_csv(file)
    wkts = df['wkt'].to_numpy()
    labels = df['label'].to_numpy()

    if dataset_size:
        assert dataset_size < len(df), "Dataset size exceed the size of origial dataset"
        wkts = wkts[:dataset_size]
        labels = labels[:dataset_size]
    
    if train:
        train_data, val_data, train_labels, val_labels = train_test_split(wkts, labels, test_size=split_ratio, random_state=42)
        train_tokens, train_labels, train_mask = get_trainable_dataset(train_data, train_labels, max_seq_len, with_mask=with_mask)
        val_tokens, val_labels, val_mask = get_trainable_dataset(val_data, val_labels, max_seq_len, train=False, with_mask=with_mask)
        return train_tokens, train_labels, train_mask, val_tokens, val_labels, val_mask
    else:
        test_tokens, test_labels, test_mask = get_trainable_dataset(wkts, labels, max_seq_len, train=False, with_mask=with_mask)
        return test_tokens, test_labels, test_mask


def normalize(data, method="min_max"):
    """
    Normalize data
    """
    if method == "min_max":
        # Compute min and max values along the num_features dimension
        min_vals = np.min(data[:, :, :2], axis=1, keepdims=True)
        max_vals = np.max(data[:, :, :2], axis=1, keepdims=True)
        # Perform min-max normalization
        data[:, :, :2] -= min_vals
        data[:, :, :2] /=  max_vals - min_vals

    else:
        mean = np.mean(data[:, :, :2], axis=1, keepdims=True)
        std = np.std(data[:, :, :2], axis=1, keepdims=True)
        
        data[:, :, :2] -= mean
        data[:, :, :2] /=  std
    return data


def get_trainable_dataset(wkts, labels, max_seq_len, train=True, with_mask=False):
    """
    Convert wkt data into trainable tensors
    """
    geoms, start_points = [], []
    for wkt in wkts:
        geom = vectorize_wkt(wkt, max_points=max_seq_len, simplify=True, fixed_size=True)
        geoms.append(geom)
        if with_mask:
            num_point = gv.num_points_from_wkt(wkt)
            if  num_point > max_seq_len:
                num_point = max_seq_len
            start_points.append(num_point)

    tokens = np.stack(geoms, axis=0)

    # tokens = np.diff(tokens, axis=1)
    # tokens = np.concatenate([tokens, np.zeros((tokens.shape[0], 1, tokens.shape[2]))], axis=1)

    tokens = normalize(tokens, method="min_max")

    # if train:
    #     gs.fit(tokens)
    # tokens = gs.transform(tokens)
    tokens = torch.tensor(tokens, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    if with_mask:
        start_points = torch.tensor(start_points).unsqueeze(1)
        indices = torch.arange(max_seq_len).unsqueeze(0)
        mask = indices >= start_points
        return tokens, labels, mask

    return tokens, labels, None


def get_trainable_dataset_with_len_filter(wkts, types, max_seq_len, train=True): # TODO - 1. split into train, validate, test. 2. randomly sample
    geoms, labels, start_points = [], [], []
    for i, wkt in enumerate(wkts):
        num_point = gv.num_points_from_wkt(wkt)
        if  num_point > max_seq_len:
             continue # If cannot perform simplify process (error occurs)
        geom = gv.vectorize_wkt(wkt, max_points=max_seq_len, simplify=True, fixed_size=True)
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


def prepare_dataset_archaeology(max_seq_len=64, batch_size=32, dataset_size=1000):
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

    train_tokens, train_labels, train_mask = get_trainable_dataset_with_len_filter(ori_train_data, ori_train_labels, max_seq_len)
    val_tokens, val_labels, val_mask = get_trainable_dataset_with_len_filter(ori_val_data, ori_val_labels, max_seq_len, train=False)
    test_tokens, test_labels, test_mask = get_trainable_dataset_with_len_filter(ori_test_data, ori_test_labels, max_seq_len, train=False)

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