import os
import pickle
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import pathlib
import torch
from scipy import sparse as sp
import json

from collections import defaultdict

from conf import LOG_DIR


def permute(arr: np.ndarray):
    idx_perm = np.random.permutation(arr.size)
    return arr[idx_perm]


def mod_split(arr: np.ndarray, fold_n: int, cv_n_folds: int):
    """
    It creates the fold using the mod operator. fold_n is the test fold, (fold_n+1)%cv_n_folds is the validation fold.
    The rest is training.
    :param arr: array to be split in three parts
    :param fold_n: index of the test set in {0, ..., self.cv_n_folds}
    :param cv_n_folds: number of total folds in the cross validation
    :return: tr, vd, te split arrays
    """
    # Ensure that fold_n is less than cv_n_folds
    assert fold_n < cv_n_folds
    mods = np.mod(arr, cv_n_folds)

    te_indx = np.where(mods == fold_n)[0]
    vd_indx = np.where(mods == (fold_n + 1) % cv_n_folds)[0]

    tr = np.delete(arr, np.concatenate((te_indx, vd_indx)))
    vd = arr[vd_indx]
    te = arr[te_indx]
    return tr, vd, te


def idx_split(arr: np.ndarray, n_train: int, n_heldout: int):
    """
    Splits an array in training, validation, and test set with n_train, n_heldout, and n_heldout entries respectively.
    (negligible 1 or 2 entries differences may exist)

    :param arr: array to be split in three parts
    :param n_train: number of training entries
    :param n_heldout: number of validation entries (= number of test entries)
    :return: tr, vd, te split arrays
    """
    # Ensure that the total amount of entries is more or less equal to the size of the array
    assert abs((n_train + 2 * n_heldout) - arr.size) <= 2
    tr = arr[: n_train]
    vd = arr[n_train: n_train + n_heldout]
    te = arr[n_train + n_heldout:]
    return tr, vd, te


def idx_sequential_split(arr: np.ndarray, n_train: int, n_heldout: int):
    """
    Splits an array in training, validation, and test set with n_train, n_heldout, and n_heldout entries respectively.
    (negligible 1 or 2 entries differences may exist)

    :param arr: array to be split in three parts
    :param n_train: number of training entries
    :param n_heldout: number of validation entries (= number of test entries)
    :return: tr, vd, te split arrays
    """
    # Ensure that the total amount of entries is more or less equal to the size of the array
    assert abs((n_train + 2 * n_heldout) - arr.size) <= 2
    tr = arr[: n_train]
    vd = arr[n_train: n_train + n_heldout]
    te = arr[n_train + n_heldout:]
    return tr, vd, te


def filt(df: pd.DataFrame, tids: np.ndarray):
    """
    Filters validation and test data by discarding track ids (tids) not present in the training data.
    It also discards users with fewer than 5 interactions (since they cannot be split in a 80-20% fashion)
    :param df: pd DataFrame of validation data or test data
    :param tids: list of track ids present in the training data
    :return: the filtered dataframe + temp information for logging
    """
    num_, usr_, trk_ = len(df), df.user_id.nunique(), df.track_id.nunique()
    df = df[df.track_id.isin(set(tids))]
    df = df.groupby("user_id").filter(lambda x: len(x.drop_duplicates()) >= 5)
    return df, num_, usr_, trk_


def playcounts(df: pd.DataFrame):
    """
    Generates the playcount column
    :param df: either the train,val,test dataframe containing the interaction data
    :return: the original dataframe with the playcount column appended
    """
    df["play_count"] = 1

    # Aggregating
    df = df.groupby(["user_id", "new_track_id"]).count().reset_index()
    return df


def random_item_splitter(df: pd.DataFrame):
    """
    Performs the 80-20% item split. For a specific user (either in the validation or test set), it randomly
    picks 80% of the listened items as training and places the remaining 20% in the test set
    :param df: either validation of test dataframe containing the interaction data
    :return: two dataframes containing the interaction data for training and test data (e.g. vdalidation training and validation testing)
    """
    grouped = df.groupby("user_id")

    tr_idxs, te_idxs = list(), list()

    for i, (_, group) in enumerate(grouped):
        # Fixed at 80-20 split
        if len(group) >= 5:
            # Sampling test items
            test = group.sample(frac=0.2, replace=False).index

            # Training is the leftover data
            train = group[~group.index.isin(set(test))].index

            tr_idxs += train.tolist()
            te_idxs += test.tolist()
        else:
            raise Exception("A user has fewer than 5 different tracks")

    tr_data = df.loc[tr_idxs]
    te_data = df.loc[te_idxs]

    return tr_data, te_data


def sparsify(df: pd.DataFrame, n_tracks: int):
    """
    Creates a sparse representation of the data. To do so, it also assigns a new user_id depending on the position.

    :param df: pandas DataFrame containing the interaction data
    :return: sparse csr matrix of the data and the original dataframe with the new_user_id column
    """
    # Assign new user id for position within the matrix
    df = df.merge(
        df.user_id.drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index": "new_user_id"}))
    sp_df = sp.csr_matrix((df.play_count, (df.new_user_id, df.new_track_id)),
                          shape=(df.new_user_id.max() + 1, n_tracks))
    return sp_df, df


def save_data(dir_path: str, pandas_data: dict, scipy_data: dict, new_tids: pd.DataFrame):
    pandas_dir_path = os.path.join(dir_path, "pandas/")
    scipy_dir_path = os.path.join(dir_path, "scipy/")

    pathlib.Path(pandas_dir_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(scipy_dir_path).mkdir(parents=True, exist_ok=True)

    # Saving pandas data
    for file_name, df in pandas_data.items():
        file_path = os.path.join(pandas_dir_path, file_name + '.csv')
        df.to_csv(file_path, index=False)

    # Saving scipy data
    for file_name, sps in scipy_data.items():
        file_path = os.path.join(scipy_dir_path, file_name + '.npz')
        sp.save_npz(file_path, sps)

    # Saving new_tids mapping
    tids_path = os.path.join(dir_path, 'new_tids.csv')
    new_tids.to_csv(tids_path, index=False)

    return pandas_dir_path, scipy_dir_path, tids_path


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pickle_load(file_path):
    with open(file_path, 'rb') as fh:
        return pickle.load(fh)


def pickle_dump(obj, file_path):
    with open(file_path, 'wb') as fh:
        return pickle.dump(obj, fh)


def json_load(file_path):
    with open(file_path, 'r') as fh:
        return json.load(fh)


def json_dump(obj, file_path):
    with open(file_path, 'w') as fh:
        return json.dump(obj, fh, indent=4)


def replace_special_path_chars(path):
    # replace any sort of characters that don't work on either Windows or Linux
    # special chars from https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
    special_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    path = path.replace(": ", '=')
    for c in special_chars:
        path = path.replace(c, '_')

    return path


def load_config(file_path):
    """
    Loads the configuration file and splits it to fixed parameters,
    and parameters to perform a grid search over.

    For better visibility, parameters should be structured in modules in the
    config file. Since grid search requires 1D dictionaries, the module name
    is appended to the actual parameter name and 1D dictionaries are created.

    :param file_path: The path to the config file
    :return: a tuple of (fixed parameters, grid search parameters)
    """
    config_dict = json_load(file_path)

    fixed_params = {}
    grid_search_params = {}
    for k in config_dict.keys():
        if "_search_" in k:
            prefix = k[:k.index("_search_params")]
            grid_search_params.update({f"{prefix}|{k}": v for k, v in config_dict[k].items()})
        else:
            prefix = k[:k.index("_params")]
            fixed_params.update({f"{prefix}|{k}": v for k, v in config_dict[k].items()})
    return fixed_params, grid_search_params


def modularize_config(conf: dict):
    """
    Brings a previously flattened dictionary (e.g., by load_config() above),
    back to its original module structure.
    :param conf: The config dict to modularize
    :return:
    """
    sc = defaultdict(lambda: dict())
    for k, v in conf.items():
        prefix, name = k.split("|", maxsplit=1)
        sc[prefix][name] = v
    return dict(sc)


def flatten_config(conf: dict):
    flattened_conf = dict()
    for k, v in conf.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                flattened_conf[k + "|" + k1] = v1
    return flattened_conf


def prepare_config(conf1, conf2, n_items):
    # join params together
    config = deepcopy(conf1)
    config.update(conf2)
    config = deepcopy(config)  # copy to prevent modification of the parameters
    config = modularize_config(config)

    # append the first (last) layer size for the VAE
    config["model"]["p_dims"] += [n_items]
    return config


def load_config_eval(config_file):
    config = json_load(config_file)
    use_adv_network = bool(c.get("in_use")) if (c := config.get("adv")) is not None else False

    return config, use_adv_network


def get_log_dir(dataset_name, experiment_type):
    now = datetime.now()
    log_base_str = LOG_DIR.replace("<dataset>", dataset_name)
    return log_base_str.format(experiment_type, now.strftime("%Y-%m-%d %H-%M-%S"))


def pretty_print(d: dict):
    print(json.dumps(d, indent="  "))
