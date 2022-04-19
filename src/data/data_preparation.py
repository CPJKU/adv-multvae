import os
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

from src.data.lfm_data_splitter import UserGroup


def split_interactions(interaction_matrix, test_size=0.8, random_state=42):
    user_idxs, movie_idxs, _ = sp.find(interaction_matrix == 1)

    tr_ind, te_ind = train_test_split(np.arange(len(user_idxs)), test_size=test_size, random_state=random_state)

    tr_values = np.ones(len(tr_ind))
    tr_matrix = sp.csr_matrix((tr_values, (user_idxs[tr_ind], movie_idxs[tr_ind])), shape=interaction_matrix.shape)

    te_values = np.ones(len(te_ind))
    te_matrix = sp.csr_matrix((te_values, (user_idxs[te_ind], movie_idxs[te_ind])), shape=interaction_matrix.shape)

    return tr_matrix, te_matrix


def perform_kfold_split(n_folds, interaction_matrix, df_user_info, storage_dir, random_state=42):
    print(f"Creating {n_folds} folds for cross validation")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # generate splits
    X = np.arange(interaction_matrix.shape[0])
    fold_indices = np.array([indices for _, indices in kf.split(X)])

    # Split data into different folds and store them to reduce compute time while training
    for i in range(n_folds):
        print("Splitting data for fold", i)

        fold_dir = os.path.join(storage_dir, str(i))
        os.makedirs(fold_dir, exist_ok=True)

        n_tr_folds = n_folds - 2
        tr_indices = np.concatenate(fold_indices[((np.arange(n_tr_folds) + i) % n_folds)])
        tr_indices = np.sort(tr_indices)  # ensure sorted indices to keep user - gender information

        tr_im = interaction_matrix[tr_indices, :]
        tr_user_info = df_user_info.iloc[tr_indices].copy()
        tr_user_info.reset_index(drop=True, inplace=True)
        tr_user_info["userID"] = tr_user_info.index

        tr_user_info.to_csv(os.path.join(fold_dir, "tr_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "sp_tr_data.npz"), tr_im)

        # over-sample data, as it may be used in some training procedures
        ros = RandomOverSampler(random_state=random_state)
        tr_im_os, y_os = ros.fit_resample(X=tr_im, y=tr_user_info["gender"])

        # generate new dataframe for oversampled data
        df_y_os = pd.DataFrame(y_os, columns=["gender"])
        df_y_os["userID"] = df_y_os.index
        df_y_os = df_y_os[["userID", "gender"]]  # switch order of columns

        df_y_os.to_csv(os.path.join(fold_dir, "up_tr_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "up_sp_tr_data.npz"), tr_im_os)

        # ===== Validation data ======
        vd_indices = np.sort(fold_indices[(n_tr_folds + i) % n_folds])
        vd_im = interaction_matrix[vd_indices, :]
        vd_tr_im, vd_te_im = split_interactions(vd_im, random_state=random_state)

        vd_user_info = df_user_info.iloc[vd_indices].copy()
        vd_user_info.reset_index(drop=True, inplace=True)
        vd_user_info["userID"] = vd_user_info.index

        vd_user_info.to_csv(os.path.join(fold_dir, "vd_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "sp_vd_tr_data.npz"), vd_tr_im)
        sp.save_npz(os.path.join(fold_dir, "sp_vd_te_data.npz"), vd_te_im)

        # ===== Test data ======
        te_indices = np.sort(fold_indices[(n_tr_folds + i + 1) % n_folds])
        te_im = interaction_matrix[te_indices, :]
        te_tr_im, te_te_im = split_interactions(te_im, random_state=random_state)

        te_user_info = df_user_info.iloc[te_indices].copy()
        te_user_info.reset_index(drop=True, inplace=True)
        te_user_info["userID"] = te_user_info.index

        te_user_info.to_csv(os.path.join(fold_dir, "te_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "sp_te_tr_data.npz"), te_tr_im)
        sp.save_npz(os.path.join(fold_dir, "sp_te_te_data.npz"), te_te_im)

        # VALIDATING
        # Ensure that no indices overlap between the different data sets
        n_indices_total = len(tr_indices) + len(vd_indices) + len(te_indices)
        all_indices = np.union1d(np.union1d(tr_indices, vd_indices), te_indices)

        print(f"Fold {i}: No indices overlap in the different data sets?", n_indices_total == len(all_indices))


def get_user_groups(storage_dir, trait, oversampled=False):
    if trait != "gender":
        raise AttributeError("Currently only the trait 'gender' is supported")

    tr_file_name = "up_tr_user_info.csv" if oversampled else "tr_user_info.csv"
    tr_user_info = pd.read_csv(os.path.join(storage_dir, tr_file_name))
    vd_user_info = pd.read_csv(os.path.join(storage_dir, "vd_user_info.csv"))
    te_user_info = pd.read_csv(os.path.join(storage_dir, "te_user_info.csv"))

    user_groups = []
    for g in ["m", "f"]:
        user_group = UserGroup("gender", g, None)
        user_group.tr_uids = tr_user_info.loc[tr_user_info["gender"] == g].index.to_numpy()
        user_group.tr_idxs = user_group.tr_uids

        user_group.vd_uids = vd_user_info.loc[vd_user_info["gender"] == g].index.to_numpy()
        user_group.vd_idxs = user_group.vd_uids

        user_group.te_uids = te_user_info.loc[te_user_info["gender"] == g].index.to_numpy()
        user_group.te_idxs = user_group.te_uids

        user_groups.append(user_group)

    return user_groups


def ensure_make_data(data_dir, n_folds, target_path, random_state=42):
    prev_state = None
    state_file = os.path.join(target_path, "used_state.txt")
    if os.path.exists(state_file):
        with open(state_file, "r") as fh:
            prev_state = int(fh.read())

    if random_state != prev_state:
        interaction_matrix = sp.load_npz(os.path.join(data_dir, "interactions.npz"))
        df_user_info = pd.read_csv(os.path.join(data_dir, "user_info.csv"))

        perform_kfold_split(n_folds, interaction_matrix, df_user_info,
                            storage_dir=target_path,
                            random_state=random_state)

        with open(state_file, "w") as fh:
            fh.write(str(random_state))


def prepare_fair_data(split, n_users, user_groups_per_trait):
    # Create DataFrame that contains information about the demographics of each user in current split
    df_id_traits = pd.DataFrame(np.arange(n_users), columns=["id"])

    traits = list(user_groups_per_trait.keys())
    for trait, user_groups in user_groups_per_trait.items():

        # create new column for trait
        df_id_traits = df_id_traits.assign(**{trait: None})

        for user_group in user_groups:
            idxs = user_group.get_split_indices(split)

            # assign value of trait to user
            df_id_traits.loc[idxs, trait] = user_group.name

        for k, v in df_id_traits[trait].value_counts().items():
            print(f"Trait '{trait}': Dataset contains {v} users with '{k}' attribute")

    df_id_traits = df_id_traits.drop(columns=["id"])  # id not needed anymore (used for construction purposes)

    # determine how many different attributes a trait has (e.g. for encoding purposes)
    counts_per_trait = {trait: len(df_id_traits[trait].value_counts()) for trait in traits}

    # create encoding matrix
    user_traits_encoding = np.empty(shape=(n_users, len(traits)))
    for i, trait in enumerate(traits):
        le = LabelEncoder()
        user_traits_encoding[:, i] = le.fit_transform(df_id_traits[trait])

    return df_id_traits, counts_per_trait, user_traits_encoding

