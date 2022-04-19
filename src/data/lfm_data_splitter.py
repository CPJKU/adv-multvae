import os
import pickle

import numpy as np
import pandas as pd
from scipy import sparse as sp

from src.data.UserGroup import UserGroup
from src.utils.helper import idx_split, filt, random_item_splitter, playcounts, permute, sparsify, save_data, mod_split


class DataSplitter:
    """
    Class used to generated the dataset in the following steps:
    - Splits the users in training (80%), validation (10%), and test (10%) or using cross validation with 5 folds.
    - Filters the data (tracks not in the training data are discarded, users with fewer than 5 LEs are also removed)
    - Generates the playcount for each user
    - For validation and test users, the tracks listened are partitioned in training items (80%) and test items (20%)

    """
    INF_STR = "{:10d} entries {:7d} users {:7d} items for {} data"
    LST_STR = "{:10d} entries {:7d} users {:7d} items (lost)"
    DIR_NAME = os.path.join("{}", "{}")
    DIR_TR_NAME = os.path.join("{}", "{}", "{}")

    def __init__(self, data_path, demo_path=None, out_dir='../../data/', perc_train=80, cv_n_folds=5):
        """
        :param data_path: path to the file containing interactions user_id, track_id
        :param demo_path: path to the demographic information of the users such as gender, country, and age
        :param out_dir: path where the generated dataset is saved
        :param perc_train: % of training users
        :param cv_n_folds: number of folds in the cross_validation
        """
        self.data_path = data_path
        self.demo_path = demo_path

        self.inter = pd.read_csv(self.data_path, sep='\t', names=['user_id', 'track_id', 'play_count'])[
            ['user_id', 'track_id']]

        if self.demo_path:
            self.demo = pd.read_csv(self.demo_path, sep='\t',
                                    names=['user_name', 'country', 'age', 'gender', 'timestamp'])

        self.out_dir = out_dir

        self.n_users = self.inter.user_id.nunique()
        self.n_items = self.inter.track_id.nunique()

        self.n_train = (self.n_users * perc_train) // 100
        self.n_heldout = (self.n_users - self.n_train) // 2
        self.cv_n_folds = cv_n_folds

    def _split(self, tr_uids: np.ndarray, vd_uids: np.ndarray, te_uids: np.ndarray):
        """
        Internal method for the splitting procedure
        """
        # Extract data
        tr_data = self.inter[self.inter.user_id.isin(tr_uids)]
        print(DataSplitter.INF_STR.format(len(tr_data), tr_data.user_id.nunique(), tr_data.track_id.nunique(),
                                          "Training"))
        # Only tracks in the training data are considered
        tids = tr_data.track_id.drop_duplicates().values
        self.n_items = len(tids)

        vd_data = self.inter[self.inter.user_id.isin(vd_uids)]
        vd_data, num_, usr_, its_ = filt(vd_data, tids)
        print(DataSplitter.INF_STR.format(len(vd_data), vd_data.user_id.nunique(), vd_data.track_id.nunique(),
                                          "Validation"))
        print(DataSplitter.LST_STR.format(num_ - len(vd_data), usr_ - vd_data.user_id.nunique(),
                                          its_ - vd_data.track_id.nunique()))

        te_data = self.inter[self.inter.user_id.isin(te_uids)]
        te_data, num_, usr_, its_ = filt(te_data, tids)
        print(DataSplitter.INF_STR.format(len(te_data), te_data.user_id.nunique(), te_data.track_id.nunique(),
                                          "Test"))
        print(DataSplitter.LST_STR.format(num_ - len(te_data), usr_ - te_data.user_id.nunique(),
                                          its_ - te_data.track_id.nunique()))

        # Re-indexing for the track_id
        new_tids = tr_data.track_id.drop_duplicates().sort_values().reset_index(drop=True).reset_index().rename(
            columns={"index": "new_track_id"})

        tr_data = tr_data.merge(new_tids).drop(columns="track_id")
        vd_data = vd_data.merge(new_tids).drop(columns="track_id")
        te_data = te_data.merge(new_tids).drop(columns="track_id")

        # Generates playcounts
        tr_data = playcounts(tr_data)
        vd_data = playcounts(vd_data)
        te_data = playcounts(te_data)

        # Splitting item data
        vd_tr_data, vd_te_data = random_item_splitter(vd_data)
        te_tr_data, te_te_data = random_item_splitter(te_data)

        sp_tr_data, tr_data = sparsify(tr_data, self.n_items)
        sp_vd_data, vd_data = sparsify(vd_data, self.n_items)
        sp_te_data, te_data = sparsify(te_data, self.n_items)
        sp_vd_tr_data, vd_tr_data = sparsify(vd_tr_data, self.n_items)
        sp_vd_te_data, vd_te_data = sparsify(vd_te_data, self.n_items)
        sp_te_tr_data, te_tr_data = sparsify(te_tr_data, self.n_items)
        sp_te_te_data, te_te_data = sparsify(te_te_data, self.n_items)

        pandas_data = {
            'tr_data': tr_data,
            'vd_data': vd_data,
            'vd_tr_data': vd_tr_data,
            'vd_te_data': vd_te_data,
            'te_data': te_data,
            'te_tr_data': te_tr_data,
            'te_te_data': te_te_data
        }

        scipy_data = {
            'sp_tr_data': sp_tr_data,
            'sp_vd_data': sp_vd_data,
            'sp_vd_tr_data': sp_vd_tr_data,
            'sp_vd_te_data': sp_vd_te_data,
            'sp_te_data': sp_te_data,
            'sp_te_tr_data': sp_te_tr_data,
            'sp_te_te_data': sp_te_te_data,
        }

        return pandas_data, scipy_data, new_tids

    def sample_split(self, seed: int):
        """
        Users are sampled at random for the split sets.
        """
        self.out_dir = self.out_dir.format('seed')
        np.random.seed(seed)

        # Extract user_ids
        uids = self.inter.user_id.drop_duplicates().values

        # Permute array
        uids = permute(uids)

        # Split user ids
        tr_uids, vd_uids, te_uids = idx_split(uids, self.n_train, self.n_heldout)

        pandas_data, scipy_data, new_tids = self._split(tr_uids, vd_uids, te_uids)

        # Saving data
        dir_name = DataSplitter.DIR_NAME.format(os.path.basename(self.data_path).split('.')[0], seed)
        dir_path = os.path.join(self.out_dir, dir_name)
        pandas_dir_path, scipy_dir_path, tids_path = save_data(dir_path, pandas_data, scipy_data, new_tids)

        # Saving uids
        uids_dic = {
            'tr_uids': tr_uids,
            'vd_uids': vd_uids,
            'te_uids': te_uids
        }
        uids_dic_path = os.path.join(dir_path, 'uids_dic.pkl')
        pickle.dump(uids_dic, open(uids_dic_path, 'wb'))

        return pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path

    def cv_split(self, fold_n: int):
        """
        Users are split according to the fold number.
        """
        self.out_dir = self.out_dir.format('fold_n')
        # Extract user_ids
        uids = self.inter.user_id.drop_duplicates().values

        # Split user ids
        tr_uids, vd_uids, te_uids = mod_split(uids, fold_n, self.cv_n_folds)

        pandas_data, scipy_data, new_tids = self._split(tr_uids, vd_uids, te_uids)

        # Saving data
        dir_name = DataSplitter.DIR_NAME.format(os.path.basename(self.data_path).split('.')[0], fold_n)
        dir_path = os.path.join(self.out_dir, dir_name)
        pandas_dir_path, scipy_dir_path, tids_path = save_data(dir_path, pandas_data, scipy_data, new_tids)

        # Saving uids
        uids_dic = {
            'tr_uids': tr_uids,
            'vd_uids': vd_uids,
            'te_uids': te_uids
        }
        uids_dic_path = os.path.join(dir_path, 'uids_dic.pkl')
        pickle.dump(uids_dic, open(uids_dic_path, 'wb'))

        return pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path

    def get_paths(self, fold_n: int = None, seed: int = None):
        """
        Returns the dataset given a seed or the fold number

        :param fold_n: fold number should be in {0, ..., self.cv_n_folds}
        :param seed: random seed for the splits
        :return: paths to the data
        """
        dir_name = DataSplitter.DIR_NAME.format(os.path.basename(self.data_path).split('.')[0],
                                                seed if seed else fold_n)

        dir_path = os.path.join(self.out_dir.format('seed' if seed is not None else 'fold_n'), dir_name)

        # Assuming that if the main dir exists, all files will be present
        if os.path.isdir(dir_path):
            pandas_dir_path = os.path.join(dir_path, "pandas")
            scipy_dir_path = os.path.join(dir_path, "scipy")
            uids_dic_path = os.path.join(dir_path, 'uids_dic.pkl')
            tids_path = os.path.join(dir_path, 'new_tids.csv')
        else:
            print("Data not found, generating new split")
            if seed is not None:
                print("Seed: {:10d}".format(seed))
                pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path = self.sample_split(seed)
            elif fold_n is not None:
                print("Fold number: {:10d}".format(fold_n))
                pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path = self.cv_split(fold_n)
            else:
                raise ValueError('Either seed or fold_n should be not None!')

        # Reads new_tids.csv in order to update the number of items
        self.n_items = len(pd.read_csv(tids_path)['new_track_id'])
        return pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path

    def get_user_groups(self, demo_trait: str):
        if not hasattr(self, "demo"):
            raise Exception("No path to user demographic file!")

        # Split by gender
        if demo_trait == 'gender':
            # User ids are encoded by position
            m_uids = self.demo[self.demo.gender == 'm'].index.values
            f_uids = self.demo[self.demo.gender == 'f'].index.values
            return [UserGroup(demo_trait, 'm', m_uids), UserGroup(demo_trait, 'f', f_uids)]
        else:
            raise ValueError('Demographic trait not yet implemented')

    def get_user_groups_indices(self, pandas_dir_path: str, demo_trait=None, up_sample=False):
        tr_data = pd.read_csv(os.path.join(pandas_dir_path, 'tr_data.csv' if not up_sample else 'up_tr_data.csv'))[
            ['user_id', 'new_user_id']].drop_duplicates()
        vd_data = pd.read_csv(os.path.join(pandas_dir_path, 'vd_data.csv'))[
            ['user_id', 'new_user_id']].drop_duplicates()
        te_data = pd.read_csv(os.path.join(pandas_dir_path, 'te_data.csv'))[
            ['user_id', 'new_user_id']].drop_duplicates()

        user_groups = self.get_user_groups(demo_trait)
        for user_group in user_groups:
            user_group.tr_uids = tr_data[tr_data.user_id.isin(set(user_group.uids))].user_id.values
            user_group.vd_uids = vd_data[vd_data.user_id.isin(set(user_group.uids))].user_id.values
            user_group.te_uids = te_data[te_data.user_id.isin(set(user_group.uids))].user_id.values
            user_group.tr_idxs = tr_data[tr_data.user_id.isin(set(user_group.uids))].new_user_id.values
            user_group.vd_idxs = vd_data[vd_data.user_id.isin(set(user_group.uids))].new_user_id.values
            user_group.te_idxs = te_data[te_data.user_id.isin(set(user_group.uids))].new_user_id.values

        return user_groups

    def up_sample_train_data_path(self, pandas_dir_path: str, scipy_dir_path: str, demo_trait=None):
        """
        Upsamples train data by adding records to tr_data.csv and rows to sp_tr_data.npz
        :param demo_trait:
        :param pandas_dir_path:
        :param scipy_dir_path:
        :return:
        """
        up_tr_data_path = os.path.join(pandas_dir_path, 'up_tr_data.csv')
        up_sp_tr_data_path = os.path.join(scipy_dir_path, 'up_sp_tr_data.npz')

        if os.path.isfile(up_tr_data_path) and os.path.isfile(up_sp_tr_data_path):
            # Already computed
            return up_tr_data_path, up_sp_tr_data_path

        print('Upsampled data not found - Running Upsampling')

        tr_data = pd.read_csv(os.path.join(pandas_dir_path, 'tr_data.csv'))
        sp_tr_data = sp.load_npz(os.path.join(scipy_dir_path, 'sp_tr_data.npz'))

        # Note: In case of other demographic traits, use imblearn's RandomOverSampler instead!
        if demo_trait != 'gender':
            raise ValueError('Upsampling not yet implemented for other traits!')

        male_group, female_group = self.get_user_groups_indices(pandas_dir_path, demo_trait)

        # ASSUMING THAT FEMALE IS THE MINORITY GROUP
        n_samples = len(male_group.tr_uids) - len(female_group.tr_uids)
        assert n_samples > 0, 'Number of samples for Upsampling is negative or 0. ' \
                              'Are you sure you are using the right user groups?'

        # Sampling randomly with replacement uids from the specified user_group
        chosen = np.random.choice(np.arange(len(female_group.tr_uids)), n_samples, replace=True)
        chosen_uids = female_group.tr_uids[chosen]
        chosen_idxs = female_group.tr_idxs[chosen]

        # -- Updating tr_data.csv -- #
        uids, counts = np.unique(chosen_uids, return_counts=True)
        dup = pd.DataFrame({'user_id': uids, 'repeat': counts})

        # Duplicate by index
        dup = dup.set_index('user_id')['repeat']
        dup = dup.loc[dup.index.repeat(dup)]
        # Removing not used_information
        dup = dup.reset_index().drop(columns='repeat')

        # Assiging a new user_id to the duplicated users
        dup['new_user_id'] = range(tr_data.new_user_id.nunique(), tr_data.user_id.nunique() + len(dup))

        # Fetching the user_ids from the original dataset and merging with the duplicates.
        up_uids = pd.concat([tr_data[['user_id', 'new_user_id']].drop_duplicates(), dup])

        # Finally, duplicating the interaction data
        up_tr_data = pd.merge(tr_data.drop(columns='new_user_id'), up_uids, how='left', on='user_id').sort_values(
            'new_user_id')

        up_data = sp_tr_data[chosen_idxs, :]
        up_sp_tr_data = sp.csr_matrix(sp.vstack((sp_tr_data, up_data)))

        up_tr_data.to_csv(up_tr_data_path, index=False)
        sp.save_npz(up_sp_tr_data_path, up_sp_tr_data)

        return up_tr_data_path, up_sp_tr_data_path
