from src.data.BaseDataset import BaseDataset
from src.data.data_preparation import prepare_fair_data, get_user_groups
from src.data.lfm_data_splitter import DataSplitter


class BaseFairDataset(BaseDataset):
    """
    Base fairness dataset class that all fair datasets should build upon
    """

    def __init__(self, data_dir, split='train', transform=None, up_sample=False, traits=("gender",)):
        super().__init__(data_dir, split, transform, up_sample)
        self.traits = traits
        self.df_id_traits = None
        self.counts_per_trait = None
        self.user_traits_encoding = None
        self.user_groups_per_trait = None

    def __getitem__(self, idx):
        x_sample, y_sample = super().__getitem__(idx)
        trait_encoding = self.user_traits_encoding[idx, :]
        return x_sample, y_sample, trait_encoding


class FairLFM2bDataset(BaseFairDataset):
    """
    Last.fm-2b dataset with additional demographic traits
    """

    def __init__(self, data_dir, demo_dir, data_splitter: DataSplitter,
                 split='train', transform=None, up_sample=False, traits=("gender",)):
        super().__init__(data_dir=data_dir, split=split, transform=transform,
                         up_sample=up_sample, traits=traits)

        self.user_groups_per_trait = {
            trait: data_splitter.get_user_groups_indices(demo_dir, trait, up_sample) for trait in traits
        }
        self.df_id_traits, self.counts_per_trait, self.user_traits_encoding = prepare_fair_data(split,
                                                                                                self.n_users,
                                                                                                self.user_groups_per_trait)


class FairMovieLensDataset(BaseFairDataset):
    """
    MovieLens dataset with additional demographic traits
    """

    def __init__(self, data_dir, split='train', transform=None, up_sample=False, traits=("gender",)):
        super().__init__(data_dir=data_dir, split=split, transform=transform,
                         up_sample=up_sample, traits=traits)

        self.user_groups_per_trait = {
            trait: get_user_groups(storage_dir=data_dir, trait=trait, oversampled=up_sample) for trait in traits
        }
        self.df_id_traits, self.counts_per_trait, self.user_traits_encoding = prepare_fair_data(split,
                                                                                                self.n_users,
                                                                                                self.user_groups_per_trait)
