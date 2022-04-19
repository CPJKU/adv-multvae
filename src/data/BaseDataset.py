import os
from scipy import sparse
from torch.utils.data import Dataset

split_filename_dict = {
    "inputs": {
        "train": "sp_tr_data.npz",
        "val": "sp_vd_tr_data.npz",
        "test": "sp_te_tr_data.npz"
    },
    "targets": {
        "train": None,
        "val": "sp_vd_te_data.npz",
        "test": "sp_te_te_data.npz"
    }
}


class BaseDataset(Dataset):
    """
    Base dataset class that all datasets should build upon
    """

    def __init__(self, data_dir, split='train', transform=None, up_sample=False):
        super().__init__()

        self.which = split
        self.data_dir = data_dir
        self.transform = transform
        self.up_sample = up_sample
        self.is_train_set = split == "train"

        # Determine input file and load data
        inputs_file_name = split_filename_dict["inputs"][split]
        if self.is_train_set and self.up_sample:
            inputs_file_name = "up_" + inputs_file_name
        self.data = sparse.load_npz(os.path.join(self.data_dir, inputs_file_name))

        # Determine target file and load data
        targets_file_name = split_filename_dict["targets"][split]

        if self.is_train_set:
            # During training, we want to recreate the input
            self.targets = self.data
        else:
            self.targets = sparse.load_npz(os.path.join(self.data_dir, targets_file_name))

        self.n_users = self.data.shape[0]
        self.n_items = self.data.shape[1]
        self.__ensure_types()

    def __len__(self):
        return self.n_users

    def __ensure_types(self):
        self.data = self.data.astype("float32")
        self.targets = self.targets.astype("float32")

    def __getitem__(self, idx):
        x_sample = self.data[idx, :].toarray().squeeze()
        if self.transform:
            x_sample = self.transform(x_sample)

        y_sample = self.targets[idx, :].toarray().squeeze()

        return x_sample, y_sample
