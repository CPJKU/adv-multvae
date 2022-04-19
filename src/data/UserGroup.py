import numpy as np


class UserGroup:
    """
    Holds information of a specific user group such as name of the group, the user ids, and the postional index
    """

    def __init__(self, type: str, name: str, uids: np.ndarray):
        self.type = type
        self.name = name
        self.uids = uids
        self.tr_uids = None
        self.vd_uids = None
        self.te_uids = None
        self.tr_idxs = None
        self.vd_idxs = None
        self.te_idxs = None

    def get_split_indices(self, split="train"):
        if split == "train":
            return self.tr_idxs
        elif split == "val":
            return self.vd_idxs
        elif split == "test":
            return self.te_idxs
        else:
            raise Exception("Dataset string entered is not valid! Please choose from [train,val,test]")
