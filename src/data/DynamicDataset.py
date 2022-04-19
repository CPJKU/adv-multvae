from torch.utils.data import Dataset
from src.data.BaseDataset import BaseDataset
from copy import deepcopy


class DynamicFeedbackDataset(Dataset):
    def __init__(self, dataset):
        assert(isinstance(dataset, BaseDataset))
        # to not mess up any other experiments, we store an actual copy of the dataset
        self.dataset = deepcopy(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def include_feedback(self, feedback):
        """
        Includes users feedback into the dataset
        feedback ... list of indices for items that the user just interacted with
        """
        added_feedback = []
        for i, f in enumerate(feedback):
            added_feedback.append(f[(self.dataset.data[i, f] != 1).toarray().flatten()])
            self.dataset.data[i, f] = 1
        return added_feedback

