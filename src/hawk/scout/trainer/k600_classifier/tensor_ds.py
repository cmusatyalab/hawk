import torch
from torch.utils.data import Dataset

class PTListDataset(Dataset):
    def __init__(self, list_file, transform=None, target_transform=None):
        # read all (path, label) pairs up front
        with open(list_file, "r") as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]
        self.samples = [(p, int(lbl)) for p, lbl in lines]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label