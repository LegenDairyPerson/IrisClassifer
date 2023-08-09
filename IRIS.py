import pandas as pd
from torch.utils.data import Dataset
import torch

class datasetIRIS(Dataset):
    def __init__(self, file_path, transforms = None):
        self.data = pd.read_csv(file_path)
        self.transforms = transforms
        self.labels = {
            "Iris-setosa" : 0,
            "Iris-versicolor" : 1,
            "Iris-virginica" : 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index, 1:5])
        label = str(self.data.iloc[index, 5])
        if self.transforms is not None:
            data = self.transforms(data)
        # print(data, self.labels[label])
        return data, self.labels[label]