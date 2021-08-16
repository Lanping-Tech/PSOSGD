from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DealDataset(Dataset):

    def __init__(self):
        x = datasets.load_diabetes()['data']
        y = np.expand_dims(preprocessing.scale(np.array(datasets.load_diabetes()['target'])), axis=1)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def load(batch_size, shuffle=True):
    return DataLoader(DealDataset(), batch_size=batch_size, shuffle=shuffle)
