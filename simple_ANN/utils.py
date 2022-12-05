from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class VEDataset(Dataset):
    def __init__(self, json_file, index_in = 36):
        '''
        param json_file: path to the json dump of a pandas dataframe
        type json_file: str
        
        param index_in: the column index where the input parameters end, i.e. input dimension, default to 36
        type index_in: int
        '''
        self.df = pd.read_json(json_file)
        self.len = len(self.df)
        self.index_in = index_in
        
    def __getitem__(self, index):
        item = {}
        item['input'] = torch.as_tensor(self.df.iloc[index][:self.index_in].values,dtype=torch.float)
        item['output'] = torch.as_tensor(self.df.iloc[index][self.index_in:].values,dtype=torch.float)
        return item

    def __len__(self):
        return self.len