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

class VECNNDataset(Dataset):
    def __init__(
        self,
        json_file,
        image_col='intph_img',
        input_idx=[1,37],
        output_idx=[37,67]):
        '''
        param json_file: path to the json dump of a pandas dataframe
        type json_file: str
        
        param image_col: the column name of the column that contains microstructure image file name in dataframe
        type image_col: str

        param input_idx: the column index range where the input parameters start and end.
        type index_idx: List(Int)

        param output_idx: the column index range where the output parameters start and end.
        tpye output_idx: List(Int)
        '''
        self.df = pd.read_json(json_file)
        self.len = len(self.df)
        self.image_col = image_col
        self.input_idx = input_idx
        self.output_idx = output_idx
        
    def __getitem__(self, index):
        item = {}
        item['image'] = torch.as_tensor(np.load(self.df.iloc[index][self.image_col])[np.newaxis,...],dtype=torch.float)
        item['input'] = torch.as_tensor(self.df.iloc[index][self.input_idx[0]:self.input_idx[1]].values,dtype=torch.float)
        item['output'] = torch.as_tensor(self.df.iloc[index][self.output_idx[0]:self.output_idx[1]].values,dtype=torch.float)
        return item

    def __len__(self):
        return self.len

# mean absolute percentage error used by Yixing
# MAPE = 1/N * sum_N{|(y_pred-y_true)/y_true|*100%}
def MAPELoss(output, target):
    loss = torch.mean(torch.abs((output - target)/target)*100)
    return loss
