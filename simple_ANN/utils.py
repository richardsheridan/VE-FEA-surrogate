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
        item['input'] = torch.as_tensor(self.df.iloc[index,:self.index_in].values,dtype=torch.float)
        item['output'] = torch.as_tensor(self.df.iloc[index,self.index_in:].values,dtype=torch.float)
        return item

    def __len__(self):
        return self.len

class VEDatasetV2(Dataset):
    '''
    Upgraded dataset for VE response data with support for cross-feature scaling.
    '''
    def __init__(
        self,
        json_files,
        descriptor_dim=6,
        ve_dim=30,
        num_ve=1,
        scaling=False,
        ):
        '''
        param json_files: a list of path to the json dump of a pandas dataframe
        type json_files: List[str]
        
        param descriptor_dim: the # of columns that are the descriptor params,
            default to the leading 6 columns
        type descriptor_dim: int

        param ve_dim: the # of columns that reports the ve response, can
            be E', E", or tan delta, default to following the descriptor columns
            with value of 30
        type ve_dim: int

        param num_ve: the # of ve response type, if we are creating a dataset
            for tan delta, then it's 1, if we are creating a dataset for E' and
            tan delta, then it's 2, default to 1
        type num_ve: int

        '''
        self.df = pd.DataFrame()
        for file in json_files:
            self.df = pd.concat([self.df, pd.read_json(file)],ignore_index=True)
        self.len = len(self.df)
        # the column index where model inputs end
        self.index_in = descriptor_dim + num_ve*ve_dim
        self.descriptor_dim = descriptor_dim
        self.ve_dim = ve_dim
        self.num_ve = num_ve
        if scaling:
            self.df = self.scaling_df(self.df)

    def _scaling(self, df):
        '''
        Apply min-max scaling to ve cols and add scaling factors as a new col.
        '''
        df = df.copy()
        # generate max and min for each ve
        for i in range(self.num_ve):
            ve_in_start = self.descriptor_dim + i*self.ve_dim
            ve_in_end = ve_in_start + self.ve_dim
            ve_out_start = self.index_in + i*self.ve_dim
            ve_out_end = ve_out_start + self.ve_dim
            ve_max = df[df.columns[ve_in_start:ve_in_end]].max(axis=1)
            ve_min = df[df.columns[ve_in_start:ve_in_end]].min(axis=1)
            # scale input
            df[df.columns[ve_in_start:ve_in_end]] = df[df.columns[ve_in_start:ve_in_end]].apply(lambda x:(x-min_ve)/(max_ve-min_ve))
            # scale output
            df[df.columns[ve_out_start:ve_out_end]] = df[df.columns[ve_out_start:ve_out_end]].apply(lambda x:(x-min_ve)/(max_ve-min_ve))
            # save scale to df
            df[f'max_{i}'] = ve_max
            df[f'min_{i}'] = ve_min
        return df

    def __getitem__(self, index):
        item = {}
        item['input'] = torch.as_tensor(self.df.iloc[index,:self.index_in].values,dtype=torch.float)
        item['output'] = torch.as_tensor(self.df.iloc[index,self.index_in:self.index_in+self.num_ve*self.ve_dim].values,dtype=torch.float)
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
        item['image'] = torch.as_tensor(
            np.load(
                self.df.iloc[index][self.image_col]
                )[np.newaxis,...],
            dtype=torch.float)
        # df is now of mixed type, must cast back to float before convert to tensor
        item['input'] = torch.as_tensor(
            self.df.iloc[index][self.input_idx[0]:self.input_idx[1]]
            .astype('float')
            .to_numpy(),
            dtype=torch.float
            )
        item['output'] = torch.as_tensor(
            self.df.iloc[index][self.output_idx[0]:self.output_idx[1]]
            .astype('float')
            .to_numpy(),
            dtype=torch.float)
        return item

    def __len__(self):
        return self.len

# mean absolute percentage error used by Yixing
# MAPE = 1/N * sum_N{|(y_pred-y_true)/y_true|*100%}
def MAPELoss(output, target):
    loss = torch.mean(torch.abs((output - target)/target)*100)
    return loss
