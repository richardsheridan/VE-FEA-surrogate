from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

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
        ignore_columns=['intph_img', 'percolation'],
        **kwargs,
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
        # drop columns specified in `ignore_columns`
        self.df = self.df.drop(columns=ignore_columns)
        # the column index where model inputs end
        self.index_in = descriptor_dim + num_ve*ve_dim
        self.descriptor_dim = descriptor_dim
        self.ve_dim = ve_dim
        self.num_ve = num_ve
        self.scaling_factor = {}
        if scaling:
            self.df = self.scaling_df(self.df)
        
    def scaling_df(self, df):
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
            df[df.columns[ve_in_start:ve_in_end]] = df[df.columns[ve_in_start:ve_in_end]].apply(lambda x:(x-ve_min)/(ve_max-ve_min))
            # scale output
            df[df.columns[ve_out_start:ve_out_end]] = df[df.columns[ve_out_start:ve_out_end]].apply(lambda x:(x-ve_min)/(ve_max-ve_min))
            # save scale to df
            df[f'max_{i}'] = ve_max
            df[f'min_{i}'] = ve_min
            # save scale to class
            self.scaling_factor[i] = {'max':ve_max, 'min':ve_min}
        return df

    def __getitem__(self, index):
        item = {}
        item['input'] = torch.as_tensor(self.df.iloc[index,:self.index_in].values,dtype=torch.float)
        item['output'] = torch.as_tensor(self.df.iloc[index,self.index_in:self.index_in+self.num_ve*self.ve_dim].values,dtype=torch.float)
        return item

    def __len__(self):
        return self.len

    def scale_back(self, prediction, ve_id=0):
        assert (ve_id >= 0 and ve_id < self.num_ve), "0 <= ve_id < num_ve is required"
        min_matrix = self.scaling_factor[ve_id]['min'].values.reshape(-1,1).repeat(self.ve_dim,axis=1)
        max_matrix = self.scaling_factor[ve_id]['max'].values.reshape(-1,1).repeat(self.ve_dim,axis=1)
        # prediction must be an array or tensor
        return prediction*(max_matrix-min_matrix)+min_matrix


# mean absolute percentage error used by Yixing
# MAPE = 1/N * sum_N{|(y_pred-y_true)/y_true|*100%}
def MAPELoss(output, target):
    loss = torch.mean(torch.abs((output - target)/target)*100)
    return loss
