import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleANN(nn.Module):
    def __init__(self, input_dim=36, hidden_1=64, hidden_2=64, output_dim=30,
        dropout=0.2):
        super().__init__()
        # number of hidden nodes in each layer (64)
        # hidden_1 = 64
        # hidden_2 = 64
        self.linear_relu_stack_with_dropout = nn.Sequential(
            nn.Linear(input_dim, hidden_1), # linear layer (36 -> hidden_1)
            nn.ReLU(), # ReLU activation function
            nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
            nn.Linear(hidden_1,hidden_2), # linear layer (n_hidden -> hidden_2)
            nn.ReLU(), # ReLU activation function
            nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
            nn.Linear(hidden_2,output_dim) # linear layer (n_hidden -> 10)
        )
        
    def forward(self,x):
        logits = self.linear_relu_stack_with_dropout(x)
        return logits

class SplitANN(nn.Module):
    def __init__(self, descriptor_dim=6, input_split_dim=30, hidden_1=128,
        hidden_2=128, output_split_dim=30, dropout=0.2):
        super().__init__()
        self.ep_half = SimpleANN(
            input_dim=descriptor_dim+input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_split_dim,
            dropout=dropout
        )
        self.epp_half = SimpleANN(
            input_dim=descriptor_dim+input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_split_dim,
            dropout=dropout
        )
        self.descriptor_dim = descriptor_dim
        self.input_split_dim = input_split_dim
        
    def forward(self,x):
        # x is of size batch_size*input_dim
        x_ep = x[:,:self.descriptor_dim + self.input_split_dim] # x[:,:36]
        x_epp = torch.cat((x[:,:self.descriptor_dim],x[:,self.descriptor_dim+self.input_split_dim:]),1) # torch.cat((x[:,:6],x[:,36:]),1)
        logits_ep = self.ep_half(x_ep)
        logits_epp = self.epp_half(x_epp)
        return torch.cat((logits_ep,logits_epp),1)


# class HalfMLP(nn.Module):
#     def __init__(self, descriptor_dim=6, input_split_dim=30, hidden_1=128,
#         hidden_2=128, output_split_dim=30, dropout=0.2):
#         super().__init__()
#         self.linear_relu_stack_with_dropout = nn.Sequential(
#             nn.Linear(descriptor_dim+input_split_dim, hidden_1), # linear layer (36 -> hidden_1)
#             nn.ReLU(), # ReLU activation function
#             nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
#             nn.Linear(hidden_1,hidden_2), # linear layer (hidden_1 -> hidden_2)
#             nn.ReLU(), # ReLU activation function
#             nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
#             nn.Linear(hidden_2,output_split_dim) # linear layer (hidden_2 -> 30)
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack_with_dropout(x)
#         return logits