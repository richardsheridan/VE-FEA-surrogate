import torch.nn as nn
import torch.nn.functional as F

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