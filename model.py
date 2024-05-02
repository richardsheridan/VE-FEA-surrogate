import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleANN(nn.Module):
    def __init__(self, input_dim=36, hidden_1=64, hidden_2=64, output_dim=30,
        dropout=0.2, activation='gelu', wide_and_deep_index=None, **kwargs):
        # wide_and_deep_index indicates which input features are for wide NN
        super().__init__()
        self.wide_and_deep_index = wide_and_deep_index
        # number of hidden nodes in each layer (64)
        # hidden_1 = 64
        # hidden_2 = 64
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise Exception("activation choices are limited to 'gelu','relu','leakyrelu','tanh'.")
        self.linear_relu_stack_with_dropout = nn.Sequential(
            nn.Linear(input_dim, hidden_1), # linear layer (36 -> hidden_1)
            self.activation,
            nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
            nn.Linear(hidden_1,hidden_2), # linear layer (n_hidden -> hidden_2)
            self.activation,
            nn.Dropout(dropout), # dropout layer (p=0.2), dropout prevents overfitting of data
        )
        if wide_and_deep_index:
            self.last_hidden = nn.Linear(len(wide_and_deep_index)+hidden_2,
                                         output_dim) # linear layer (n_hidden -> 10)
        else:
            self.last_hidden = nn.Linear(hidden_2,output_dim) # linear layer (n_hidden -> 10)

    def forward(self,x):
        deep = self.linear_relu_stack_with_dropout(x)
        if self.wide_and_deep_index:
            wide = x[...,self.wide_and_deep_index]
            logits = self.last_hidden(torch.cat((wide,deep), -1))
        else:
            logits = self.last_hidden(deep)
        return logits

class SplitANN(nn.Module):
    def __init__(self, descriptor_dim=6, input_split_dim=30, hidden_1=128,
        hidden_2=128, output_split_dim=30, dropout=0.2, activation='gelu',
        wide_and_deep_index=None):
        super().__init__()
        self.top_half = SimpleANN(
            input_dim=descriptor_dim+input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_split_dim,
            dropout=dropout,
            activation=activation,
            wide_and_deep_index=wide_and_deep_index,
        )
        self.btm_half = SimpleANN(
            input_dim=descriptor_dim+input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_split_dim,
            dropout=dropout,
            activation=activation,
            wide_and_deep_index=wide_and_deep_index,
        )
        self.descriptor_dim = descriptor_dim
        self.input_split_dim = input_split_dim
        
    def forward(self,x):
        # x is of size batch_size*input_dim
        x_top = x[...,:self.descriptor_dim + self.input_split_dim] # x[:,:36]
        x_btm = torch.cat(
            (
                x[...,:self.descriptor_dim],
                x[...,self.descriptor_dim+self.input_split_dim:self.descriptor_dim+self.input_split_dim*2]
            ),
            -1
        ) # torch.cat((x[:,:6],x[:,36:66]),1)
        logits_top = self.top_half(x_top)
        logits_btm = self.btm_half(x_btm)
        return torch.cat((logits_top,logits_btm),-1)


class CNN(nn.Module):
    def __init__(
        self,
        conv_kernel_size=3,
        stride=1,
        padding=1,
        input_ch_1=1,
        output_ch_1=16,
        output_ch_2=32,
        output_ch_3=64,
        pool_kernel_size=2,
        image_dim=500,
        embedding_dim=32,
        ):
        super().__init__()
        # (batch_size,1,500,500)
        self.conv1 = nn.Conv2d(input_ch_1, output_ch_1,
            kernel_size=conv_kernel_size, stride=stride, padding=padding)
        # (batch_size,16,500,500) or (N_in,C_in,H_in,W_in)
        # H_out = (H_in-dilation*(kernel_size-1)+2*padding-1)/stride+1
        # dilation default to 1
        # (500-3+2*1)/1+1=500
        H_out = (image_dim-(conv_kernel_size-1)+2*padding-1)//stride+1
        self.bn1 = nn.BatchNorm2d(output_ch_1)
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_kernel_size)
        # (batch_size,16,250,250)
        H_out = H_out//pool_kernel_size
        self.conv2 = nn.Conv2d(output_ch_1, output_ch_2,
            kernel_size=conv_kernel_size, stride=stride, padding=padding)
        # (batch_size,32,250,250)
        # (250-3+2*1)/1+1=250
        H_out = (H_out-(conv_kernel_size-1)+2*padding-1)//stride+1
        self.bn2 = nn.BatchNorm2d(output_ch_2)
        # after pooling (batch_size,32,125,125)
        H_out = H_out//pool_kernel_size
        self.conv3 = nn.Conv2d(output_ch_2, output_ch_3,
            kernel_size=conv_kernel_size, stride=stride, padding=padding)
        # (batch_size,64,125,125)
        H_out = (H_out-(conv_kernel_size-1)+2*padding-1)//stride+1
        self.bn3 = nn.BatchNorm2d(output_ch_3)
        # after pooling (batch_size,64,62,62)
        H_out = H_out//pool_kernel_size
        self.fc1 = nn.Linear(output_ch_3*H_out*H_out, embedding_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        return x

class SimpleANN_CNN(nn.Module):
    def __init__(
        self,
        descriptor_dim=6,
        input_split_dim=30,
        hidden_1=64,
        hidden_2=64,
        output_split_dim=30,
        dropout=0.2,
        conv_kernel_size=3,
        stride=1,
        padding=1,
        input_ch_1=1,
        output_ch_1=16,
        output_ch_2=32,
        output_ch_3=64,
        pool_kernel_size=2,
        image_dim=500,
        embedding_dim=32,
        ):
        super().__init__()
        self.CNN_upstream = CNN(
            conv_kernel_size=3,
            stride=1,
            padding=1,
            input_ch_1=1,
            output_ch_1=16,
            output_ch_2=32,
            output_ch_3=64,
            pool_kernel_size=2,
            image_dim=500,
            embedding_dim=32
            )
        self.ANN_downstream = SimpleANN(
            input_dim=embedding_dim+descriptor_dim+input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_split_dim,
            dropout=dropout
            )
        
    def forward(self,x,y):
        # x is for image tensor input
        # y is for descriptor and master curve input
        image_embedding = self.CNN_upstream(x)
        merged_x = torch.cat((image_embedding,y),-1)
        logits = self.ANN_downstream(merged_x)
        return logits
