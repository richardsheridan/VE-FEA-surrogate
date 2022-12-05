import torch
import torch.nn as nn
from torch.utils.data import DataLoader,SubsetRandomSampler
from utils import VEDataset
import numpy as np
from torch.optim import Adam
from model import SimpleANN
from tqdm import tqdm
import json

# seed
seed = 27

# input dimension
input_dim = 36

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# load model
model = SimpleANN().to(device)
print(model)

# save model to this file
save_to = 'hs1-64_hs2-64_ep-100_bs-32_model'

# number of training epochs
EPOCHS = 100
# how many samples per batch to load
BATCH_SIZE = 32
# number of subprocesses to use for data loading
NUM_WORKERS = 0
# percentage of training set to use as validation
VALID_FRACTION = 0.2
# learning rate
LR = 0.001

# Datasets and Generators
train_data = VEDataset('../full_train.json', index_in = input_dim)
test_data = VEDataset('../full_test.json', index_in = input_dim)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.seed(seed)
np.random.shuffle(indices)
split = int(np.floor(VALID_FRACTION * num_train))
train_index, valid_index = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

# prepare data loaders
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, 
                          sampler = train_sampler, num_workers = NUM_WORKERS)
valid_loader = DataLoader(train_data, batch_size = BATCH_SIZE,
                          sampler = valid_sampler, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE,
                         num_workers = NUM_WORKERS)

# define the train process
def train(model, x, y, optimizer, criterion):
    # clear the gradients of all optimized variables
    model.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(x)
    # calculate the loss
    loss =criterion(output,y)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()

    return loss, output

# define the loss function, use mean square error loss
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = LR)

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity

# init history
history = {'train_loss':[], 'valid_loss':[], 'test_loss':None}

# start training
for epoch in range(EPOCHS):
    # monitor losses
    train_loss = 0
    valid_loss = 0
    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for batch in tqdm(train_loader):
        x_train, y_train = batch['input'], batch['output']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(model,x_train,y_train,optimizer,criterion)
        train_loss += loss.item() * x_train.size(0)

    ######################    
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    for batch in tqdm(valid_loader):
        x_val, y_val = batch['input'], batch['output']
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(x_val)
        # calculate the loss
        loss = criterion(output,y_val)
        # update running validation loss 
        valid_loss = loss.item() * x_val.size(0)
    
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    # save loss to history
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), f'{save_to}.pt')
        valid_loss_min = valid_loss


# evaluate on test dataset
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
# load the best model
model.load_state_dict(torch.load(f'{save_to}.pt'))
model.eval() # prep model for evaluation
for batch in tqdm(test_loader):
    x_test, y_test = batch['input'], batch['output']
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(x_test)
    # calculate the loss
    loss = criterion(output,y_test)
    # update running validation loss 
    test_loss = loss.item() * x_test.size(0)
    
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))
# save to history
history['test_loss'] = test_loss

# save history to json
with open(f'{save_to}_history.json','w') as f:
    json.dump(history, f)
    print(f'history saved to {save_to}_history.json')

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_csv_path")
#     parser.add_argument("--feature_target_names_path")
#     parser.add_argument("--output_json_path", default=None)
#     parser.add_argument("--log_dir")
#     parser.add_argument("--model_dir")
#     parser.add_argument("--epochs", type=int, default=2000)
#     args = parser.parse_args()

#     train(
#         data_csv_path=args.data_csv_path,
#         feature_target_names_path=args.feature_target_names_path,
#         output_json_path=args.output_json_path,
#         log_dir=args.log_dir,
#         model_dir=args.model_dir,
#         epochs=args.epochs,
#     )