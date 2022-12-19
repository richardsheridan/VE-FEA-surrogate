import torch
import torch.nn as nn
from torch.utils.data import DataLoader,SubsetRandomSampler
from utils import VEDataset
import numpy as np
from torch.optim import Adam, SGD, AdamW, Adadelta, Adagrad
from torch.optim.lr_scheduler import ExponentialLR
from model import SimpleANN, SplitANN
from tqdm import tqdm
import json
import logging
import os

# define the train process
def train(model, x, y, optimizer, criterion, output_split_dim, w_loss1, w_loss2):
    # clear the gradients of all optimized variables
    model.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(x)
    # calculate the loss
    loss1 = criterion(output[:output_split_dim],y[:output_split_dim])
    loss2 = criterion(output[output_split_dim:],y[output_split_dim:])
    loss = w_loss1*loss1 + w_loss2*loss2
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()

    return loss, output

def run_training(
    data_train_json_dir:str,
    data_test_json_dir:str,
    seed: int,
    input_dim: int,
    descriptor_dim: int,
    input_split_dim: int,
    hidden_1: int,
    hidden_2: int,
    dropout: float,
    output_dim: int,
    output_split_dim: int,
    save_to: str,
    EPOCHS: int,
    BATCH_SIZE: int,
    NUM_WORKERS: int,
    VALID_FRACTION: float,
    LR: float,
    weight_decay: float,
    optimizer:str,
    scheduler:str,
    loss_fn:str,
    model_for:str,
    no_cuda:bool,
    w_loss1:float,
    w_loss2:float,
    ):
    # create the 'save_to' folder if not exist
    # don't check for existence of the folder because it should not be
    os.mkdir(save_to)

    # configure logging
    logging.basicConfig(level=logging.INFO, 
        filename=f'{save_to}/training_log.log',
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')

    # CUDA for PyTorch
    # check CUDA only when use_gpu is True, otherwise use cpu
    use_cuda = torch.cuda.is_available() if not no_cuda else False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # load model
    if model_for == 'tand':
        model = SimpleANN(
            input_dim=input_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_dim=output_dim,
            dropout=dropout
            ).to(device)
    elif model_for == 'ep_epp':
        model = SplitANN(
            descriptor_dim=descriptor_dim,
            input_split_dim=input_split_dim,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            output_split_dim=output_split_dim,
            dropout=dropout
            ).to(device)
        input_dim = descriptor_dim + input_split_dim*2
    print(model)

    # Load weight
    if model_for == 'ep_epp':
        # upper half for E'
        model.ep_half.load_state_dict(torch.load('ep_hs1-128_hs2-128_do-0.2_ep-400_bs-32_lr-0.0001_opt-adam_sch-no_loss-mse/model.pt'))
        print("E' half weight loaded from ep_hs1-128_hs2-128_do-0.2_ep-400_bs-32_lr-0.0001_opt-adam_sch-no_loss-mse/model.pt")
        # lower half for tand
        model.epp_half.load_state_dict(torch.load('tand_hs1-128_hs2-128_ep-100_bs-32_lr-0.001_opt-adam_sch-exp_loss-mse/model.pt'))
        print("Tan Delta half weight loaded from tand_hs1-128_hs2-128_ep-100_bs-32_lr-0.001_opt-adam_sch-exp_loss-mse/model.pt.")
    # Datasets and Generators
    train_data = VEDataset(data_train_json_dir, index_in = input_dim)
    test_data = VEDataset(data_test_json_dir, index_in = input_dim)

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

    # define the loss function, use mean square error loss by default
    if loss_fn == 'l1':
        criterion = nn.L1Loss()
    elif loss_fn == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # define the optimizer, use Adam by default
    if optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=LR, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(model.parameters(), lr=LR, weight_decay=weight_decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=LR, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    # define the scheduler, use no scheduler by default
    no_scheduler = False
    if scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    else:
        no_scheduler = True

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
            loss, predictions = train(model,x_train,y_train,optimizer,criterion,
                output_split_dim, w_loss1, w_loss2)
            train_loss += loss.item() * x_train.size(0)
        if not no_scheduler:
            scheduler.step()

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
            valid_loss += loss.item() * x_val.size(0)
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, 
            train_loss,
            valid_loss
            ))
        logging.info('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
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
            logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), f'{save_to}/model.pt')
            valid_loss_min = valid_loss


    # evaluate on test dataset
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    # load the best model
    model.load_state_dict(torch.load(f'{save_to}/model.pt'))
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
        test_loss += loss.item() * x_test.size(0)
        
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))
    # save to history
    history['test_loss'] = test_loss

    # save history to json
    with open(f'{save_to}/history.json','w') as f:
        json.dump(history, f)
        print(f'history saved to {save_to}/history.json')
        logging.info(f'history saved to {save_to}/history.json')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train_json_dir", type=str,
        default='../full_train.json',
        help='path to the json file containing the dataframe of training data')
    parser.add_argument("--data_test_json_dir", type=str,
        default='../full_test.json',
        help='path to the json file containing the dataframe of test data')
    parser.add_argument("--input_dim", type=int, default=66,
        help='dimension of the input')
    parser.add_argument("--descriptor_dim", type=int, default=6,
        help='dimension of the descriptors in the input')
    parser.add_argument("--input_split_dim", type=int, default=30,
        help="dimension of the matrix E' and E'' in the input")
    parser.add_argument("--hidden_1", type=int, default=64,
        help='dimension of hidden layer 1')
    parser.add_argument("--hidden_2", type=int, default=64,
        help='dimension of hidden layer 2')
    parser.add_argument("--dropout", type=float, default=0.2,
        help='dropout ratio')
    parser.add_argument("--output_dim", type=int, default=60,
        help='dimension of the output')
    parser.add_argument("--output_split_dim", type=int, default=30,
        help="dimension of the composite E' and E'' in the output")
    parser.add_argument("-s", "--seed", type=int, default=27,
        help='random seed for train/valid split')
    parser.add_argument("-b", "--batch_size", type=int, default=32,
        help='the batch size for training, default to 32')
    parser.add_argument("-e", "--epochs", type=int, default=100,
        help='number of training epochs, default to 100')
    parser.add_argument("--num_workers", type=int, default=0,
        help='number of subprocesses to use for data loading, default to 0')
    parser.add_argument("--valid_fraction", type=float, default=0.2,
        help='fraction of training set to use as validation, default to 0.2')
    parser.add_argument('-l', "--lr", type=float, default=0.001,
        help='peak learning rate, default to 1e-3')
    parser.add_argument("--wd", type=float, default=0.01,
        help='weight decay of optimizer, default to 1e-2')
    parser.add_argument("--save_to", type=str, default=None,
        help='save model and history file under this name')
    parser.add_argument("--optimizer", type=str, default='adam',
        choices=['adam','sgd','adamw','adadelta','adagrad'],
        help='optimizer to be used for training')
    parser.add_argument("--scheduler", type=str, default='no',
        choices=['no','exp'],
        help='scheduler to be used for training')
    parser.add_argument("--loss_fn", type=str, default='mse',
        choices=['mse','l1','crossentropy'],
        help='save model and history file under this name')
    parser.add_argument("--task_name", type=str, default='unknown',
        help='name of the task, eg. ep_epp, tand')
    parser.add_argument("--model", type=str, default='tand',
        choices=['tand','ep_epp'],
        help='modeling for tand or ep_epp')
    parser.add_argument('--no_cuda', default=False, action='store_true',
        help='use gpu or cpu, default to gpu')
    parser.add_argument('--w_loss1', type=float, default=1,
        help="weight assigned to loss term 1 (E')")
    parser.add_argument('--w_loss2', type=float, default=1,
        help="weight assigned to loss term 2 (tand)")

    args = parser.parse_args()

    # generate save_to if not provided
    if not args.save_to:
        args.save_to = f'{args.task_name}_hs1-{args.hidden_1}_hs2-{args.hidden_2}_do-{args.dropout}_ep-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}_opt-{args.optimizer}_sch-{args.scheduler}_loss-{args.loss_fn}'
    
    run_training(
        data_train_json_dir=args.data_train_json_dir,
        data_test_json_dir=args.data_test_json_dir,
        seed=args.seed,
        input_dim=args.input_dim,
        descriptor_dim=args.descriptor_dim,
        input_split_dim=args.input_split_dim,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        dropout=args.dropout,
        output_dim=args.output_dim,
        output_split_dim=args.output_split_dim,
        save_to=args.save_to,
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        NUM_WORKERS=args.num_workers,
        VALID_FRACTION=args.valid_fraction,
        LR=args.lr,
        weight_decay=args.wd,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        loss_fn=args.loss_fn,
        model_for=args.model,
        no_cuda=args.no_cuda,
        w_loss1=args.w_loss1,
        w_loss2=args.w_loss2,
    )