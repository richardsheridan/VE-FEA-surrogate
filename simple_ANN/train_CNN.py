import torch
import torch.nn as nn
from torch.utils.data import DataLoader,SubsetRandomSampler
from utils import VECNNDataset
import numpy as np
from torch.optim import Adam, SGD, AdamW, Adadelta, Adagrad
from torch.optim.lr_scheduler import ExponentialLR
from model import SimpleANN_CNN
from tqdm import tqdm
import json
import logging
import os

# define the train process
def train(model, image, x, y, optimizer, criterion):
    # clear the gradients of all optimized variables
    model.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(image,x)
    # calculate the loss
    loss =criterion(output,y)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()

    return loss, output

def run_training(
    data_train_json_dir:str,
    data_test_json_dir:str,
    seed: int,
    image_col: str,
    input_idx: int,
    output_idx: int,
    descriptor_dim: int,
    input_split_dim: int,
    hidden_1: int,
    hidden_2: int,
    dropout: float,
    output_split_dim: int,
    conv_kernel_size: int,
    stride: int,
    padding: int,
    input_ch_1: int,
    output_ch_1: int,
    output_ch_2: int,
    output_ch_3: int,
    pool_kernel_size: int,
    image_dim: int,
    embedding_dim: int,
    save_to: str,
    EPOCHS: int,
    BATCH_SIZE: int,
    NUM_WORKERS: int,
    VALID_FRACTION: float,
    LR: float,
    weight_decay: float,
    optimizer:str,
    scheduler:str,
    gamma: float,
    loss_fn:str,
    # model_for:str,
    no_cuda:bool,
    ):
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
    model = SimpleANN_CNN(
        descriptor_dim=descriptor_dim,
        input_split_dim=input_split_dim,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        output_split_dim=output_split_dim,
        dropout=dropout,
        conv_kernel_size=conv_kernel_size,
        stride=stride,
        padding=padding,
        input_ch_1=input_ch_1,
        output_ch_1=output_ch_1,
        output_ch_2=output_ch_2,
        output_ch_3=output_ch_3,
        pool_kernel_size=pool_kernel_size,
        image_dim=image_dim,
        embedding_dim=embedding_dim
        ).to(device)

    # Datasets and Generators
    train_data = VECNNDataset(
        json_file=data_train_json_dir,
        image_col=image_col,
        input_idx=input_idx,
        output_idx=output_idx
        )
    test_data = VECNNDataset(
        json_file=data_test_json_dir,
        image_col=image_col,
        input_idx=input_idx,
        output_idx=output_idx
        )

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
        scheduler = ExponentialLR(optimizer, gamma=gamma)
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
            im_train, x_train, y_train = batch['image'], batch['input'], batch['output']
            im_train = im_train.to(device)
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train(model,im_train,x_train,y_train,optimizer,criterion)
            train_loss += loss.item() * x_train.size(0)
        if not no_scheduler:
            scheduler.step()

        ######################    
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for batch in tqdm(valid_loader):
            im_val, x_val, y_val = batch['image'], batch['input'], batch['output']
            im_val = im_val.to(device)
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(im_val,x_val)
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
        im_test, x_test, y_test = batch['image'], batch['input'], batch['output']
        im_test = im_test.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(im_test,x_test)
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
    parser.add_argument("--image_col", type=str, default='intph_img',
        help='the column name of the column that contains microstructure image file name in dataframe')
    parser.add_argument("--input_idx", type=int, nargs='+', default=[1,37],
        help='index range of input features in the dataframe')
    parser.add_argument("--output_idx", type=int, nargs='+', default=[37,67],
        help='index range of input features in the dataframe')
    parser.add_argument("--descriptor_dim", type=int, default=6,
        help='dimension of the descriptors in the input')
    parser.add_argument("--input_split_dim", type=int, default=30,
        help="dimension of the matrix E' and E'' in the input")
    parser.add_argument("--hidden_1", type=int, default=64,
        help='dimension of hidden layer 1')
    parser.add_argument("--hidden_2", type=int, default=64,
        help='dimension of hidden layer 2')
    parser.add_argument("--output_split_dim", type=int, default=30,
        help="dimension of the composite E' and E'' in the output")
    parser.add_argument("--dropout", type=float, default=0.2,
        help='dropout ratio')
    parser.add_argument("--conv_kernel_size", type=int, default=3,
        help="kernel size in the convolutional layer")
    parser.add_argument("--stride", type=int, default=1,
        help="stride size in the convolutional layer")
    parser.add_argument("--padding", type=int, default=1,
        help="padding size in the convolutional layer")
    parser.add_argument("--input_ch_1", type=int, default=1,
        help="number of input channels in the 1st convolutional layer")
    parser.add_argument("--output_ch_1", type=int, default=16,
        help="number of output channels in the 1st convolutional layer")
    parser.add_argument("--output_ch_2", type=int, default=32,
        help="number of output channels in the 2nd convolutional layer")
    parser.add_argument("--output_ch_3", type=int, default=64,
        help="number of output channels in the 3rd convolutional layer")
    parser.add_argument("--pool_kernel_size", type=int, default=2,
        help="kernel size in the max pooling layer")
    parser.add_argument("--image_dim", type=int, default=500,
        help="image size, image must has identical width and height")
    parser.add_argument("--embedding_dim", type=int, default=32,
        help="output image embedding size")
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
    parser.add_argument("--gamma", type=float, default=0.9,
        help='decay parameter gamma to be used with scheduler, default to 0.9')
    parser.add_argument("--loss_fn", type=str, default='mse',
        choices=['mse','l1','crossentropy'],
        help='save model and history file under this name')
    parser.add_argument("--task_name", type=str, default='unknown',
        help='name of the task, eg. ep_epp, tand')
    parser.add_argument('--no_cuda', default=False, action='store_true',
        help='use gpu or cpu, default to gpu')
    args = parser.parse_args()

    # generate save_to if not provided
    if not args.save_to:
        args.save_to = f'{args.task_name}_hs1-{args.hidden_1}_hs2-{args.hidden_2}_do-{args.dropout}_ep-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}_opt-{args.optimizer}_sch-{args.scheduler}_loss-{args.loss_fn}'

    # create the 'save_to' folder if not exist
    # don't check for existence of the folder because it should not be
    os.mkdir(args.save_to)
    
    # save parser config
    with open(f'{args.save_to}/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    run_training(
        data_train_json_dir=args.data_train_json_dir,
        data_test_json_dir=args.data_test_json_dir,
        seed=args.seed,
        image_col=args.image_col,
        input_idx=args.input_idx,
        output_idx=args.output_idx,
        descriptor_dim=args.descriptor_dim,
        input_split_dim=args.input_split_dim,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        dropout=args.dropout,
        output_split_dim=args.output_split_dim,
        conv_kernel_size=args.conv_kernel_size,
        stride=args.stride,
        padding=args.padding,
        input_ch_1=args.input_ch_1,
        output_ch_1=args.output_ch_1,
        output_ch_2=args.output_ch_2,
        output_ch_3=args.output_ch_3,
        pool_kernel_size=args.pool_kernel_size,
        image_dim=args.image_dim,
        embedding_dim=args.embedding_dim,
        save_to=args.save_to,
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        NUM_WORKERS=args.num_workers,
        VALID_FRACTION=args.valid_fraction,
        LR=args.lr,
        gamma=args.gamma,
        weight_decay=args.wd,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        loss_fn=args.loss_fn,
        no_cuda=args.no_cuda,
    )
