#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

tqdm.pandas(desc='Progress:')
# In[7]:


def get_tand(tsv):
    try:
        data = np.loadtxt('../'+tsv,delimiter='\t',skiprows=1)
        freq = data[:,0]
        ep = data[:,1]
        epp = data[:,2]
        tand = epp/ep
        tdpk_h = max(tand)
        tdpk_f = freq[np.argmax(tand)]
        return tdpk_h,tdpk_f,*tand
    except:
        return [None]*(2+30)


# In[9]:


def get_volume_fractions(intph_img):
    try:
        microstructure = np.load('../'+intph_img)
        n_filler = np.count_nonzero(microstructure == 0)
        n_intph = np.count_nonzero(microstructure == 1)
        if len(np.unique(microstructure))==2:
            n_intph = 0 # if no interphase assigned
        total = microstructure.shape[0]*microstructure.shape[1]
        return n_filler/total,n_intph/total
    except:
        return None,None


# Define a function to preprocess matrix master curve by getting their tan delta at 30 freq corresponding to the freq intervals in the composite VE curve.

# In[15]:


def interpolate(x, ref_x, ref_y, log_x = True, log_y = False, return_log = False):
    '''
    x: new x coord to be interpolated
    
    ref_x: iterable x for reference
    
    ref_y: iterable y for reference
    
    log_x: whether to interpolate on log(x), default to True
    
    log_y: whether to interpolate on log(y), default to False
    
    return_log: whether to return log(y), default to False
    '''
    # find the two nearest points in reference curve
    p1,p2 = np.argsort(np.abs(ref_x - x))[:2]
    # interpolate
    x1,x2,y1,y2 = ref_x[p1],ref_x[p2],ref_y[p1],ref_y[p2]
    if log_x:
        x1 = np.log10(x1)
        x2 = np.log10(x2)
        x = np.log10(x)
    if log_y:
        y1 = np.log10(y1)
        y2 = np.log10(y2)
    y = (y1-y2)/(x1-x2)*(x-x1)+y1
    if log_y and not return_log:
        y = 10**y
    return y

def preprocess_master_curve(master_curve, ref_freq):
    mc = np.loadtxt('../'+master_curve)
    return [interpolate(x,mc[:,0],mc[:,2]/mc[:,1],log_x=True,log_y=False) for x in ref_freq]


# Include E' and E''

# In[17]:


def get_ep_epp(tsv, return_log=True):
    try:
        data = np.loadtxt('../'+tsv,delimiter='\t',skiprows=1)
        freq = data[:,0]
        ep = data[:,1]*1e6
        epp = data[:,2]*1e6
        if return_log:
            ep = np.log10(ep)
            epp = np.log10(epp)
        return *ep,*epp
    except:
        return [None]*(30+30)



# In[19]:


def preprocess_master_curve_for_ep_epp(master_curve, ref_freq):
    mc = np.loadtxt('../'+master_curve)
    return [interpolate(x,mc[:,0],mc[:,1]*1e6,log_x=True,log_y=True,return_log=True) for x in ref_freq] + [interpolate(x,mc[:,0],mc[:,2]*1e6,log_x=True,log_y=True,return_log=True) for x in ref_freq]


# In[1]:


def raw_json_to_train_test(json_dir, matrix, mode, split=0.2, seed=27):
    df = pd.read_json(json_dir,orient="index")
    # assume freq intervals are unchanged throughout the json
    ref_curve = np.loadtxt(f'../{df.index[0]}',delimiter='\t',skiprows=1)
    ref_freq = ref_curve[:,0]
    # add matrix info
    df['matrix'] = matrix
    # move ve response file name out of the index col
    df['VE_response'] = df.index
    df = df.dropna()
    # get tan delta peak height & frequency
    # get all 30 tan delta values of PNC
    df['tan_d_peak_height'],df['tan_d_peak_freq'],    df['tan_d_0'],df['tan_d_1'],df['tan_d_2'],df['tan_d_3'],df['tan_d_4'],df['tan_d_5'],    df['tan_d_6'],df['tan_d_7'],df['tan_d_8'],df['tan_d_9'],df['tan_d_10'],df['tan_d_11'],    df['tan_d_12'],df['tan_d_13'],df['tan_d_14'],df['tan_d_15'],df['tan_d_16'],df['tan_d_17'],    df['tan_d_18'],df['tan_d_19'],df['tan_d_20'],df['tan_d_21'],df['tan_d_22'],df['tan_d_23'],    df['tan_d_24'],df['tan_d_25'],df['tan_d_26'],df['tan_d_27'],df['tan_d_28'],df['tan_d_29'],    = zip(*df.VE_response.apply(get_tand))
    # get vf and vt
    df['VfActual'],df['Vt'] = zip(*df.intph_img.apply(get_volume_fractions))
    df = df.dropna()
    # Unpack the layers column. Originally it's a list.
    df['layer_0'] = df.layers.apply(lambda x:x[0])
    # get all 30 tan delta values of pure polymer
    df['mc_tand_0'],df['mc_tand_1'],df['mc_tand_2'],df['mc_tand_3'],df['mc_tand_4'],df['mc_tand_5'],    df['mc_tand_6'],df['mc_tand_7'],df['mc_tand_8'],df['mc_tand_9'],df['mc_tand_10'],df['mc_tand_11'],    df['mc_tand_12'],df['mc_tand_13'],df['mc_tand_14'],df['mc_tand_15'],df['mc_tand_16'],df['mc_tand_17'],    df['mc_tand_18'],df['mc_tand_19'],df['mc_tand_20'],df['mc_tand_21'],df['mc_tand_22'],df['mc_tand_23'],    df['mc_tand_24'],df['mc_tand_25'],df['mc_tand_26'],df['mc_tand_27'],df['mc_tand_28'],df['mc_tand_29'],    = zip(*df.master_curve.apply(lambda x:preprocess_master_curve(x,ref_freq)))
    # get all 30 E' values and 30 E'' values of PNC
    df['ep_0'],df['ep_1'],df['ep_2'],df['ep_3'],df['ep_4'],df['ep_5'],    df['ep_6'],df['ep_7'],df['ep_8'],df['ep_9'],df['ep_10'],df['ep_11'],    df['ep_12'],df['ep_13'],df['ep_14'],df['ep_15'],df['ep_16'],df['ep_17'],    df['ep_18'],df['ep_19'],df['ep_20'],df['ep_21'],df['ep_22'],df['ep_23'],    df['ep_24'],df['ep_25'],df['ep_26'],df['ep_27'],df['ep_28'],df['ep_29'],    df['epp_0'],df['epp_1'],df['epp_2'],df['epp_3'],df['epp_4'],df['epp_5'],    df['epp_6'],df['epp_7'],df['epp_8'],df['epp_9'],df['epp_10'],df['epp_11'],    df['epp_12'],df['epp_13'],df['epp_14'],df['epp_15'],df['epp_16'],df['epp_17'],    df['epp_18'],df['epp_19'],df['epp_20'],df['epp_21'],df['epp_22'],df['epp_23'],    df['epp_24'],df['epp_25'],df['epp_26'],df['epp_27'],df['epp_28'],df['epp_29'],    = zip(*df.VE_response.apply(get_ep_epp))
    # get all 30 E' values and 30 E'' values of pure polymer
    df['mc_ep_0'],df['mc_ep_1'],df['mc_ep_2'],df['mc_ep_3'],df['mc_ep_4'],df['mc_ep_5'],    df['mc_ep_6'],df['mc_ep_7'],df['mc_ep_8'],df['mc_ep_9'],df['mc_ep_10'],df['mc_ep_11'],    df['mc_ep_12'],df['mc_ep_13'],df['mc_ep_14'],df['mc_ep_15'],df['mc_ep_16'],df['mc_ep_17'],    df['mc_ep_18'],df['mc_ep_19'],df['mc_ep_20'],df['mc_ep_21'],df['mc_ep_22'],df['mc_ep_23'],    df['mc_ep_24'],df['mc_ep_25'],df['mc_ep_26'],df['mc_ep_27'],df['mc_ep_28'],df['mc_ep_29'],    df['mc_epp_0'],df['mc_epp_1'],df['mc_epp_2'],df['mc_epp_3'],df['mc_epp_4'],df['mc_epp_5'],    df['mc_epp_6'],df['mc_epp_7'],df['mc_epp_8'],df['mc_epp_9'],df['mc_epp_10'],df['mc_epp_11'],    df['mc_epp_12'],df['mc_epp_13'],df['mc_epp_14'],df['mc_epp_15'],df['mc_epp_16'],df['mc_epp_17'],    df['mc_epp_18'],df['mc_epp_19'],df['mc_epp_20'],df['mc_epp_21'],df['mc_epp_22'],df['mc_epp_23'],    df['mc_epp_24'],df['mc_epp_25'],df['mc_epp_26'],df['mc_epp_27'],df['mc_epp_28'],df['mc_epp_29'],    = zip(*df.master_curve.apply(lambda x:preprocess_master_curve_for_ep_epp(x,ref_freq)))
    df = df.copy().dropna()
    # replace intph_img column with absolute path
    df['intph_img'] = df['intph_img'].apply(lambda x:os.path.abspath('.'+x))
    # get percolation flag
    unique_ms = set()
    intph_img_2_tup = {}
    intph_tup_2_img = {}
    for intph_img in df['intph_img'].unique():
        intph_thickness = re.search(r'intph_([\d]+)_',intph_img)
        thickness = '0'
        if intph_thickness:
            thickness = intph_thickness.group(1)
        unique_ms.add((thickness, os.path.split(intph_img)[1]))
        intph_img_2_tup[intph_img] = (thickness, os.path.split(intph_img)[1])
        intph_tup_2_img[(thickness, os.path.split(intph_img)[1])] = intph_img
    percolation_data = {i:is_percolated(intph_tup_2_img[i]) for i in tqdm(unique_ms)}
    df['percolation'] = df.intph_img.progress_apply(lambda x:percolation_data[intph_img_2_tup[x]])
    # train/test split, only split when split is positive
    if split > 0:
        df_train, df_test = train_test_split(df, test_size=split, random_state=seed)
    else:
        df_train = df.copy()
    # configure columns for different output mode
    # add interphased microstructure and percolation flag to cols as well
    ep_tand_cols = ['ParRu', 'ParRv', 'VfActual', 'Vt', 'intph_shift', 'intph_l_brd',
                'mc_ep_0', 'mc_ep_1', 'mc_ep_2', 'mc_ep_3', 'mc_ep_4', 'mc_ep_5', 
                'mc_ep_6', 'mc_ep_7', 'mc_ep_8', 'mc_ep_9', 'mc_ep_10', 'mc_ep_11',
                'mc_ep_12', 'mc_ep_13', 'mc_ep_14', 'mc_ep_15', 'mc_ep_16', 'mc_ep_17',
                'mc_ep_18', 'mc_ep_19', 'mc_ep_20', 'mc_ep_21', 'mc_ep_22', 'mc_ep_23',
                'mc_ep_24', 'mc_ep_25', 'mc_ep_26', 'mc_ep_27', 'mc_ep_28', 'mc_ep_29',
                'mc_tand_0', 'mc_tand_1', 'mc_tand_2', 'mc_tand_3', 'mc_tand_4', 'mc_tand_5', 
                'mc_tand_6', 'mc_tand_7', 'mc_tand_8', 'mc_tand_9', 'mc_tand_10', 'mc_tand_11',
                'mc_tand_12', 'mc_tand_13', 'mc_tand_14', 'mc_tand_15', 'mc_tand_16', 'mc_tand_17',
                'mc_tand_18', 'mc_tand_19', 'mc_tand_20', 'mc_tand_21', 'mc_tand_22', 'mc_tand_23',
                'mc_tand_24', 'mc_tand_25', 'mc_tand_26', 'mc_tand_27', 'mc_tand_28', 'mc_tand_29',
                'ep_0', 'ep_1', 'ep_2', 'ep_3', 'ep_4', 'ep_5', 'ep_6',
                'ep_7', 'ep_8', 'ep_9', 'ep_10', 'ep_11', 'ep_12',
                'ep_13', 'ep_14', 'ep_15', 'ep_16', 'ep_17', 'ep_18',
                'ep_19', 'ep_20', 'ep_21', 'ep_22', 'ep_23', 'ep_24',
                'ep_25', 'ep_26', 'ep_27', 'ep_28', 'ep_29',
                'tan_d_0', 'tan_d_1', 'tan_d_2', 'tan_d_3', 'tan_d_4', 'tan_d_5', 'tan_d_6',
                'tan_d_7', 'tan_d_8', 'tan_d_9', 'tan_d_10', 'tan_d_11', 'tan_d_12',
                'tan_d_13', 'tan_d_14', 'tan_d_15', 'tan_d_16', 'tan_d_17', 'tan_d_18',
                'tan_d_19', 'tan_d_20', 'tan_d_21', 'tan_d_22', 'tan_d_23', 'tan_d_24',
                'tan_d_25', 'tan_d_26', 'tan_d_27', 'tan_d_28', 'tan_d_29', 'intph_img', 'percolation'
               ]
    ep_epp_cols = ['ParRu', 'ParRv', 'VfActual', 'Vt', 'intph_shift', 'intph_l_brd',
                'mc_ep_0', 'mc_ep_1', 'mc_ep_2', 'mc_ep_3', 'mc_ep_4', 'mc_ep_5', 
                'mc_ep_6', 'mc_ep_7', 'mc_ep_8', 'mc_ep_9', 'mc_ep_10', 'mc_ep_11',
                'mc_ep_12', 'mc_ep_13', 'mc_ep_14', 'mc_ep_15', 'mc_ep_16', 'mc_ep_17',
                'mc_ep_18', 'mc_ep_19', 'mc_ep_20', 'mc_ep_21', 'mc_ep_22', 'mc_ep_23',
                'mc_ep_24', 'mc_ep_25', 'mc_ep_26', 'mc_ep_27', 'mc_ep_28', 'mc_ep_29',
                'mc_epp_0', 'mc_epp_1', 'mc_epp_2', 'mc_epp_3', 'mc_epp_4', 'mc_epp_5', 
                'mc_epp_6', 'mc_epp_7', 'mc_epp_8', 'mc_epp_9', 'mc_epp_10', 'mc_epp_11',
                'mc_epp_12', 'mc_epp_13', 'mc_epp_14', 'mc_epp_15', 'mc_epp_16', 'mc_epp_17',
                'mc_epp_18', 'mc_epp_19', 'mc_epp_20', 'mc_epp_21', 'mc_epp_22', 'mc_epp_23',
                'mc_epp_24', 'mc_epp_25', 'mc_epp_26', 'mc_epp_27', 'mc_epp_28', 'mc_epp_29',
                'ep_0', 'ep_1', 'ep_2', 'ep_3', 'ep_4', 'ep_5', 'ep_6',
                'ep_7', 'ep_8', 'ep_9', 'ep_10', 'ep_11', 'ep_12',
                'ep_13', 'ep_14', 'ep_15', 'ep_16', 'ep_17', 'ep_18',
                'ep_19', 'ep_20', 'ep_21', 'ep_22', 'ep_23', 'ep_24',
                'ep_25', 'ep_26', 'ep_27', 'ep_28', 'ep_29',
                'epp_0', 'epp_1', 'epp_2', 'epp_3', 'epp_4', 'epp_5', 'epp_6',
                'epp_7', 'epp_8', 'epp_9', 'epp_10', 'epp_11', 'epp_12',
                'epp_13', 'epp_14', 'epp_15', 'epp_16', 'epp_17', 'epp_18',
                'epp_19', 'epp_20', 'epp_21', 'epp_22', 'epp_23', 'epp_24',
                'epp_25', 'epp_26', 'epp_27', 'epp_28', 'epp_29', 'intph_img', 'percolation'
               ]
    tand_cols = ['ParRu', 'ParRv', 'VfActual', 'Vt', 'intph_shift', 'intph_l_brd',
                'mc_tand_0', 'mc_tand_1', 'mc_tand_2', 'mc_tand_3', 'mc_tand_4', 'mc_tand_5', 
                'mc_tand_6', 'mc_tand_7', 'mc_tand_8', 'mc_tand_9', 'mc_tand_10', 'mc_tand_11',
                'mc_tand_12', 'mc_tand_13', 'mc_tand_14', 'mc_tand_15', 'mc_tand_16', 'mc_tand_17',
                'mc_tand_18', 'mc_tand_19', 'mc_tand_20', 'mc_tand_21', 'mc_tand_22', 'mc_tand_23',
                'mc_tand_24', 'mc_tand_25', 'mc_tand_26', 'mc_tand_27', 'mc_tand_28', 'mc_tand_29',
                'tan_d_0', 'tan_d_1', 'tan_d_2', 'tan_d_3', 'tan_d_4', 'tan_d_5', 'tan_d_6',
                'tan_d_7', 'tan_d_8', 'tan_d_9', 'tan_d_10', 'tan_d_11', 'tan_d_12',
                'tan_d_13', 'tan_d_14', 'tan_d_15', 'tan_d_16', 'tan_d_17', 'tan_d_18',
                'tan_d_19', 'tan_d_20', 'tan_d_21', 'tan_d_22', 'tan_d_23', 'tan_d_24',
                'tan_d_25', 'tan_d_26', 'tan_d_27', 'tan_d_28', 'tan_d_29', 'intph_img', 'percolation'
               ]
    ep_cols = ['ParRu', 'ParRv', 'VfActual', 'Vt', 'intph_shift', 'intph_l_brd',
                'mc_ep_0', 'mc_ep_1', 'mc_ep_2', 'mc_ep_3', 'mc_ep_4', 'mc_ep_5', 
                'mc_ep_6', 'mc_ep_7', 'mc_ep_8', 'mc_ep_9', 'mc_ep_10', 'mc_ep_11',
                'mc_ep_12', 'mc_ep_13', 'mc_ep_14', 'mc_ep_15', 'mc_ep_16', 'mc_ep_17',
                'mc_ep_18', 'mc_ep_19', 'mc_ep_20', 'mc_ep_21', 'mc_ep_22', 'mc_ep_23',
                'mc_ep_24', 'mc_ep_25', 'mc_ep_26', 'mc_ep_27', 'mc_ep_28', 'mc_ep_29',
                'ep_0', 'ep_1', 'ep_2', 'ep_3', 'ep_4', 'ep_5', 'ep_6',
                'ep_7', 'ep_8', 'ep_9', 'ep_10', 'ep_11', 'ep_12',
                'ep_13', 'ep_14', 'ep_15', 'ep_16', 'ep_17', 'ep_18',
                'ep_19', 'ep_20', 'ep_21', 'ep_22', 'ep_23', 'ep_24',
                'ep_25', 'ep_26', 'ep_27', 'ep_28', 'ep_29', 'intph_img', 'percolation'
               ]
    # dump tand, ep, ep_tand if mode is "all"
    if mode == 'all':
        for dump_mode,cols in [('tand',tand_cols), ('ep',ep_cols), ('ep_tand',ep_tand_cols)]:
            df_train_dump = df_train[cols].reset_index(drop=True)
            if split > 0:
                df_test_dump = df_test[cols].reset_index(drop=True)
            else:
                df_test_dump = None
            # dump to json
            df_train_dump.to_json(f'{matrix}_{dump_mode}_train.json')
            print(f"Train dataframe dumped to {matrix}_{dump_mode}_train.json")
            if split > 0:
                df_test_dump.to_json(f'{matrix}_{dump_mode}_test.json')
                print(f"Test dataframe dumped to {matrix}_{dump_mode}_test.json")
        return df_train_dump, df_test_dump
    # otherwise, get the columns to be dumped
    if mode == 'ep_tand':
        cols = ep_tand_cols
    elif mode == 'tand':
        cols = tand_cols
    elif mode == 'ep':
        cols = ep_cols
    elif mode == 'ep_epp':
        cols = ep_epp_cols
    # select columns
    df_train_dump = df_train[cols].reset_index(drop=True)
    if split > 0:
        df_test_dump = df_test[cols].reset_index(drop=True)
    else:
        df_test_dump = None
    # dump to json
    df_train_dump.to_json(f'{matrix}_{mode}_train.json')
    print(f"Train dataframe dumped to {matrix}_{mode}_train.json")
    if split > 0:
        df_test_dump.to_json(f'{matrix}_{mode}_test.json')
        print(f"Test dataframe dumped to {matrix}_{mode}_test.json")
    return df_train_dump, df_test_dump

# a function to determine whether a numpy array M is percolated or not
def is_percolated(intph_img):
    # Load microstructure with interphase
    M = np.load(intph_img)
    # Skip no interphase cases, meaning max value < 2 (0: particle, 1 to x-1: interphase, x: matrix)
    matrix_code = M.max()
    if matrix_code < 2:
        return False
    # Only keep open cells
    M = ((M > 0) & (M < matrix_code)).astype('uint8')
    # Find all open cells in the top row of the array
    open_cells = np.flatnonzero(M[0])
    # Define a stack to store cells to be visited
    stack = [(0, j) for j in open_cells]
    # Define a set to store visited cells
    visited = set()
    # Define a loop to visit all reachable cells
    while stack:
        i, j = stack.pop()
        visited.add((i, j))
        # Check if the current cell is in the bottom row
        if i == M.shape[0] - 1:
            return True
        # Add all neighboring open cells to the stack
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if (ni >= 0 and ni < M.shape[0] and
                nj >= 0 and nj < M.shape[1] and
                M[ni, nj] == 1 and (ni, nj) not in visited):
                stack.append((ni, nj))
    # Left to right
    stack = [(i, 0) for i in np.flatnonzero(M[:,0])]
    # reset visited
    visited = set()
    # Define a loop to visit all reachable cells
    while stack:
        i, j = stack.pop()
        visited.add((i, j))
        # Check if the current cell is in the rightmost col
        if j == M.shape[1] - 1:
            return True
        # Add all neighboring open cells to the stack
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if (ni >= 0 and ni < M.shape[0] and
                nj >= 0 and nj < M.shape[1] and
                M[ni, nj] == 1 and (ni, nj) not in visited):
                stack.append((ni, nj))
    # If no percolating path was found, return False
    return False