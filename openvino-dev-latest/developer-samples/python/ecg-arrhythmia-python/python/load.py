import csv
import numpy as np
import os
import scipy.io as sio
import tqdm

STEP = 256

from time import time
import sys
from qarpo.demoutils import progressUpdate

mean = 7.4661856 
std = 236.10312 
classes = {'A' : 0, 'N' : 1, 'O' : 2, '~' : 3}

def process_x(x):
    x = np.expand_dims(x,axis=0)
    x = (x - mean) / std
    x = x[:, :, None]
    return x

def load_dataset(data_csv, progress_bar=True):
    with open(data_csv, 'r') as fid:
        data = csv.reader(fid)
        rows = [row for row in data]
        names = [row[0] for row in rows]
        ecgs = []
        labels = [classes[row[1]] for row in rows]
        sample_count = len(names)
        time_start = time()
        if progress_bar == True:
            names = tqdm.tqdm(names)
        for i, d in enumerate(names):
            ecgs.append(load_ecg('/data/ecg/training/' + d + '.mat'))
            if progress_bar != True:
                progressUpdate('./logs/' + os.environ['PBS_JOBID']  + '_load.txt', time()-time_start, i+1, sample_count)        
        sizes = []
        for item in ecgs:    
            sizes.append(len(item))
        return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]
