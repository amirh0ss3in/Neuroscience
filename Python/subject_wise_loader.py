import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))+'/'
cwd = cwd.replace('\\','/')
load_path = cwd+'Data_subj/'
ids = [1,3,4,6,7,8,9,10,12,17,18]

def data_subject(subj_index):
    i = ids[subj_index]
    ifp_subj = np.load(load_path+f'ifp_subj{i}.npy')
    labels = np.load(load_path+f'labels_subj{i}.npy')
    xtrain, xtest, ytrain, ytest = train_test_split(ifp_subj, labels, test_size=0.2, random_state=42,  shuffle=True)
    return xtrain, xtest, ytrain, ytest

def load_time(subj_index):
    i = ids[subj_index]
    return np.load(load_path+f'time_subj{i}.npy')

def load_talairach(subj_index):
    i = ids[subj_index]
    return np.load(load_path+f'talairach_subj{i}.npy')

def stratified_split(subj_index, return_fold, k=5, seed=42):
    i = ids[subj_index]
    ifp_subj = np.load(load_path+f'ifp_subj{i}.npy')
    labels = np.load(load_path+f'labels_subj{i}.npy')

    if return_fold < 1 or return_fold > k:
        raise ValueError(f'Fold number must be between 1 and k (At this run, k is {k}.)')
    else:
        return_fold -= 1
        stratifiedkf=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
        train_indexs , val_indexs = [] , []
        for k, (train_index, val_index) in enumerate(stratifiedkf.split(ifp_subj, labels)):
            train_indexs.append(np.array(train_index))
            val_indexs.append(np.array(val_index))
        
        
        train_indexs , val_indexs = np.array(train_indexs) , np.array(val_indexs)
        

        xtrain = ifp_subj[train_indexs[return_fold]]
        ytrain = labels[train_indexs[return_fold]]
        xval = ifp_subj[val_indexs[return_fold]]
        yval = labels[val_indexs[return_fold]]
        del ifp_subj, labels
        return xtrain, xval, ytrain, yval

