import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
load_path = cwd+'Data_subj\\'
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