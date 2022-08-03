import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
load_path = cwd+'Data_subj\\'
ids = [1,3,4,6,7,8,9,10,12,17,18]

def data_subject(subj_index):
    i = ids[subj_index]
    X = np.load(load_path+f'ifp_subj{i}.npy')
    Y = np.load(load_path+f'labels_subj{i}.npy')
    return X, Y



def load_time(subj_index):
    i = ids[subj_index]
    return np.load(load_path+f'time_subj{i}.npy')

def load_talairach(subj_index):
    i = ids[subj_index]
    return np.load(load_path+f'talairach_subj{i}.npy')

# data_fold(subj_index=0, fold = 1)