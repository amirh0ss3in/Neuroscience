from scipy.io import loadmat
import os
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
path_listdir = os.listdir(cwd+'..\\data\\locations')
for i in range(20):
    if f'electrode_locations_subj{i}.mat' in path_listdir:
        path_subj = cwd+f'..\\data\\locations\\electrode_locations_subj{i}.mat'
        # print(path_subj+os.listdir(path_subj)[0],path_subj+os.listdir(path_subj)[1])
        mat_subj = loadmat(file_name=path_subj)
        talairach_subj = mat_subj['talairach']
        save_path = cwd+'Data_subj\\'
        np.save(save_path+f'talairach_subj{i}.npy',talairach_subj)
