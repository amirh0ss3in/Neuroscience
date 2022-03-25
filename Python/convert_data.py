from scipy.io import loadmat
import os
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
path_listdir = os.listdir(cwd+'..\\data\\IFP')

for i in range(20):
    if f'subj{i}' in path_listdir:
        path_subj = cwd+f'..\\data\\IFP\\subj{i}\\'
        # print(path_subj+os.listdir(path_subj)[0],path_subj+os.listdir(path_subj)[1])
        mat_subj = loadmat(file_name=path_subj+os.listdir(path_subj)[0])
        time_subj = mat_subj['t'][0]
        ifp_subj = mat_subj['d_all'][0]
        ifp_subj = np.asarray([i.T for i in ifp_subj]).T
        labels = mat_subj['p_all'][:,4]
        print(f'subj{i}:')
        print(ifp_subj.shape)
        print(labels.shape,'\n')
        save_path = cwd+'Data_subj\\'
        np.save(save_path+f'time_subj{i}.npy',time_subj)
        np.save(save_path+f'ifp_subj{i}.npy',ifp_subj)
        np.save(save_path+f'labels_subj{i}.npy',labels)
