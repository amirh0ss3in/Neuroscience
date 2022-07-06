from scipy.signal import filtfilt
from scipy import signal
import numpy as np
from load_data import xtests
import matplotlib.pyplot as plt

def d_ifp(X, n=20):
    b = [1.0 / n] * n
    a = 1
    init_arr = np.zeros(shape=X.shape)
    for trial in range(X.shape[0]):
        for channel in range(X.shape[2]):
            init_arr[trial,:,channel] = filtfilt(b, a, X[trial,:,channel])
    return init_arr
