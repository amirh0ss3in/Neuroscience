from tabnanny import verbose
import tensorflow as tf 
import numpy as np
import os
import matplotlib.pyplot as plt 
import load_best
from subject_wise_loader import data_subject, load_time, load_talairach
from visualize_tal import plot_coordinates
from tqdm import tqdm

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

def only_channel(X, channel):
    modified = np.zeros(X.shape)
    modified[:,:,channel] = X[:,:,channel]
    return modified

def find_max_channel_activity(subj_index, treshhold = None, visualize_filters = False):
    ids = [1,3,4,6,7,8,9,10,12,17,18]
    version = 1
    subj = f"CNN_1D_{version}_subj{ids[subj_index]}"
    best_model = load_best.load(subj = subj, version = version)
    time = load_time(subj_index)
    xtrain, xtest, ytrain, ytest = data_subject(subj_index)
    ypred = np.argmax(best_model.predict(xtest),axis=1)
    correct_inds = [ypred[i] == ytest[i] for i in range(len(ytest))]
    input_shape = xtrain.shape[1:]
    ytrain[ytrain == 7] = 5
    ytest[ytest == 7] = 5
    ytrain = ytrain - 1
    ytest = ytest - 1
    model = tf.keras.Model(inputs=best_model.inputs[0], outputs=best_model.layers[len(best_model.layers)-2].output)
    model.summary()
    outs = [] 
    if treshhold is None:
        treshhold = 0.1
    else:
        pass
    for channel in tqdm(range(xtest.shape[-1])):
        modified = only_channel(xtest, channel)
        predicts = model.predict(modified, verbose=0)
        for ind in range(len(predicts)):
            if correct_inds[ind] and np.max(predicts[ind])>treshhold:
                outs.append([ind, channel])
                if visualize_filters:
                    plt.figure(1)
                    plt.subplot(211)
                    plt.plot(time, modified[ind])
                    plt.title(f'label: {int(ytest[ind])},\nchannel index:{channel}')
                    plt.subplot(212)
                    time_p = time[::len(modified[ind])//len(predicts[ind])]
                    plt.plot(time_p[:len(predicts[ind])] , predicts[ind])
                    plt.show()

    if not outs:
        raise ValueError("The model is not confident enough or the treshhold is way too high.")
    
    outs = np.asarray(outs)
    c = np.zeros(xtest.shape[-1])
    for i in range(xtest.shape[-1]):
        for j in outs[:,1]:
            if i == j:
                c[i] += 1
    
    directory = cwd+f'Max channel activity\\CNN_1D_{version}_subj{ids[subj_index]}\\'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.bar(np.linspace(1, xtest.shape[-1], xtest.shape[-1]),c)
    plt.savefig(directory + f'All_channels_activity_treshhold( {treshhold} ).png')
    plt.close()
    # plt.show() 
    
    top_k = 5
    idx = (-c).argsort()[:top_k]
    talairach = load_talairach(subj_index)

    print('Top indices: ',idx,'\n')
    print('\nTop talairach coordinates:\n',talairach[idx])
    for ind, i in enumerate(talairach[idx]):
        plot_coordinates(i,input_coordinate = 'tal', colored = True, save_path = directory + f'{ind+1}_top_treshhold( {treshhold} ).png')

def main(subj_index, treshhold = None, visualize_filters = False):
    try:
        find_max_channel_activity(subj_index, treshhold=treshhold, visualize_filters=visualize_filters)
    except ValueError:
        print("The model is not confident enough or the treshhold is way too high.")

treshholds = np.linspace(0.1,0.9,9)
for i in range(len(treshholds)):
    for subj_index in range(11):
        main(subj_index, treshhold = treshholds[i])