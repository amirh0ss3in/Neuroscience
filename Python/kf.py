from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def ttv_kfold(X,Y,return_fold, k=5,seed=42):
    stratifiedkf=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
    x_, xtest, y_, ytest = train_test_split(X, Y, test_size=0.2, random_state=seed,  shuffle=True)
    if return_fold < 1 or return_fold > k:
        raise ValueError(f'Fold number must be between 1 and k (At this run, k is {k}.)')
    else:        
        return_fold -= 1
        train_indexs , val_indexs = [] , []
        for k, (train_index, val_index) in enumerate(stratifiedkf.split(x_, y_)):
            train_indexs.append(train_index)
            val_indexs.append(val_index)

        train_indexs , val_indexs = np.array(train_indexs) , np.array(val_indexs)
        xtrain = X[train_indexs[return_fold]]
        ytrain = Y[train_indexs[return_fold]]
        xval = X[val_indexs[return_fold]]
        yval = Y[val_indexs[return_fold]]

        return xtrain, ytrain, xval, yval, xtest, ytest

""" 
usecase:
ttv_kfold(X,Y,return_fold= 1)
ttv_kfold(X,Y,return_fold= 2)
ttv_kfold(X,Y,return_fold= 3)
ttv_kfold(X,Y,return_fold= 4)
ttv_kfold(X,Y,return_fold= 5)
 
"""