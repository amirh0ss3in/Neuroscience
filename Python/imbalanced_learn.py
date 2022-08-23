import os
import numpy as np
from collections import Counter

cwd = os.path.dirname(os.path.abspath(__file__))+'/'
cwd = cwd.replace('\\','/')

def iBL(xtrain, ytrain, method, log = True):
    
    counter_before = Counter(ytrain)

    balance = method()

    X, ytrain_balanced = balance.fit_resample(xtrain[:,:,0], ytrain)
    xtrain_balanced = np.zeros([X.shape[0],X.shape[1],xtrain.shape[-1]])

    for i in range(xtrain.shape[-1]):
        xtrain_balanced[:,:,i], _ = balance.fit_resample(xtrain[:,:,i], ytrain)

    counter_after = Counter(ytrain_balanced)
    
    if log:
        print("=====================================================================================================")
        print('\nBefore oversampling:', counter_before, '\nshape xtrain:', xtrain.shape)
        print('\n\nAfter oversampling:', counter_after, '\nshape xtrain balanced:' ,xtrain_balanced.shape,'\n')
        print(f'\nTotal number of {xtrain_balanced.shape[0] - xtrain.shape[0]} samples were added.\n')
        print("=====================================================================================================")
    
    return xtrain_balanced, ytrain_balanced


# # USAGE:
# from imblearn.over_sampling import SMOTE
# iBL(xtrain, ytrain, SMOTE)


# from subject_wise_loader import stratified_split
# from imblearn.over_sampling import SMOTE

# ids = [1,3,4,6,7,8,9,10,12,17,18]
# for s in range(11):
#     SEED = 42
#     SUBJ_INDEX = s
#     FOLD = 1

#     xtrain, xtest, ytrain, ytest = stratified_split(SUBJ_INDEX, return_fold=FOLD, k=5, seed=SEED)

#     # preprocess data
#     ytrain[ytrain == 7] = 5
#     ytest[ytest == 7] = 5
#     ytrain = ytrain - 1
#     ytest = ytest - 1
#     print(f's{s}, id{ids[s]}')
#     iBL(xtrain, ytrain, SMOTE)