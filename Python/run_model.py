from base_hypermodel import BaseHyperModel
import keras_tuner as kt
from subject_wise_loader import stratified_split
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

cwd = os.path.dirname(os.path.abspath(__file__))+'/'

def main(SUBJ_INDEX, NUM_EPOCHS , BATCH_SIZE, NUM_INITIAL_POINTS, MAX_TRIALS, BETA, ALPHA, OBJECTIVE, SEED = 42):
    ids = [1,3,4,6,7,8,9,10,12,17,18]
    
    xtrain, xtest, ytrain, ytest = stratified_split(SUBJ_INDEX, return_fold=1, k=5, seed=SEED)

    # preprocess data
    ytrain[ytrain == 7] = 5
    ytest[ytest == 7] = 5
    ytrain = ytrain - 1
    ytest = ytest - 1

    input_shape = xtrain.shape[1:]
    n_class = len(set(ytest.flatten())|set(ytrain.flatten()))

    ytrain = tf.keras.utils.to_categorical(ytrain)
    ytest = tf.keras.utils.to_categorical(ytest)

    hyper_model = BaseHyperModel(input_shape = input_shape, n_class = n_class) 

    tuner_bo = kt.BayesianOptimization(
                seed=42,
                hypermodel = hyper_model,
                objective=OBJECTIVE,
                max_trials= MAX_TRIALS,
                num_initial_points=NUM_INITIAL_POINTS,
                alpha = ALPHA,
                beta= BETA,
                project_name=cwd+'Subject wise\\'+f'CNN_1D_1_subj{ids[SUBJ_INDEX]}'
            )

    # print(xtest.shape, ytest.shape, xtrain.shape, ytrain.shape)

    tuner_bo.search(xtrain, ytrain, validation_data=(xtest, ytest), batch_size= BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2)
    best_model = tuner_bo.get_best_models(num_models=1)[0]
    best_model.evaluate(xtest, ytest)
    best_model.save(cwd+'Subject wise/'+f'CNN_1D_1_subj{ids[SUBJ_INDEX]}/'+'best_model.h5')

    ytest_pred = best_model.predict(xtest)
    ytest = np.argmax(ytest, axis=1)
    ytest_pred = np.argmax(ytest_pred, axis=1)
    log_path = cwd+'Subject wise/'+f'CNN_1D_1_subj{ids[SUBJ_INDEX]}/'

    print(confusion_matrix(ytest,ytest_pred))
    print(ytest_pred.shape)
    print(classification_report(ytest,ytest_pred,digits=4))

    conf_mat = confusion_matrix(ytest,ytest_pred)
    report = classification_report(ytest,ytest_pred,digits=4)
    text_file = open(log_path+'classification_report.txt', "w")
    n = text_file.write(report)
    text_file.close()
    np.savetxt(log_path+'confusion_matrix.txt',conf_mat,fmt='%d')
    best_model.summary()
    tuner_bo.results_summary(num_trials=1)
    print('SUBJ_INDEX: ', SUBJ_INDEX)


NUM_EPOCHS = 30
BATCH_SIZE = 32
NUM_INITIAL_POINTS = 100
MAX_TRIALS = 200
BETA = 15
ALPHA = 0.01
OBJECTIVE = kt.Objective('val_f1_score', direction='max')
# OBJECTIVE = kt.Objective(' val_fbeta_score', direction='max')
# OBJECTIVE = kt.Objective('val_accuracy', direction='max')
SUBJ_INDEX = 0

if __name__ == '__main__':
    main(SUBJ_INDEX,
         NUM_EPOCHS, 
         BATCH_SIZE, 
         NUM_INITIAL_POINTS, 
         MAX_TRIALS, 
         BETA, 
         ALPHA,
         OBJECTIVE)