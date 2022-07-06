from subject_wise_loader import data_subject
import os
import keras_tuner as kt
import tensorflow as tf 
import numpy as np
from keras_tuner import BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from kf import ttv_kfold
import keras.backend as K

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

# https://keras.io/guides/keras_tuner/distributed_tuning/
# https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe

def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i in range(hp.Int("conv_layers", 2, 5, default=3)): 
        x = tf.keras.layers.Conv1D(
            filters=hp.Int("filters_" + str(i), 32, 1024, step=32, default=512),
            kernel_size=hp.Int("kernel_size_" + str(i), 5, 18, default=12),
            activation='relu',
            padding="same",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = tf.keras.layers.MaxPooling1D()(x)
        else:
            x = tf.keras.layers.AveragePooling1D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    act = hp.Choice("act", ["sigmoid", "softmax"], default='softmax')
    outputs = tf.keras.layers.Dense(n_class, activation=act)(x)

    model = tf.keras.Model(inputs, outputs)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log", default=4e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=tf.Variable(learning_rate), clipnorm=1.)
    model.compile(
        optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def main(subj_number, fold, max_trials = 100, epochs = 100):
    ids = [1,3,4,6,7,8,9,10,12,17,18]

    X, Y = data_subject(subj_number)

    global input_shape 
    global n_class

    input_shape = X.shape[1:]

    Y[Y == 7] = 5
    Y = Y - 1
    n_class = len(set(Y)|set(Y))

    xtrain, ytrain, xval, yval, xtest, ytest = ttv_kfold(X,Y,return_fold= fold)

    ytrain = tf.keras.utils.to_categorical(ytrain)
    yval = tf.keras.utils.to_categorical(yval)
    ytest = tf.keras.utils.to_categorical(ytest)

    log_dir = cwd+'trial_logs/' + f'CNN_1D_1_subj{ids[subj_number]}\\fold{fold}'

    hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch')

    tuner_bo = BayesianOptimization(
                hypermodel = build_model,
                objective='val_accuracy',
                max_trials=max_trials,
                seed=42,
                project_name=cwd+'Subject wise\\'+f'CNN_1D_1_subj{ids[subj_number]}\\fold{fold}'
                
            )

    tuner_bo.search(xtrain, ytrain, validation_data=(xval, yval), batch_size=64, epochs=epochs, verbose=1, callbacks = [hist_callback])
    best_model = tuner_bo.get_best_models(num_models=1)[0]
    best_model.evaluate(xtest, ytest)
    best_model.save(cwd+'Subject wise\\'+f'CNN_1D_1_subj{ids[subj_number]}\\'+f'best_model_fold{fold}.h5')

    ytest_pred = best_model.predict(xtest)
    ytest = np.argmax(ytest, axis=1)
    ytest_pred = np.argmax(ytest_pred, axis=1)
    log_path = cwd+'Subject wise\\'+f'CNN_1D_1_subj{ids[subj_number]}\\'

    print(confusion_matrix(ytest,ytest_pred))
    print(ytest_pred.shape)
    print(classification_report(ytest,ytest_pred,digits=4))

    conf_mat = confusion_matrix(ytest,ytest_pred)
    report = classification_report(ytest,ytest_pred,digits=4)
    text_file = open(log_path+f'classification_report_fold{fold}.txt', "w")
    n = text_file.write(report)
    text_file.close()
    np.savetxt(log_path+f'confusion_matrix_fold{fold}.txt',conf_mat,fmt='%d')
    best_model.summary()
    tuner_bo.results_summary(num_trials=1)
    print('subj_index: ', subj_number)


if __name__ == "__main__":
    for subj_number in range(11):
        for fold in range(1,6):
            main(subj_number, fold)
