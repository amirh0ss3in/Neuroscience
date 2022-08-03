import tensorflow as tf 
import os

def load(subj,version=None):
    cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
    folder = "Archive\\Subject wise\\"
    # subj = "CNN_1D_1_subj1\\"
    if version is not None:
        path = cwd+folder+subj+f"\\best_model_{version}.h5"
    else:
        path = cwd+folder+subj+"\\best_model.h5"
    # print(path)
    model = tf.keras.models.load_model(path)
    # model.summary()
    return model
