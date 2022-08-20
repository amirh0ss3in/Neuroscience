import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import keras_tuner as kt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
cwd = os.path.dirname(os.path.abspath(__file__))+'/'


class BaseHyperModel(kt.HyperModel):

    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

    def build(self, hp):

        """Builds a convolutional model."""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        n_layers = hp.Int("conv_layers", 3, 8, default=5)
        n_dense_layers = hp.Int("dense_layers", 0, 4, default=1)

        activation_dict = {"ReLU":tf.keras.activations.relu, "Swish":tf.keras.activations.swish, "Mish":tfa.activations.mish}
        pooling_dict = {"max":tf.keras.layers.MaxPooling1D(), "avg":tf.keras.layers.AveragePooling1D()}
        normalization_dict = {"batch":tf.keras.layers.BatchNormalization, "instance":tfa.layers.InstanceNormalization}

        for i in range(n_layers):
            activation_choice_Conv=hp.Choice("act_Conv_"+ str(i), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = tf.keras.layers.Conv1D(
                filters=hp.Int("filters_" + str(i), 4, 128, step=4, default=48),
                kernel_size=hp.Int("kernel_size_" + str(i), 4, 128, step=4, default=48),
                activation=activation_dict[activation_choice_Conv],
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float("l2_" + str(i), 0, 0.05, default=0.01))
            )(x)

            pooling_choice = hp.Choice("pooling_" + str(i), ["max", "avg"])
            if i < n_layers-2:
                x = pooling_dict[pooling_choice](x)

            normalization_choice = hp.Choice("normalization_" + str(i), ["batch", "instance"], default="batch")
            x = normalization_dict[normalization_choice]()(x)
            x = tf.keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.5, default=0.2))(x)
            activation_choice_pooling=hp.Choice("act_pooling_" + str(i), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = activation_dict[activation_choice_pooling](x)
        
        global_dict = {"global_max_pooling": tf.keras.layers.GlobalMaxPooling1D(), "global_average_pooling": tf.keras.layers.GlobalAveragePooling1D(), "Flatten": tf.keras.layers.Flatten()}
        x = global_dict[hp.Choice("pooling_global", ["global_max_pooling", "global_average_pooling", "Flatten"], default="global_max_pooling")](x)

        for j in range(n_dense_layers):
            activation_choice_dense_in=hp.Choice("act_dense_in_"+ str(j), ["ReLU", "Swish", "Mish"], default="ReLU")
            activation_choice_dense_out=hp.Choice("act_dense_out"+ str(j), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = tf.keras.layers.Dense(hp.Int("dense_" + str(j), 8, 128, step=8, default=32), activation=activation_dict[activation_choice_dense_in])(x)
            x = tf.keras.layers.Dropout(hp.Float("dropout_dense_" + str(j), 0, 0.5, default=0.2))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = activation_dict[activation_choice_dense_out](x)
        act = hp.Choice("act", ["sigmoid", "softmax"], default='softmax')
        outputs = tf.keras.layers.Dense(self.n_class, activation=act)(x)
        model = tf.keras.Model(inputs, outputs)

        learning_rate = hp.Float("learning_rate", 2e-5, 2e-2, sampling="log", default=1e-3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=tf.Variable(learning_rate), clipnorm=1.)
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy", 
            tfa.metrics.FBetaScore(num_classes=self.n_class, beta=2.0, threshold=0.5, average = 'weighted'),
            tfa.metrics.F1Score(num_classes=self.n_class, threshold=0.5, average = 'weighted')]
        )
        # model.summary()
        return model
