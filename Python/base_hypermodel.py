import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import keras_tuner as kt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
cwd = os.path.dirname(os.path.abspath(__file__))+'/'
cwd = cwd.replace('\\','/')



class BaseHyperModel(kt.HyperModel):

    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

    def build(self, hp):

        conv_layers_min=6
        conv_layers_max=9
        conv_layers_default=8

        dense_layers_min=0
        dense_layers_max=2
        dense_layers_default=1

        filters_min=4
        filters_max=128
        filters_step=4
        filters_default=48

        kernel_size_min=4
        kernel_size_max=24
        kernel_size_step=2
        kernel_size_default=6

        excluded_pool_layer_ratio=0.4
        excluded_pool_layer_default=2

        pooling_size_default=2

        # For convolution block
        dropout_min=0
        dropout_max=0.5
        dropout_default=0.2

        dense_neurons_min=8
        dense_neurons_max=128
        dense_neurons_step=8
        dense_neurons_default=48

        dropout_dense_min=0
        dropout_dense_max=0.5
        dropout_dense_default=0.2

        kernel_constraint=3.


        learning_rate_min=1e-4
        learning_rate_max=2e-2
        learning_rate_default=1e-3
        """Builds a convolutional model."""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        n_layers = hp.Int("conv_layers", conv_layers_min, conv_layers_max, default=conv_layers_default)
        n_dense_layers = hp.Int("dense_layers", dense_layers_min, dense_layers_max, default=dense_layers_default)

        activation_dict = {"ReLU": tf.keras.activations.relu, "GELU":tf.keras.activations.gelu, "ELU":tf.keras.activations.elu, "SELU":tf.keras.activations.selu, "SoftPlus":tf.keras.activations.softplus, "Swish":tf.keras.activations.swish, "Mish":tfa.activations.mish}

        for i in range(n_layers):
            activation_choice_Conv=hp.Choice("act_Conv_"+ str(i), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = tf.keras.layers.Conv1D(
                filters=hp.Int("filters_" + str(i), filters_min, filters_max, step=filters_step, default=filters_default),
                kernel_size=hp.Int("kernel_size_" + str(i), kernel_size_min, kernel_size_max, step=kernel_size_step, default=kernel_size_default),
                activation=activation_dict[activation_choice_Conv],
                padding='same',
            )(x)

            if 12 < n_layers:
                excluded_pool_layer = np.round(excluded_pool_layer_ratio*n_layers)
            elif 11 < n_layers <= 12:
                excluded_pool_layer = excluded_pool_layer_default+2
            elif 10 < n_layers <= 11:
                excluded_pool_layer = excluded_pool_layer_default+1
            else:
                excluded_pool_layer = excluded_pool_layer_default

            if i < n_layers-excluded_pool_layer :
                pooling_choice = hp.Choice("pooling_" + str(i), ["max", "avg"])
                pooling_size_choice = pooling_size_default
                if pooling_choice == "max":
                    x = tf.keras.layers.MaxPooling1D(pool_size=pooling_size_choice)(x)
                else:
                    x = tf.keras.layers.AveragePooling1D(pool_size=pooling_size_choice)(x)
            normalization_choice = hp.Choice("normalization_" + str(i), ["batch", "layer", "instance"], default="batch")
            
            if normalization_choice == "batch":
                x = tf.keras.layers.BatchNormalization()(x)
            elif normalization_choice == "layer":
                x = tf.keras.layers.LayerNormalization()(x)
            elif normalization_choice == "instance":
                x = tfa.layers.InstanceNormalization()(x)

            dropout_ratio = hp.Float("dropout_" + str(i), dropout_min, dropout_max, default=dropout_default)
            x = tf.keras.layers.Dropout(dropout_ratio)(x)
        
            activation_choice_pooling=hp.Choice("act_pooling_" + str(i), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = activation_dict[activation_choice_pooling](x)

                

        global_dict = {"global_max_pooling": tf.keras.layers.GlobalMaxPooling1D(), "global_average_pooling": tf.keras.layers.GlobalAveragePooling1D(), "Flatten": tf.keras.layers.Flatten()}
        x = global_dict[hp.Choice("pooling_global", ["global_max_pooling", "global_average_pooling", "Flatten"], default="global_max_pooling")](x)

        for j in range(n_dense_layers):
            activation_choice_dense_in=hp.Choice("act_dense_in_"+ str(j), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = tf.keras.layers.Dense(hp.Int("dense_neurons_" + str(j), dense_neurons_min, dense_neurons_max, step=dense_neurons_step, default=dense_neurons_default), activation=activation_dict[activation_choice_dense_in], kernel_constraint=tf.keras.constraints.max_norm(kernel_constraint))(x)
            x = tf.keras.layers.Dropout(hp.Float("dropout_dense_" + str(j), dropout_dense_min, dropout_dense_max, default=dropout_dense_default))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            activation_choice_dense_out=hp.Choice("act_dense_out_"+ str(j), ["ReLU", "Swish", "Mish"], default="ReLU")
            x = activation_dict[activation_choice_dense_out](x)

        output_act = hp.Choice("output_act", ["softmax", "sigmoid"] , default="softmax")
        outputs = tf.keras.layers.Dense(self.n_class, activation=output_act)(x)

        model = tf.keras.Model(inputs, outputs)

        learning_rate = hp.Float("learning_rate", learning_rate_min, learning_rate_max, sampling="log", default=learning_rate_default)
        optimizer = tf.keras.optimizers.Adam(learning_rate=tf.Variable(learning_rate), clipnorm=1.)
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy"] 
            # tfa.metrics.FBetaScore(num_classes=self.n_class, beta=2.0, threshold=0.5, average = 'weighted'),
            # tfa.metrics.F1Score(num_classes=self.n_class, threshold=0.5, average = 'weighted')]
        )

        # model.summary()
        return model
