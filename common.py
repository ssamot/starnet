from keras.layers import Input, Activation, Dropout, GaussianNoise, multiply, Lambda, concatenate
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.layers import PReLU, LeakyReLU
import tensorflow as tf


batch_size = 512
#scaler_x = StandardScaler()
#scaler_y = StandardScaler()


scaler_x = MinMaxScaler(feature_range=(-1,1))
scaler_y = MinMaxScaler(feature_range=(-1,1))
#scaler_y1 = MinMaxScaler(feature_range=(-1,1))

#scaler_x = RobustScaler()
#scaler_y = RobustScaler()

last_layer_neurons = "sigmoid"
#last_layer_neurons = "linear"

def get_optimizer():
    """
    Optimizer
    """
    return Adam(learning_rate=0.001)
    #return SGD(0.001, nesterov=True, momentum=0.5)


def get_regularizer_loss():
    """
    Regularizer loss to use
    """
    return l2(0.001)
    #None


def get_kernel_constraints():
    """
    Kernel constraint for network parameters during optimization
    """
    return None


def build_model(input_size, n_neurons=2, loss='mse', enable_propensity=True):
    """
    Creates model for regression.
    In this example we create two different models, one that does the regression on the IHDP dataset
    and one that embeds the original number of features and can be used for visualisation.
    Only the regression model is used by CounterfactualNNRegressor, so we don't return the embeddings model

    Param:
    ------
    input_size :        int
                        number of features (after removing the 'treatment' variable)

    n_neurons :     int
                    number of outputs of the model

    loss :          str
                    Loss for the compilation of the model
                    
    enable_propensity :    bool
                    Whether to use propensity scores in the model

    Returns:
    --------
    model :             Regressor
                        The model will be trained on the IHDP dataset.
                        It has two heads (i.e., outputs): one for the control and one for the treatment.

    """

    # Embeddings model
    embedding_input = Input(shape=(input_size,))
    embedding = embedding_input
    # for _ in range(1):  # first two layers are 64 hidden neurons with ELU activation
    #     embedding = Dense(256, activation='linear', kernel_constraint=get_kernel_constraints(),
    #                     kernel_regularizer=get_regularizer_loss(), use_bias=False)(embedding)
    #     embedding = BatchNormalization()(embedding)
    #     embedding = Activation("elu")(embedding)
    #
    #     embedding = Dropout(0.5)(embedding)
    #output layer: n_neurons neurons with tanh activations
    n_neurons = int(n_neurons)
    if n_neurons > 0.5:
        embedding = Dense(n_neurons, activation='linear', kernel_constraint=get_kernel_constraints(),
                          kernel_regularizer=get_regularizer_loss(), use_bias=False)(embedding)
        embedding = BatchNormalization()(embedding)
        embedding = Activation("tanh")(embedding)

        #embedding = Dropout(0.5)(embedding)
        #embedding = PReLU()(embedding)
        #embedding = GaussianNoise(0.01)(embedding)

    # Regressor model will receive the outputs of the embeddings model as inputs
    if n_neurons > 0.5:
        regressor_input = Input(shape=(n_neurons,))
    else:
        regressor_input = Input(shape=(input_size,))
    regressor = regressor_input
    
    if enable_propensity:
        propensity = BatchNormalization()(regressor)
        propensity = Dense(2, kernel_regularizer=None, activation="softmax")(propensity)
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(propensity)
        
    regressor_treatment = BatchNormalization()(regressor)
    if enable_propensity:
        regressor_treatment = Dense(1, kernel_regularizer=get_regularizer_loss(),
                                    activation=last_layer_neurons)(concatenate([regressor_treatment, splits[1]]))
    else:
        regressor_treatment = Dense(1, kernel_regularizer=get_regularizer_loss(),
                                    activation=last_layer_neurons)(regressor_treatment)
    
    regressor_treatment = Model(inputs=[regressor_input], outputs=[regressor_treatment], name='t_model')

    regressor_control = BatchNormalization()(regressor)
    if enable_propensity:
        regressor_control = Dense(1, kernel_regularizer=get_regularizer_loss(),
                                  activation=last_layer_neurons)(concatenate([regressor_control, splits[0]]))
    else:
        regressor_control = Dense(1, kernel_regularizer=get_regularizer_loss(),
                                  activation=last_layer_neurons)(regressor_control)
    # regressor_control = BatchNormalization()(regressor_control)
    regressor_control = Model(inputs=[regressor_input], outputs=[regressor_control], name='c_model')
    # The regressor has two outputs: one for the factual and one for the counterfactual.
    # The first output is the 'control' case (i.e., no treatment, t=0);
    # the second is the 'treatment' case (i.e., t=1)

    if enable_propensity:
        propensity = Model(inputs=[regressor_input], outputs=[propensity], name='propensity')
        model = Model(inputs=[embedding_input],
                      outputs=[regressor_control([embedding]),
                               regressor_treatment([embedding]),
                               propensity([embedding])])
        model.compile(optimizer=get_optimizer(), loss={'c_model': loss, 't_model': loss, 'propensity': loss},
                      loss_weights=[1, 1, 1])
    else:
        model = Model(inputs=[embedding_input],
                      outputs=[regressor_control([embedding]), regressor_treatment([embedding])])
        model.compile(optimizer=get_optimizer(), loss={'c_model': loss, 't_model': loss})
    # print(embedding)
    embeddings_model = Model(inputs=[embedding_input], outputs=[embedding])
    # embeddings_model.compile(optimizer=get_optimizer(), loss="mse")

    return model, embeddings_model