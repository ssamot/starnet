"""
This module gathers counterfactual neural network-based models for classification and regression.
"""

import numpy as np
import keras.backend as K
from keras.utils import to_categorical
from tensorflow import Graph, Session
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from abc import ABCMeta, abstractmethod
from .utils import batch_generator, dump_csv, batch_generator_weighted
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, mean_squared_error



__all__ = ["CounterfactualNNRegressor"]

N_THREADS = 1

# =============================================================================
# Base Counterfactual NN
# =============================================================================


class BaseCounterfactualNN(BaseEstimator, metaclass=ABCMeta):
    """Base class for Counterfactual NN models.

    Warning: this class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, n_iterations, input_size, build_model, batch_size,
                 plot, mixup, is_classifier, n_neurons, loss, enable_propensity):
        self.n_iterations = n_iterations
        self.input_size = input_size
        self.batch_size = batch_size
        self.is_classifier = is_classifier
        self.is_fitted_ = False
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session()

            # K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=N_THREADS,
            #                                      inter_op_parallelism_threads=N_THREADS))

            with self.session.as_default():
                self.model, self.embs = build_model(input_size=input_size, n_neurons=n_neurons, loss=loss,
                                                    enable_propensity=enable_propensity)

        self.plot = plot
        self.mixup = mixup
        self.build_model = build_model
        self.n_neurons = n_neurons
        self.loss = loss
        self.enable_propensity = enable_propensity

    def get_params(self, deep=True):
        return {"n_iterations": self.n_iterations,
                "batch_size": self.batch_size,
                "plot": self.plot,
                "mixup": self.mixup,
                "input_size": self.input_size,
                "build_model": self.build_model,
                "n_neurons": self.n_neurons,
                "loss": self.loss,
                "is_classifier": self.is_classifier,
                "enable_propensity": self.enable_propensity
                }


    def fit(self, X, y, sample_weight=None):
        """
        Build a counterfactual NN estimator from a training set (X, y).

        """
        #print(weights)
        #exit()
        X, y = check_X_y(X, y, accept_sparse=True)
        X, t = X[:, :-1], X[:, -1]  # Last feature is the treatment
        #print(sample_weight)
        n_samples, n_features = X.shape
        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))

        with self.graph.as_default():
            with self.session.as_default():
                self.darken = self.__train(X, t, y, sample_weight)
        self.is_fitted_ = True

        if self.is_classifier:
            self.classes_ = np.array([0, 1])  # only binary classification allowed for now.

        return self

    def __train(self, X, t, y, sample_weight):
        """
        Train the model
        """
        enable_validation = True

        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype=int)
            enable_validation = True

            validation_x = X[sample_weight == 0]
            validation_y = y[sample_weight == 0]
            validation_t = t[sample_weight == 0]

            X = np.repeat(X, sample_weight, axis=0)
            y = np.repeat(y, sample_weight, axis=0)
            t = np.repeat(t, sample_weight, axis=0)

        c_idx = np.where(t == 0)[0]
        t_idx = np.where(t == 1)[0]
        print(len(c_idx), len(t_idx), X.shape)
        # exit()
        #max_batch_size = np.min([len(c_idx), len(t_idx)])
        batch_size_c = np.min([len(c_idx), self.batch_size])
        batch_size_t = np.min([len(t_idx), self.batch_size])
        # print(batch_size_c, batch_size_t, len(X),  np.sum(sample_weight))

        train_c_idx = c_idx
        train_t_idx = t_idx

        c_idx_batch_generator = batch_generator([train_c_idx], batch_size_c, replace=False)
        t_idx_batch_generator = batch_generator([train_t_idx], batch_size_t, replace=False)

        log = []
        activations = []

        best_weights = None
        best_score = 0
        tolerance = 0
        score_val = 0


        # sample_weight = np.array(sample_weight)
        # if((sample_weight.sum() -1) < 0.001):
        #     sample_weight*=len(sample_weight)
        #     #print(sample_weight)

        for i in range(self.n_iterations):
            c_batch_idx = tuple(next(c_idx_batch_generator))
            t_batch_idx = tuple(next(t_idx_batch_generator))

            X_batch_f = np.concatenate([X[c_batch_idx], X[t_batch_idx]])
            y_batch_c = np.concatenate([y[c_batch_idx], np.zeros_like(y[t_batch_idx])])
            y_batch_t = np.concatenate([np.zeros_like(y[c_batch_idx]), y[t_batch_idx]])

            sample_weights_c = np.concatenate([np.ones(shape=len(c_batch_idx[0])),
                                               np.zeros(shape=len(t_batch_idx[0]))])
            sample_weights_t = np.concatenate([np.zeros(shape=len(c_batch_idx[0])),
                                               np.ones(shape=len(t_batch_idx[0]))])

            if self.enable_propensity:  # TODO: Spyros to check this is correct and whether we need separate options
                                        #       depending on the "sample_weight is None" condition
                p_weights = np.concatenate([np.ones(shape=len(c_batch_idx[0])), np.ones(shape=len(t_batch_idx[0]))])
            #
            # print(sample_weights_t.shape)
            # print(sample_weights_c.shape)
            #all_idx = np.concatenate([t_idx, c_idx])
            if self.enable_propensity:
                training_score = self.model.train_on_batch(X_batch_f,
                                                           [y_batch_c, y_batch_t, to_categorical(sample_weights_t)],
                                                           sample_weight=[sample_weights_c, sample_weights_t, p_weights])
            else:
                training_score = self.model.train_on_batch(X_batch_f, [y_batch_c, y_batch_t],
                                                           sample_weight=[sample_weights_c, sample_weights_t])
            #print(training_score)
            ## sharpening

            if i % 1000 == 0 and enable_validation:
                # MSE on training batch
                #print(X.shape, np.array(t).reshape(-1, 1).shape)
                # TODO: this is the full dataset... do we want just one batch?

                if(sample_weight is None):
                    tr_y_hat = self.predict(np.concatenate([X, t[:, np.newaxis]], axis=1))
                    score_tr = -mean_squared_error(y, tr_y_hat)
                    print("Iteration %d: training MSE = %.4f" % (i, -score_tr))
                else:
                    tr_y_hat = self.predict(np.concatenate([X[sample_weight > 0], t[:, np.newaxis][sample_weight > 0]], axis=1))
                    score_tr = -mean_squared_error(y[sample_weight > 0], tr_y_hat)
                    validation_x_t = np.concatenate([validation_x, validation_t[:, np.newaxis]], axis=1)
                    validation_y_hat = self.predict(validation_x_t)
                    score_val = -mean_squared_error(validation_y, validation_y_hat)
                    print("Iteration %d: training MSE = %.4f; validation MSE = %.4f" % (i, -score_tr, -score_val))
                
                # if best_weights is None:
                #     best_weights = self.model.get_weights()
                #     best_score = score_val
                # else:
                #     # best_score > 0.5
                #     if score_val <= best_score:
                #         tolerance += 1
                #         #print(tolerance)
                #     else:
                #         best_weights = self.model.get_weights()
                #         best_score = score_val
                #         tolerance = 0
                # if tolerance > 10:
                #     pass
                #     #self.model.set_weights(best_weights)
                #     #break
        if self.plot:
            dump_csv("data-rand.csv", log, ["Iteration", "Metric", "Type"])
            with open('activations', 'wb') as fp:
                pickle.dump(activations, fp)
        print("BEST SCORE - VALIDATION", int(self.n_neurons), -best_score, -score_val,  i)
        return None

# =============================================================================
# Public estimators
# =============================================================================

class CounterfactualNNRegressor(BaseCounterfactualNN, RegressorMixin):
    """A neural network regressor that can be used for counterfactual things.

     Parameters
     ----------
     n_iterations :  int
                     The number of interations ...

     input_size :    array-like, shape (n_samples,) or (n_samples, n_outputs)
                     The number of


     build_models :  function
                     build_models() should receive an integer as an input (the number of features in the dataset,
                     excluding the treatment variable) and return a regressor that can be trained on the
                     dataset under study.

     batch_size :    int
                     TODO

     plot :          bool
                     Whether to plot TODO

     mixup :         bool
                     TODO


    See also
    --------
    CounterfactualNNRegressor


    References
    ----------
    .. [1] Our shiny paper

     """

    def __init__(self, n_iterations, input_size, build_model, batch_size=32, plot=False, mixup=False, n_neurons=32,
                 loss="mse", is_classifier=False, enable_propensity=False):
        super().__init__(n_iterations=n_iterations, input_size=input_size, build_model=build_model,
                         batch_size=batch_size, plot=plot, mixup=mixup, is_classifier=False, loss=loss,
                         n_neurons=n_neurons, enable_propensity=enable_propensity)

    def fit(self, X, y, sample_weight = None):
        """
        Build a counterfactual NN regressor from the training set (X, y).

        Parameters
        -----------
        X :   array-like, shape (n_samples, n_features)
              The training input samples.
              We assume that the last feature is the presence/absence of treatment in the sample.

        y :  array-like, shape = (n_samples,)
             The target values (real numbers).

        sample_weight :  array-like, shape = (n_samples) or None
                         Sample weights. If None, then samples are equally weighted.
                         Not currently used

        Returns
        -------
        self : object
        """

        #print("TF")
        #exit()

        #sample_weight = None

        super().fit(X, y, sample_weight)
        return self

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y_hat : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        n_samples = X.shape[0]
        X, t = X[:, :-1], np.array(X[:, -1], dtype=int)
        with self.graph.as_default():
            with self.session.as_default():
                if self.is_classifier:  # Classification
                    probabilities = self.predict_proba(X)
                    y_hat = np.argmax(probabilities, axis=1)
                else:  # Regression
                    ct = self.model.predict([X], batch_size=1000)
                    if not self.enable_propensity:
                        ct = np.squeeze(np.array(ct))  # this returns two rows/sample;
                    else:
                        ct = np.squeeze(np.array(ct[0:2]))
                    # the first for the control and the second for the treatment
                    y_hat = np.asarray([ct[t[i], i] for i in range(n_samples)]).reshape(n_samples)
                    #print(y_hat.shape)
                return y_hat

    def predict_embs(self, X):
            with self.graph.as_default():
                with self.session.as_default():
                   return self.embs.predict(X)
