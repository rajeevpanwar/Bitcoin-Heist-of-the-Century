#A custom wrapper for Keras's scikit-learn wrapper class. Function is taken as is
# from https://github.com/keras-team/keras/blob/master/keras/wrappers/scikit_learn.py
# but the fit function has  been customized to include an argument to reset the
# object's states.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import types

import numpy as np

from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.util.tf_export import keras_export


from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


class Keras_Custom_Fitter(KerasRegressor):
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if (losses.is_categorical_crossentropy(self.model.loss) and
                len(y.shape) != 2):
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        history = self.model.fit(x, y, **fit_args)
        self.model.reset_states()

        return history