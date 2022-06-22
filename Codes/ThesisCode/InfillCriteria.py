# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: wangronin
_________________________________________________________
Qi: modified the base class to fit in current pipeline

"""

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import sqrt, exp
from scipy.stats import norm

from .RandomForest import predictRandomForest
from .RbfInter import predictRBFinter
from .SVMSklearn import predictSVM


# warnings.filterwarnings("error")

# TODO: perphas also enable acquisition function engineering here?
# meaning the combination of the acquisition functions
class InfillCriteria:
    __metaclass__ = ABCMeta

    def __init__(self, model, modelType=None, plugin=None, minimize=True):
        # plugin is the best so far fitness value found
        if modelType is None or model is None or plugin is None:
            raise ValueError("At least three parameters are needed (model, modelType, plugin)")
        self.model = model
        self.modelType = modelType
        self.minimize = minimize
        # change maximization problem to minimization
        self.plugin = plugin if self.minimize else -plugin
        # if self.plugin is None:
        #     self.plugin = np.min(model.y) if minimize else -np.max(self.model.y)

    @abstractmethod
    def __call__(self, X):
        raise NotImplementedError

    def _predict(self, X):
        if self.modelType == 'Kriging':
            y_hat = self.model.predict_values(X)
            sd2 = self.model.predict_variances(X)
            sd = sqrt(sd2)
        elif self.modelType == 'RBF':
            results = predictRBFinter(self.model, X, True)
            results = np.array(results)
            y_hat = np.reshape(results[:, 0], (X.shape[0],))
            sd2 = np.reshape(results[:, 1], (X.shape[0],))
            sd = sqrt(sd2)
        elif self.modelType == 'RF':
            y_hat, sd2 = predictRandomForest((X, self.model, True))
            sd = sqrt(sd2)
        elif self.modelType == 'SVM':
            y_hat = predictSVM((X, self.model))
            sd = 0
        else:
            raise NotImplementedError("Models are not supported.")
        if not self.minimize:
            y_hat = -y_hat
        return y_hat, sd

    def _gradient(self, X):
        y_dx, sd2_dx = self.model.gradient(X)
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd2_dx

    def check_X(self, X):
        """Keep input as '2D' object
        """
        return np.atleast_2d(X)
        # return [X] if not hasattr(X[0], '__iter__') else X


class EI(InfillCriteria):
    """
    Expected Improvement
    """

    # perhaps separate the gradient computation here
    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)
        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-10 and why the ratio if intended
        # TODO: implement a counterpart of 'sigma2' for randomforest

        if self.modelType == 'SVM':
            return y_hat

        if sd < 1e-8:
            f_value = (np.array([0.]), np.zeros((len(X[0]), 1))) if dx else np.array([0.])
            return f_value
        try:
            # TODO: I have save xcr_ becasue xcr * sd != xcr_ numerically
            # find out the cause of such an error, probably representation error...
            xcr_ = self.plugin - y_hat
            xcr = xcr_ / sd
            xcr_prob, xcr_dens = norm.cdf(xcr), norm.pdf(xcr)
            f_value = xcr_ * xcr_prob + sd * xcr_dens
        except Exception:  # in case of numerical errors
            # IMPORTANT: always keep the output in the same type
            f_value = np.array([0.])

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = -y_dx * xcr_prob + sd_dx * xcr_dens
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx
        return f_value


class MGFI(InfillCriteria):
    """
    Moment-Generating Function of Improvement
    My new acquisition function proposed in SMC'17 paper
    """

    def __init__(self, model, modelType=None, plugin=None, minimize=True, t=0.1):
        super(MGFI, self).__init__(model, modelType, plugin, minimize)
        self.t = t

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if np.isclose(sd, 0):
            return (np.array([0.]), np.zeros((len(X[0]), 1))) if dx else 0.

        try:
            y_hat_p = y_hat - self.t * sd ** 2.
            beta_p = (self.plugin - y_hat_p) / sd
            term = self.t * (self.plugin - y_hat - 1)
            f_ = norm.cdf(beta_p) * exp(term + self.t ** 2. * sd ** 2. / 2.)
        except Exception:  # in case of numerical errors
            f_ = np.array([0.])

        if np.isinf(f_):
            f_ = np.array([0.])

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)

            try:
                term = exp(self.t * (self.plugin + self.t * sd ** 2. / 2 - y_hat - 1))
                m_prime_dx = y_dx - 2. * self.t * sd * sd_dx
                beta_p_dx = -(m_prime_dx + beta_p * sd_dx) / sd

                f_dx = term * (norm.pdf(beta_p) * beta_p_dx +
                               norm.cdf(beta_p) * ((self.t ** 2) * sd * sd_dx - self.t * y_dx))
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_, f_dx
        return f_
