from __future__ import print_function

from copy import copy
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from numpy import exp, argsort, ceil, zeros, mod
from numpy.random import randint, rand, randn, geometric

from .RbfInter import predictRBFinter
from .utils import boundary_handling


#######################################
# Docker for different models
def KrgDocker(X, model):
    X = np.reshape(X, (1, -1))
    return model.predict_values(X)


def RBFDocker(X, rbfmodel, uncertainty):
    X = np.reshape(X, (1, -1))
    return predictRBFinter(rbfmodel, X, uncertainty)


def ICDocker(X, model):
    return model(X),


def SVMDocker(X, model):
    X = np.reshape(X, (1, -1))
    return model.predict(X)


def RFDocker(X, model):
    X = np.reshape(X, (1, -1))
    return model.predict(X, eval_MSE=False)
#######################################


def objective(model, modelType, minimization):
    # 'RBF' or 'Kriging'
    if modelType == 'Kriging':
        # fPV = model.predict_values
        fPV = partial(KrgDocker, model=model)
    elif modelType == 'RBF':
        # fPV = partial(predictRBFinter, rbfmodel=model, uncertainty=False)
        fPV = partial(RBFDocker, rbfmodel=model, uncertainty=False)
    elif modelType == 'MGFI' or modelType == 'EI':
        fPV = partial(ICDocker, model=model)
        minimization = False
    elif modelType == 'SVM':
        fPV = partial(SVMDocker, model=model)
    elif modelType == 'RF':
        fPV = partial(RFDocker, model=model)
    else:
        raise NotImplementedError
    return fPV, minimization

"""
Information of authors regarding Mi-ES
Created on Thu Sep  7 11:10:18 2017

@author: wangronin
"""
class Individual(list):
    """Make it possible to index Python list object using the enumerables
    """

    def __getitem__(self, keys):
        if isinstance(keys, int):
            return super(Individual, self).__getitem__(keys)
        elif hasattr(keys, '__iter__'):
            return Individual([super(Individual, self).__getitem__(int(key)) for key in keys])

    def __setitem__(self, index, values):
        # In python3 hasattr(values, '__iter__') returns True for string type...
        if hasattr(values, '__iter__') and not isinstance(values, str):
            values = Individual([_ for _ in values])
        else:
            values = [values]

        if not hasattr(index, '__iter__'):
            index = int(index)
            if hasattr(values, '__iter__'):
                if len(values) == 1:
                    values = values[0]
                else:
                    values = Individual([_ for _ in values])
            super(Individual, self).__setitem__(index, values)
        else:
            index = [i for i in index]
            if len(index) == 1:
                index = index[0]
                if len(values) == 1:
                    values = values[0]
                super(Individual, self).__setitem__(index, values)
            else:
                assert len(index) == len(values)
                for i, k in enumerate(index):
                    super(Individual, self).__setitem__(k, values[i])

    def __add__(self, other):
        return Individual(list.__add__(self, other))

    def __mul__(self, other):
        return Individual(list.__mul__(self, other))

    def __rmul__(self, other):
        return Individual(list.__mul__(self, other))


# TODO: improve efficiency, e.g. compile it with cython
class mies(object):
    def __init__(self, search_space, obj_func, x0=None, ftarget=None, max_eval=np.inf,
                 minimize=True, mu_=10, lambda_=70, sigma0=None, eta0=None, P0=None, plus_selection=False,
                 multiple_return=False,
                 verbose=False):

        self.verbose = verbose
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.eval_count = 0
        self.iter_count = 0
        self.max_eval = max_eval
        self.plus_selection = False
        self.minimize = minimize
        self.obj_func = obj_func
        self.stop_dict = {}
        self.hist_best_x = []
        self.hist_best_y = []
        self.hist_best_y_ifmax = []
        self._space = search_space
        self.var_names = self._space.var_name.tolist()
        self.param_type = self._space.var_type

        # index of each type of variables in the dataframe
        self.id_r = self._space.id_C  # index of continuous variable
        self.id_i = self._space.id_O  # index of integer variable
        self.id_d = self._space.id_N  # index of categorical variable

        # the number of variables per each type
        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)
        self.dim = self.N_r + self.N_i + self.N_d

        # by default, we use individual step sizes for continuous and integer variables
        # and global strength for the nominal variables
        self.N_p = min(self.N_d, int(1))
        self._len = self.dim + self.N_r + self.N_i + self.N_p

        # unpack interval bounds
        self.bounds_r = np.asarray([self._space.bounds[_] for _ in self.id_r])
        self.bounds_i = np.asarray([self._space.bounds[_] for _ in self.id_i])
        self.bounds_d = np.asarray([self._space.bounds[_] for _ in self.id_d])  # actually levels...

        # step default step-sizes/mutation strength
        if sigma0 is None and self.N_r:
            sigma0 = 0.05 * (self.bounds_r[:, 1] - self.bounds_r[:, 0])
        if eta0 is None and self.N_i:
            eta0 = 0.05 * (self.bounds_i[:, 1] - self.bounds_i[:, 0])
        if P0 is None and self.N_d:
            P0 = 1. / self.N_d

        # column names of the dataframe: used for slicing
        self._id_var = np.arange(self.dim)
        self._id_sigma = np.arange(self.N_r) + len(self._id_var) if self.N_r else []
        self._id_eta = np.arange(self.N_i) + len(self._id_var) + len(self._id_sigma) if self.N_i else []
        self._id_p = np.arange(self.N_p) + len(self._id_var) + len(self._id_sigma) + len(
            self._id_eta) if self.N_p else []
        self._id_hyperpar = np.arange(self.dim, self._len)

        self.multiple_return = multiple_return

        # initialize the populations
        if x0 is not None:  # given x0
            par = []
            if self.N_r:
                par += [sigma0]
            elif self.N_i:
                par += [eta0]
            elif self.N_p:
                par += [P0] * self.N_p
                # individual0 = Individual(np.r_[x0, eta0, [P0] * self.N_p])
            # individual0 = Individual(np.r_[x0, sigma0, eta0, [P0] * self.N_p])
            individual0 = Individual(np.r_[x0, par[0]])
            self.pop_mu = Individual([individual0]) * self.mu_
            fitness0 = self.evaluate(self.pop_mu[0])
            self.f_mu = np.repeat(fitness0, self.mu_)
            self.xopt = x0
            self.fopt = sum(fitness0)
        else:
            x = np.asarray(self._space.sampling(self.mu_), dtype='object')  # uniform sampling
            par = []
            if self.N_r:
                par += [np.tile(sigma0, (self.mu_, 1))]
            if self.N_i:
                par += [np.tile(eta0, (self.mu_, 1))]
            if self.N_p:
                par += [np.tile([P0] * self.N_p, (self.mu_, 1))]

            par = np.concatenate(par, axis=1)
            self.pop_mu = Individual([Individual(_) for _ in np.c_[x, par].tolist()])
            self.f_mu = self.evaluate(self.pop_mu)
            self.fopt = min(self.f_mu) if self.minimize else max(self.f_mu)
            try:
                a = int(np.nonzero(self.fopt == self.f_mu)[0][0])
            except IndexError:
                print(self.fopt)
                print(self.f_mu)
                print(np.nonzero(self.fopt == self.f_mu))
                a = np.argmin(self.f_mu) if self.minimize else np.argmax(self.f_mu)
                # raise IndexError
            self.xopt = self.pop_mu[a][self._id_var]

        self.pop_lambda = Individual([self.pop_mu[0]]) * self.lambda_
        self._set_hyperparameter()

        # stop criteria
        self.tolfun = 1e-5
        self.nbin = int(3 + ceil(30. * self.dim / self.lambda_))
        self.histfunval = zeros(self.nbin)

    def _set_hyperparameter(self):
        # hyperparameters: mutation strength adaptation
        if self.N_r:
            self.tau_r = 1 / np.sqrt(2 * self.N_r)
            self.tau_p_r = 1 / np.sqrt(2 * np.sqrt(self.N_r))

        if self.N_i:
            self.tau_i = 1 / np.sqrt(2 * self.N_i)
            self.tau_p_i = 1 / np.sqrt(2 * np.sqrt(self.N_i))

        if self.N_d:
            self.tau_d = 1 / np.sqrt(2 * self.N_d)
            self.tau_p_d = 1 / np.sqrt(2 * np.sqrt(self.N_d))

    def recombine(self, id1, id2):
        p1 = copy(self.pop_mu[id1])  # IMPORTANT: this copy is necessary
        if id1 != id2:
            p2 = self.pop_mu[id2]
            # intermediate recombination for the mutation strengths
            p1[self._id_hyperpar] = (np.array(p1[self._id_hyperpar]) +
                                     np.array(p2[self._id_hyperpar])) / 2
            # dominant recombination
            mask = randn(self.dim) > 0.5
            p1[mask] = p2[mask]
        return p1

    def select(self):
        pop = self.pop_mu + self.pop_lambda if self.plus_selection else self.pop_lambda
        fitness = np.r_[self.f_mu, self.f_lambda] if self.plus_selection else self.f_lambda

        fitness_rank = argsort(fitness)
        if not self.minimize:
            fitness_rank = fitness_rank[::-1]

        sel_id = fitness_rank[:self.mu_]
        self.pop_mu = pop[sel_id]
        self.f_mu = fitness[sel_id]

    def evaluate(self, pop):
        if not hasattr(pop[0], '__iter__'):
            pop = [pop]
        N = len(pop)
        f = np.zeros(N)
        for i, individual in enumerate(pop):
            var = individual[self._id_var]
            f[i] = np.sum(self.obj_func(var)[0])  # in case a 1-length array is returned
            self.eval_count += 1
        return f

    def mutate(self, individual):
        if self.N_r:
            self._mutate_r(individual)
        if self.N_i:
            self._mutate_i(individual)
        if self.N_d:
            self._mutate_d(individual)
        return individual

    def _mutate_r(self, individual):
        sigma = np.array(individual[self._id_sigma])
        if len(self._id_sigma) == 1:
            sigma = sigma * exp(self.tau_r * randn())
        else:
            sigma = sigma * exp(self.tau_r * randn() + self.tau_p_r * randn(self.N_r))

        # Gaussian mutation
        R = randn(self.N_r)
        x = np.array(individual[self.id_r])
        x_ = x + sigma * R

        # Interval Bounds Treatment
        x_ = boundary_handling(x_, self.bounds_r[:, 0], self.bounds_r[:, 1])

        # Repair the step-size if x_ is out of bounds
        individual[self._id_sigma] = np.abs((x_ - x) / R)
        individual[self.id_r] = x_

    def _mutate_i(self, individual):
        eta = np.array(individual[self._id_eta])
        x = np.array(individual[self.id_i])
        if len(self._id_eta) == 1:
            eta = max(1, eta * exp(self.tau_i * randn()))
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            x_ = x + geometric(p, self.N_i) - geometric(p, self.N_i)
        else:
            eta = eta * exp(self.tau_i * randn() + self.tau_p_i * randn(self.N_i))
            eta[eta > 1] = 1
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            x_ = x + np.array([geometric(p_) - geometric(p_) for p_ in p])

        # TODO: implement the same step-size repairing method here
        x_ = boundary_handling(x_, self.bounds_i[:, 0], self.bounds_i[:, 1])
        individual[self._id_eta] = eta
        individual[self.id_i] = x_

    def _mutate_d(self, individual):
        P = np.array(individual[self._id_p])
        P = 1 / (1 + (1 - P) / P * exp(-self.tau_d * randn()))
        individual[self._id_p] = boundary_handling(P, 1 / (3. * self.N_d), 0.5)[0].tolist()

        idx = np.nonzero(rand(self.N_d) < P)[0]
        for i in idx:
            level = self.bounds_d[i]
            individual[self.id_d[i]] = level[randint(0, len(level))]

    def stop(self):
        if self.eval_count > self.max_eval:
            self.stop_dict['max_eval'] = True

        if self.eval_count != 0 and self.iter_count != 0:
            fitness = self.f_lambda

            # tolerance on fitness in history
            self.histfunval[int(mod(self.eval_count / self.lambda_ - 1, self.nbin))] = fitness[0]
            if mod(self.eval_count / self.lambda_, self.nbin) == 0 and \
                    (max(self.histfunval) - min(self.histfunval)) < self.tolfun:
                self.stop_dict['tolfun'] = True

            # flat fitness within the population
            if fitness[0] == fitness[int(max(ceil(.1 + self.lambda_ / 4.), self.mu_ - 1))]:
                self.stop_dict['flatfitness'] = True

        return any(self.stop_dict.values())

    def _better(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2

    def optimize(self):
        while not self.stop():
            for i in range(self.lambda_):
                p1, p2 = randint(0, self.mu_), randint(0, self.mu_)
                individual = self.recombine(p1, p2)
                self.pop_lambda[i] = self.mutate(individual)

            self.f_lambda = self.evaluate(self.pop_lambda)
            self.select()

            curr_best = self.pop_mu[0]
            xopt_, fopt_ = curr_best[self._id_var], self.f_mu[0]
            xopt_[self.id_i] = list(map(int, xopt_[self.id_i]))

            self.iter_count += 1
            if self.multiple_return:
                ind = np.searchsorted(self.hist_best_y_ifmax, fopt_) \
                    if self.minimize else np.searchsorted(self.hist_best_y_ifmax, -fopt_)
                self.hist_best_y_ifmax.insert(ind, fopt_) \
                    if self.minimize else self.hist_best_y_ifmax.insert(ind, -fopt_)
                self.hist_best_y.insert(ind, fopt_)
                self.hist_best_x.insert(ind, xopt_)
                if len(self.hist_best_y) > 10:
                    self.hist_best_y = self.hist_best_y[:10]
                    self.hist_best_x = self.hist_best_x[:10]
                    self.hist_best_y_ifmax = self.hist_best_y_ifmax[:10]

            if self._better(fopt_, self.fopt):
                self.xopt, self.fopt = xopt_, fopt_

            if self.verbose:
                print('iteration ', self.iter_count + 1)
                print(self.xopt, self.fopt)

        self.stop_dict['funcalls'] = self.eval_count
        if self.multiple_return:
            if len(self.hist_best_y) > 5:
                hist_best_x = np.array(self.hist_best_x[:5], ndmin=2, dtype=int)
                hist_best_y = np.array(self.hist_best_y[:5])
                return hist_best_x[:5], hist_best_y[:5], self.stop_dict
            elif len(self.hist_best_y) < 1:
                hist_best_x = np.array(self.xopt, ndmin=2, dtype=int)
                hist_best_y = np.array([self.fopt])
                return hist_best_x, hist_best_y, self.stop_dict
            else:
                hist_best_x = np.array(self.hist_best_x, ndmin=2, dtype=int)
                hist_best_y = np.array(self.hist_best_y)
                return hist_best_x, hist_best_y, self.stop_dict
        else:
            return np.array(self.xopt, ndmin=2, dtype=int), np.array(self.fopt, ndmin=1), self.stop_dict
# End of MI-ES
##################################################################


# Call MI-ES in our context
def _single_mies(arguments):
    x, input_space, nIter, acquisition_func, acquisition_type, minimization, flag, multi_return = arguments
    fPV, doMin = objective(acquisition_func, acquisition_type, minimization)
    # mu_ = input_space.dim
    # lambda_ = mu_ * 7
    mu_ = 10
    lambda_ = mu_ * 7
    if flag:
        opt = mies(input_space, fPV, mu_=mu_, lambda_=lambda_, max_eval=nIter, verbose=False, minimize=doMin,
                   x0=x, multiple_return=multi_return)
    else:
        opt = mies(input_space, fPV, mu_=mu_, lambda_=lambda_, max_eval=nIter, verbose=False, minimize=doMin,
                   multiple_return=multi_return)
    xopt, fopt, stop_dict = opt.optimize()
    return xopt, fopt, stop_dict


# Interface of multi-start MI-ES
def MIES_multistart(acquisition_func, acquisition_type, nVar, space, hist_x, nRandStart,
                    nIter=None, minimization=False, multi_return=True, out_optimizer_log=None, par=None, nTotal=6):
    if par is None:
        par = Parallel(n_jobs=nTotal)
    if nIter is None:
        nIter = int(1e3 * nVar)
        # if nIter < 25000:
        #     nIter = 25000
    _, doMin = objective(acquisition_func, acquisition_type, minimization)
    process = []
    # space = OrdinalSpace([0, 2], 'I') * nVar
    for xx in hist_x:
        flag = True
        process.append((xx, space, nIter, acquisition_func, acquisition_type, minimization, flag, multi_return))
    for j in range(nRandStart):
        # xx = np.random.randint(2, size=nVar)
        xx = space.sampling()[0]
        flag = False
        process.append((xx, space, nIter, acquisition_func, acquisition_type, minimization, flag, multi_return))
    results = par(delayed(_single_mies)(p) for p in process)
    fFirst = True
    for item in results:
        if item[0].size > 0:
            if fFirst:
                new_xx = item[0]
                yy = item[1]
                fFirst = False
            else:
                tmp = new_xx
                new_xx = np.concatenate([tmp, item[0]], axis=0)
                tmp = yy
                yy = np.concatenate(([tmp, item[1]]), axis=0)

        # print(item[2])
    # new_xx = np.concatenate(, axis=0)
    # yy = np.concatenate(([item[1] for item in results]), axis=0)
    original_unique_index = np.argsort(yy)
    top_c = 10 if len(original_unique_index) > 10 else len(original_unique_index)
    top_c = 15 if len(original_unique_index) > 15 else top_c
    xs = np.ones((top_c, nVar), dtype=int)
    if doMin:
        top = original_unique_index[:top_c]
        for idx, i in enumerate(top):
            xs[idx, :] = new_xx[i]
    else:
        top = original_unique_index[:top_c]
        for idx, i in enumerate(top):
            xs[top_c - 1 - idx, :] = new_xx[i]
    return xs
