import copy
from functools import partial

import numpy as np
from joblib import Parallel, delayed

from .RbfInter import predictRBFinter


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


# The objective function for back-end optimization
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


# (1+\lambda)-EA with two mutation rates (self-adjusting)
def _single_start_two_rated(arguments):
    x, nPop, nIter, nVar, acquisition_func, acquisition_type, minimization, r = arguments
    in_x = copy.deepcopy(x)
    fPV, doMin = objective(acquisition_func, acquisition_type, minimization)
    half = nPop // 2
    # x = np.random.randint(2, size=nVar)
    y = fPV(x)[0]
    bestr = 1
    unchanged_iter = 0
    # unchange_t = nIter
    rng = np.random.default_rng()
    hist_x = []
    hist_y = []
    for t in range(nIter):
        pop = np.tile(x, (nPop, 1))
        y_pop = np.empty(nPop)
        nMu = 0
        new_Flag = False
        for i in range(nPop):
            if i < half:
                while nMu == 0:
                    nMu = np.sum(rng.random(nVar) < r / (2 * nVar))
            else:
                while nMu == 0:
                    nMu = np.sum(rng.random(nVar) < 2 * r / nVar)
            index = np.random.permutation(nVar)[:nMu]
            pop[i, index] = 1 - pop[i, index]
            # flip(pop[i, :], nMu, nVar)
            y_pop[i] = fPV(pop[i, :])[0]
        if doMin:
            best_ind = np.argmin(y_pop)
            if y_pop[best_ind] < y:
                hist_y.append(y_pop[best_ind])
                hist_x.append(pop[best_ind, :])
                x = pop[best_ind, :]
                y = y_pop[best_ind]
                new_Flag = True
                unchanged_iter = 0
            else:
                unchanged_iter += 1
        else:
            best_ind = np.argmax(y_pop)
            if y_pop[best_ind] > y:
                hist_y.append(y_pop[best_ind])
                hist_x.append(pop[best_ind, :])
                x = pop[best_ind, :]
                y = y_pop[best_ind]
                new_Flag = True
                unchanged_iter = 0
            else:
                unchanged_iter += 1

        if new_Flag:
            if best_ind < half:
                bestr = r / 2
            else:
                bestr = r * 2
        if np.random.rand() > 0.5:
            r = bestr
        elif np.random.rand() > 0.5:
            r = r / 2
        else:
            r = r * 2
        r = min(r, 2)
        r = max(nVar / 4, r)
        # if unchanged_iter == unchange_t:
        #     break
    # original_unique_index = np.argsort(hist_y)
    xs = np.ones((5, nVar), dtype=int)
    ys = np.empty(5)
    idx = 0
    if doMin:
        for i, item in enumerate(hist_x):
            if idx == 5:
                break
            if not np.array_equal(item, in_x):
                xs[idx, :] = item
                ys[idx] = hist_y[i]
                idx += 1
    else:
        for i, item in enumerate(hist_x):
            if idx == 5:
                break
            if not np.array_equal(item, in_x):
                xs[4 - idx, :] = item
                ys[4 - idx] = hist_y[i]
                idx += 1
    return xs, ys

# Multi-start (1+\lambda)-EA with two mutation rates (self-adjusting)
def onePlusLambdaEA_twoRate_multistart(acquisition_func, acquisition_type, nVar, hist_x, nRandStart, nPop=10,
                                       nIter=None, r=2,
                                       minimization=False, par=None, nTotal=8):
    _, doMin = objective(acquisition_func, acquisition_type, minimization)
    if par is None:
        par = Parallel(n_jobs=nTotal)
    if nIter is None:
        nIter = int(5e2 * nVar / nPop)

    process = []
    for xx in hist_x:
        process.append((xx, nPop, nIter, nVar, acquisition_func, acquisition_type, minimization, r))
    for j in range(nRandStart):
        xx = np.random.randint(2, size=nVar)
        process.append((xx, nPop, nIter, nVar, acquisition_func, acquisition_type, minimization, r))
    results = par(delayed(_single_start_two_rated)(p) for p in process)
    new_xx = np.concatenate(([item[0] for item in results]), axis=0)
    yy = np.concatenate(([item[1] for item in results]), axis=0)
    original_unique_index = np.argsort(yy)
    top_c = nRandStart * 5
    xs = np.ones((top_c, nVar), dtype=int)
    if doMin:
        top = original_unique_index[:top_c]
        for idx, i in enumerate(top):
            xs[idx, :] = new_xx[i]
    else:
        top = original_unique_index[-top_c:]
        for idx, i in enumerate(top):
            xs[top_c - 1 - idx, :] = new_xx[i]
    return xs
