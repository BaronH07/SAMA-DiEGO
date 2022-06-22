import copy
import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .KrigingSMT import predictKriging
from .RBFRoy import predictRoyRBFmodel
from .RandomForest import predictRandomForest
from .SVMSklearn import predictSVM


def ModelSelection(X, y_true, fCrit, nKriging=0, nRBF=0, KrgModels=None, RBFModels=None):
    if X is None or y_true is None or fCrit is None:
        raise ValueError('At least three parameters are needed (X, y_true, fCrit)')
    AllLoss = np.empty(nRBF + nKriging)
    RBFResults = [[] for i in range(nRBF)]
    krgResults = [[] for i in range(nKriging)]
    try:
        in_shape = y_true.shape[0]
    except Exception:
        in_shape = np.isscalar(y_true) * 1
    if in_shape == 1:
        X_array = np.reshape(X, (-1, X.shape[0]))
        if nRBF > 0:
            for idx, model in enumerate(RBFModels):
                # RBFResults[idx] = predictRoyRBFmodel((X_array, model, False))
                if model is None:
                    AllLoss[idx] = np.inf
                else:
                    AllLoss[idx] = abs(y_true - predictRoyRBFmodel((X_array, model, False)))
        if nKriging > 0:
            for idx, model in enumerate(KrgModels):
                if model is None:
                    AllLoss[idx+nRBF] = np.inf
                else:
                    AllLoss[idx+nRBF] = abs(y_true - predictKriging((X_array, model, False)))
                # krgResults[idx] = predictSMTKrigingModel((X_array, model, False))
    elif in_shape > 1:
        if nRBF > 0:
            # RBFResults = [[] for i in range(nRBF)]
            pool = multiprocessing.Pool(processes=nRBF)
            process = [(copy.deepcopy(X), copy.deepcopy(RBFModels[int(model)]), False) for model in range(nRBF)]
            RBFResults = pool.map(predictRoyRBFmodel, process)
            pool.close()

            for ind, result in enumerate(RBFResults):
                result = np.reshape(result, (y_true.shape[0], -1))
                AllLoss[ind] = fCrit(y_true, result)

        if nKriging > 0:
            # krgResults = [[] for i in range(nKriging)]
            pool = multiprocessing.Pool(processes=nKriging)
            process = [(copy.deepcopy(X), copy.deepcopy(KrgModels[model]), False) for model in range(nKriging)]
            krgResults = pool.map(predictKriging, process)
            pool.close()
            # evaluate Kriging
            for ind, result in enumerate(krgResults):
                result = np.reshape(result, (y_true.shape[0], -1))
                AllLoss[ind + nRBF] = fCrit(y_true, result)
    else:
        raise Exception("Invalid input in model selection")
    # Sorted index
    # [2,1,4,3] ->(sort)-> [1,0,3,2]
    sorted_ind = np.argsort(np.argsort(AllLoss))
    # best_ind = np.argmin(AllLoss)
    # else:
    #     if nRBF > 0:
    #         RBFResults = np.empty()
    #         for id, model in enumerate(RBFModels):
    #             RBFResults[id] =
    #

    # Old version, only focus on the best new
    # if best_ind < nRBF:
    #     return RBFModels[best_ind], 'RBF', best_ind
    # else:
    #     return KrgModels[best_ind - nRBF], 'Kriging', best_ind -nRBF

    return sorted_ind, AllLoss


def ModelRank(X, y_true, fCrit, Allmodels, AllType, nTotal):
    if X is None or y_true is None or fCrit is None:
        raise ValueError('At least three parameters are needed (X, y_true, fCrit)')
    AllLoss = np.empty(len(Allmodels))
    try:
        in_shape = y_true.shape[0]
    except Exception:
        in_shape = np.isscalar(y_true) * 1
    if in_shape == 1:
        X_array = np.reshape(X, (-1, X.shape[0]))
        for idx, model in enumerate(Allmodels):
            if AllType[idx] == 'Kriging':
                if model is None:
                    AllLoss[idx] = np.inf
                else:
                    AllLoss[idx] = abs(y_true - predictKriging((X_array, model, False)))
            elif AllType[idx] == 'RBF':
                if model is None:
                    AllLoss[idx] = np.inf
                else:
                    AllLoss[idx] = abs(y_true - predictRoyRBFmodel((X_array, model, False)))
            elif AllType[idx] == 'SVM':
                AllLoss[idx] = abs(y_true - predictSVM((X_array, model)))
            elif AllType[idx] == 'RF':
                AllLoss[idx] = abs(y_true - predictRandomForest((X_array, model, False)))
        # sorted_ind = np.argsort(np.argsort(AllLoss))
        sorted_ind = np.argsort(np.argsort(AllLoss))
    elif in_shape > 1:
        p = Parallel(n_jobs=nTotal)
        process = [(copy.deepcopy(X), copy.deepcopy(y_true),
                    Allmodels[modi], AllType[modi], fCrit) for modi in range(len(Allmodels))]
        AllLoss = p(delayed(_parallel_evaluate)(p) for p in process)
        # pool = multiprocessing.Pool(processes=nTotal)
        # AllLoss = pool.map(_parallel_evaluate, process)
        # pool.close()
        # pool.join()
        if fCrit is mean_squared_error or fCrit is mean_absolute_error:
            sorted_ind = np.argsort(np.argsort(AllLoss))
        elif fCrit is r2_score:
            sorted_ind = np.argsort(np.argsort(AllLoss))
            new_index = np.array([len(AllType)-1]*len(AllType)) - sorted_ind
            sorted_ind = new_index
        else:
            raise NotImplementedError
        # sorted_ind = np.argsort(np.argsort(AllLoss))
    else:
        raise Exception("Invalid input in model selection")
    return sorted_ind, AllLoss


def _parallel_evaluate(arguments):
    # x, y, KrgTypes, nKriging, RBFconfigs, RBFKernels, nRBF, SVMconfig, nSVM, nCPU, evall, console_log, mixint, ind=arguments
    x, y_true, model, modelType, fCrit = arguments

    if modelType == 'Kriging':
        # pool = multiprocessing.Pool(processes=nKriging)
        # mixint, x, y, regType, kernel, evall
        param = x, model, False
        result = predictKriging(param)
    elif modelType == 'RBF':
        param = x, model, False
        result = predictRoyRBFmodel(param)
    elif modelType == 'SVM':
        param = x, model
        # X, Y, k, d, c, e, cache_size, max_iter= arguments
        result = predictSVM(param)
    elif modelType == 'RF':
        param = x, model, False
        result = predictRandomForest(param)
    else:
        raise NotImplementedError
    result = np.reshape(result, (y_true.shape[0], -1))
    loss = fCrit(y_true, result)
    return loss