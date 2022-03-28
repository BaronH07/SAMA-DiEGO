import copy
import multiprocessing
import time

from .KrigingSMT import createKriging
from .RBFRoy import replaceKernel, createRoyRBFmodel
from .RandomForest import createRandomForest
from .SVMSklearn import createSVM


# Fit models in parallel
def Parallel_train(arguments):
    # x, y, KrgTypes, nKriging, RBFconfigs, RBFKernels, nRBF, SVMconfig, nSVM, nCPU, evall, console_log, mixint, ind=arguments
    x, y, config, modelType, evall, Gap = arguments

    if modelType == 'Kriging':
        # mixint, x, y, regType, kernel, evall
        if evall % Gap == 0:
            param = config[0], x, y, config[1], config[2], config[3], evall
            model = createKriging(param)
        else:
            model = None
    elif modelType == 'RBF':
        temp = config[0] + [config[1]]  # must be list
        param = x, y, temp, evall
        model = createRoyRBFmodel(param)
    elif modelType == 'SVM':
        param = x, y.flatten(), config[0], config[1], config[2], config[3], config[4], config[5]
        # X, Y, k, d, c, e, cache_size, max_iter= arguments
        model = createSVM(param)
    elif modelType == 'RF':
        param = x, y.flatten()
        model = createRandomForest(param)
    else:
        raise NotImplementedError
    return model

# Single fitting
def train(x, y, KrgTypes, nKriging, RBFconfigs, RBFKernels, nRBF, SVMconfig, nSVM, RFconfig, nRF, nCPU, evall,
          console_log, mixint=None):
    pool = multiprocessing.Pool(processes=nCPU)
    s = time.time()
    if nKriging > 0:
        # pool = multiprocessing.Pool(processes=nKriging)
        # mixint, x, y, regType, kernel
        process = [(copy.deepcopy(mixint), copy.deepcopy(x), copy.deepcopy(y), KrgTypes[modi][0],
                    KrgTypes[modi][1], KrgTypes[modi][2], evall) for modi in range(nKriging)]
        KrgModels = pool.map(createKriging, process)
        # pool.close()
    # print("Time to compute Kriging models  ", time.time() - s, 'seconds')
    # console_log.write('Time to compute Kriging models: ' + str(time.time() - s) + ' seconds\n')
    # s = time.time()
    # Serialization? or Parallel?
    if nRBF > 0:
        process = [
            (copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(replaceKernel(RBFconfigs, RBFKernels, modi)),
             evall) for modi in range(nRBF)]
        RBFModels = pool.map(createRoyRBFmodel, process)
    if nSVM > 0:
        process = [
            (copy.deepcopy(x), copy.deepcopy(y), conf[0], conf[1], conf[2], conf[3], conf[4], conf[5]) for conf in
            SVMconfig]
        # X, Y, k, d, c, e, cache_size, max_iter= arguments
        SVMModels = pool.map(createSVM, process)
    if nRF > 0:
        rfmodel = createRandomForest((copy.deepcopy(x), copy.deepcopy(y)))

    pool.close()
    pool.join()
