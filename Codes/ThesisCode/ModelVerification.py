import copy
import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from .KrigingSMT import predictKriging, verifyKriging
from .RBFRoy import replaceKernel, predictRoyRBFmodel, verifyRoyRBFmodel
from .RandomForest import createRandomForest, predictRandomForest
from .SVMSklearn import predictSVM, verifySVM


# Do model verification
def ModelVerification(mixint, x, y, KrgTypes, RBFconfigs, RBFKernels, SVMconfig, RFconfig,
                      fCrit, evall, time_limit=35, nCPU=None,
                      nTotal=None):
    """
    Input arguments
    mixint: mixed-integer context of SMT python package (required)
    x: the initial samples (required)
    y: the real objective values of initial samples x (required)
    ### At least one of the following models shall be specified ###
    KrgTypes: type of Kriging model, example: itertool ['constant', 'linear', 'quadratic'] × [‘abs_exp’, ‘squar_exp’, ‘act_exp’, ‘matern52’, ‘matern32’, ‘gower’]
    RBFconfigs: general RBF config, [ptail(polynomial tail), squares, smooth, Kernel]
    RBFKernels: alternative kernels for RBF model
    SVMcondig: type of SVM regression model, see defaultSVMsettings(nVar) in SVMSklearn.py (required)
    RFconfig: config for random forest (optional)

    fCrit: metrics to select the best model, mean squared error or mean absolute error or R squared
        Shall be a sklearn API
    evall: number of samples

    Optional arguments
    time_limit: the hard limit for running time
    nCPU: number of available (CPU) processes of the computer
    nTotal: number of models that survive the verification, namely, top-T
        Shall be less than nCPU or 2*nCPU depending on the hardware specifications

    @return: the configurations of survived models and the models themself
    KrgNewTypes, KrgNewModels, len(KrgNewTypes), RBFNewKernels, RBFNewModels, len(
        RBFNewKernels), SVMConfig, SVMModels, len(SVMConfig), RFNewConfig, RFModel, len(RFNewConfig)
    """

    # If we have less than 100 samples, do not split
    # Only calculate training errors
    if x.shape[0] < 100:
        X_train = x
        X_test = x
        y_train = y
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Initialize settings
    KrgModels = []
    RBFModels = []
    SVMNewModels = []
    SVMNewConfig = []
    RFNewConfig = []
    if nCPU is None:
        nCPU = multiprocessing.cpu_count() - 2
    if nTotal is None:
        nTotal = nCPU * 2
    nKriging = len(KrgTypes)
    nRBF = len(RBFKernels)
    nSVM = len(SVMconfig)
    nRF = len(RFconfig)
    # pool = multiprocessing.Pool(processes=nCPU)
    par = Parallel(n_jobs=nTotal)
    # print("Start verifying models...")
    newKrgLoss = []
    newRBFLoss = []
    SVMNewLoss = []
    RFLoss = []
    if nKriging > 0:
        # Create all Kriging models
        KrgModels = [[] for i in range(nKriging)]
        process = [(copy.deepcopy(mixint), copy.deepcopy(X_train), copy.deepcopy(y_train), KrgTypes[modi][0],
                    KrgTypes[modi][1], KrgTypes[modi][2], evall) for modi in range(nKriging)]
        KrgModels = par(delayed(verifyKriging)(p) for p in process)
        # Locate 'survived' kriging models
        KrgNonInd = [i for i, val in enumerate(KrgModels) if val is not None]
        nKriging = len(KrgNonInd)
        newKrgModels = [[] for i in range(nKriging)]
        newKrgTypes = [[] for i in range(nKriging)]

        for inew, iold in enumerate(KrgNonInd):
            if KrgModels[iold][1] <= time_limit:
                newKrgModels[inew] = KrgModels[iold][0]
                newKrgTypes[inew] = KrgTypes[iold]

        # Rank Kriging models and select top nCPU
        # nBatch = math.ceil(nKriging / nCPU)
        KrgLoss = np.empty(nKriging)
        krgResults = [[] for i in range(nKriging)]

        process = [(copy.deepcopy(X_test), copy.deepcopy(newKrgModels[model]), False) for model in range(nKriging)]
        # krgResults = pool.map(predictSMTKrigingModel, process)
        krgResults = par(delayed(predictKriging)(p) for p in process)
        # pool.close()
        # evaluate Kriging
        for ind, result in enumerate(krgResults):
            result = np.reshape(result, (y_test.shape[0], -1))
            KrgLoss[ind] = fCrit(y_test, result)
        if fCrit is mean_squared_error or fCrit is mean_absolute_error:
            index = np.argsort(KrgLoss)
        elif fCrit is r2_score:
            index = np.argsort(-KrgLoss)
        else:
            raise NotImplementedError
        # index = index.astype(int)
        nSize = nCPU if nKriging > nCPU else nKriging
        KrgTypes = [[] for i in range(nSize)]
        KrgModels = [[] for i in range(nSize)]
        newKrgLoss = [np.inf] * nSize
        for i in range(nSize):
            newKrgLoss[i] = KrgLoss[index[i]]
            KrgTypes[i] = newKrgTypes[index[i]]
            KrgModels[i] = newKrgModels[index[i]]
        print("Krigings have been verified")

    # Create all RBF models
    if nRBF > 0:
        # Create all Kriging models
        RBFModels = [[] for i in range(nRBF)]
        process = [
            (copy.deepcopy(X_train), copy.deepcopy(y_train), copy.deepcopy(replaceKernel(RBFconfigs, RBFKernels, modi)),
             evall) for modi in range(nRBF)]
        # RBFModels = pool.map(createRoyRBFmodel, process)
        RBFModels = par(delayed(verifyRoyRBFmodel)(p) for p in process)
        # pool.close()

        RBFNonInd = [i for i, val in enumerate(RBFModels) if val is not None]
        nRBF = len(RBFNonInd)
        newRBFModels = [[] for i in range(nRBF)]
        newRBFKernels = [[] for i in range(nRBF)]

        for inew, iold in enumerate(RBFNonInd):
            if RBFModels[iold][1] <= time_limit:
                newRBFModels[inew] = RBFModels[iold][0]
                newRBFKernels[inew] = RBFKernels[iold]

        # nBatch = math.ceil(nRBF / nCPU)
        # Rank Kriging models and select top nCPU
        RBFLoss = np.empty(nRBF)
        RBFResults = [[] for i in range(nRBF)]
        process = [(copy.deepcopy(X_test), copy.deepcopy(newRBFModels[int(model)]), False) for model in range(nRBF)]
        # RBFResults = pool.map(predictRoyRBFmodel, process)
        RBFResults = par(delayed(predictRoyRBFmodel)(p) for p in process)

        # evaluate RBF
        for ind, result in enumerate(RBFResults):
            result = np.reshape(result, (y_test.shape[0], -1))
            RBFLoss[ind] = fCrit(y_test, result)

        if fCrit is mean_squared_error or fCrit is mean_absolute_error:
            index = np.argsort(RBFLoss)
        elif fCrit is r2_score:
            index = np.argsort(-RBFLoss)
        else:
            raise NotImplementedError
        nSize = nCPU if nRBF > nCPU else nRBF
        RBFKernels = [[] for i in range(nSize)]
        RBFModels = [[] for i in range(nSize)]
        newRBFLoss = [np.inf] * nSize
        for i in range(nSize):
            newRBFLoss[i] = RBFLoss[index[i]]
            RBFKernels[i] = newRBFKernels[index[i]]
            RBFModels[i] = newRBFModels[index[i]]
        print("RBFs have been verified")

    if nSVM > 0:
        process = [
            (copy.deepcopy(X_train), copy.deepcopy(y_train.flatten()), conf[0], conf[1], conf[2], conf[3], conf[4],
             conf[5]) for conf in
            SVMconfig]
        # each process loads: X, Y, k, d, c, e, cache_size, max_iter= arguments
        SVMModels = par(delayed(verifySVM)(p) for p in process)
        process = [
            (copy.deepcopy(X_test), copy.deepcopy(model[0])) for model in SVMModels if model[1] <= time_limit]
        SVMResults = par(delayed(predictSVM)(p) for p in process)
        SVMLoss = np.empty(nSVM)
        for ind, result in enumerate(SVMResults):
            result = np.reshape(result, (y_test.shape[0], -1))
            SVMLoss[ind] = fCrit(y_test, result)
        if fCrit is mean_squared_error or fCrit is mean_absolute_error:
            index = np.argsort(SVMLoss)
        elif fCrit is r2_score:
            index = np.argsort(-SVMLoss)
        else:
            raise NotImplementedError
        nSize = nCPU if nSVM > nCPU else nSVM
        SVMNewConfig = [[] for i in range(nSize)]
        SVMNewModels = [[] for i in range(nSize)]
        SVMNewLoss = [np.inf] * nSize
        for i in range(nSize):
            SVMNewLoss[i] = SVMLoss[index[i]]
            SVMNewConfig[i] = SVMconfig[index[i]]
            SVMNewModels[i] = SVMModels[index[i]][0]
        print("SVMs have been verified")

    if nRF > 0:
        ## We only have one RF config by default,
        ## so setting time limit for random forest is not in our consideration
        rfmodel = createRandomForest((copy.deepcopy(X_train), copy.deepcopy(y_train.flatten())))
        rfresult = predictRandomForest((copy.deepcopy(X_test), rfmodel, False))
        # rfresult = np.reshape(rfresult, (y.shape[0], -1))
        RFLoss = fCrit(y_test, rfresult)
        RFconfig = ['default']
        print("RF has been verified")

    AllLoss = newKrgLoss + newRBFLoss + SVMNewLoss + [RFLoss]
    Allconfig = KrgTypes + RBFKernels + SVMNewConfig + RFconfig
    nKriging = len(KrgModels)
    nRBF = len(RBFModels)
    nSVM = len(SVMNewModels)
    if fCrit is mean_squared_error or fCrit is mean_absolute_error:
        index = np.argsort(AllLoss)
        index = list(index)
    elif fCrit is r2_score:
        index = np.argsort(AllLoss)
        new_index = np.flip(index)
        index = list(new_index)
    else:
        raise NotImplementedError
    nSize = nTotal if len(index) > nTotal else len(index)
    idKrg = 0
    idRBF = 0
    idSVM = 0
    RBFNewKernels = []
    RBFNewModels = []
    KrgNewTypes = []
    KrgNewModels = []
    SVMModels = []
    SVMConfig = []
    RFModel = []
    RFconfig = []

    for i in range(nSize):
        if index[i] < nKriging and nKriging > 0:
            KrgNewTypes.append(KrgTypes[index[i]])
            KrgNewModels.append(KrgModels[index[i]])
            idKrg += 1
        elif nKriging <= index[i] < nKriging + nRBF and nRBF > 0:
            RBFNewKernels.append(RBFKernels[index[i] - nKriging])
            RBFNewModels.append(RBFModels[index[i] - nKriging])
            idRBF += 1
        elif nKriging + nRBF <= index[i] < nKriging + nRBF + nSVM and nSVM > 0:
            SVMModels.append(SVMNewModels[index[i] - (nKriging + nRBF)])
            SVMConfig.append(SVMNewConfig[index[i] - (nKriging + nRBF)])
            idSVM += 1
        elif index[i] == nKriging + nRBF + nSVM and nRF > 0:
            RFModel = [rfmodel]
            RFNewConfig = ['default']
            nRF = 1

    print('Model verification is done!')
    return KrgNewTypes, KrgNewModels, len(KrgNewTypes), RBFNewKernels, RBFNewModels, len(
        RBFNewKernels), SVMConfig, SVMModels, len(SVMConfig), RFNewConfig, RFModel, len(RFNewConfig)
