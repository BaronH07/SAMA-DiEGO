import csv
import itertools
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from ioh.iohcpp import problem as IOH_PBO_problem
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from smt.applications.mixed_integer import MixedIntegerContext

from .InfillCriteria import MGFI, EI
from .KrigingSMT import LHSSampling, RandomSampling, RandomSampleGenerator
from .MIES_multistart import MIES_multistart
from .ModelSelection import ModelRank
from .ModelTrain import Parallel_train
from .ModelVerification import ModelVerification
from .OptimizeIC_multistart import onePlusLambdaEA_twoRate_multistart
from .RBFRoy import createBBOBRBFkernels
from .SVMSklearn import defaultSVMsettings
from .utils import createMIESIntInputSpace, IOH_int_function_new


## Example of dtype and drange
## SMT styled mixed integer context
# dtype = [INT, FLOAT, (ENUM, 4)]
# drange = [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]


def pipeline(problemCall, drange, dtype, initSize=None, maxEval=None, **kwargs):
    """
    Input arguments
    problemCall: function handle to the objective function (required)
    drange: lower and upper bound of the design space (required)
            [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]
    dtype: data type, a list (required)
            [INT, FLOAT, (ENUM, 4)], the INT, FLOAT, ENUM are from the smt.applications.mixed_integer.MixedIntegerContext

    Optional input arguments:
    initSize: number of initial evaluations, default=100*number of variables
    maxEval: maximum number of evaluations, default=300*number of variables
    runNo: run number, multiple run to stabilize
    KrgTypes: type of Kriging model, ['constant', 'linear', 'quadratic'] × [‘abs_exp’, ‘squar_exp’, ‘act_exp’, ‘matern52’, ‘matern32’, ‘gower’]
    RBFconfigs: general RBF config, [ptail(polynomial tail), squares, smooth, Kernel]
    RBFKernels: alternative kernels for RBF model
    ICtype: infill criteria, ['MGFI', 'EI', ]
    nSingleModel: regulate the maximum number of models from the same category to be selected at the same time in model verification
        This is to say, if nSingleModel=3, then maximally only 3 kriging models will be selected at the same time.
        Just a measure to guarantee the diversity of surrogate models in the main loop
    nTotal: number of models that survive model verification (i.e., maintained in the main loop)
    @return: the best found point (x, y)
    """
    np.seterr(over='raise')
    if problemCall is None or drange is None or dtype is None:
        raise ValueError('At least three parameters are required (problemCall, drange, dtype)')
    nVar = len(drange)
    if maxEval is None:
        # maxEval = nVar * nVar * nVar
        maxEval = 200
    if initSize is None:
        # initSize = round(np.sqrt(nVar)) * nVar  # recommended, but has to be at least larger then n+1
        # initSize = nVar + 1
        initSize = min(round(0.1 * maxEval) + 1, 4 * nVar + 1)
    if maxEval <= initSize:
        raise ValueError('The number of function evaluations shall be greater than initial size.')
    if len(dtype) != len(drange):
        raise ValueError('The amount of element in dtype and drange shall be the same.')
    # Specify parameters for initialization
    initType = kwargs.get('Init', 'LHS')
    lhs_citerion = kwargs.get('lhs_citerion', 'ese')
    #############################################################
    ## Specify surrogates/meta-models
    ## kriging model type: ['constant', 'linear', 'quadratic'] × ['abs_exp', 'squar_exp', 'act_exp', 'matern52', 'matern32', 'gower']
    # KrgSpec = ['Kriging', 'KPLS', 'KPLSK']
    # KrgReg = ['constant', 'linear', 'quadratic']
    # KrgKernel = ['abs_exp', 'squar_exp', 'act_exp', 'matern52', 'matern32', 'gower']
    # # KrgKernel = ['abs_exp', 'squar_exp', 'act_exp']
    # Krg = list(itertools.product(KrgReg, KrgKernel))
    #############################################################
    # Check KrigingSMT.py for more information
    # Krg = createKrgConfig(nVar)
    # Krg = createBBOBKrgConfig()
    KrgTypes = kwargs.get('KrgType', None)
    if KrgTypes is None:
        KrgTypes = []

    # Used to speed up computation
    # KrgGap: the number of iterations to update (between any two updates) Kriging models in main loop
    # Just for saving running time
    KrgGap = kwargs.get('KrgGap', 1)
    if len(KrgTypes) == 0:
        KrgGap = np.inf
    if KrgGap > 1 and KrgGap != np.inf:
        useGap = True
    else:
        useGap = False
    print("Use Gap", useGap)

    ## RBF config ptail(polynomial tail)=True, squares=False, smooth=0.001
    # RBFconfigs = kwargs.get('RBFconfigs', [True, False, 0.001, "CUBIC"])
    RBFconfigs = kwargs.get('RBFconfigs', [True, False, 0, "CUBIC"])
    # All RBF model type: kernels = ["CUBIC", "THINPLATESPLINE", "INVQUADRIC",
    #            "POLYHARMONIC1", "POLYHARMONIC4", "POLYHARMONIC5",
    #            "MULTIQUADRIC", "GAUSSIAN", "INVMULTIQUADRIC"]
    ## Check RBFRoy.py for more information
    rbf = createBBOBRBFkernels()
    RBFKernels = kwargs.get('RBFKernels', rbf)

    ## Configuration for support vector (machine) regression
    ## Check SVMSklearn.py for more information
    svm = defaultSVMsettings(nVar)
    SVMConfig = kwargs.get('SVMConfig', svm)
    # SVMConfig = kwargs.get('SVMConfig', [])

    ## Configurations for random forest
    ## Check RandomForest.py and RandomForestHao.py for more information
    RFConfig = kwargs.get('RFConfig', ['default'])
    # RFConfig = kwargs.get('RFConfig', [])

    # RBFKernels = kwargs.get('RBFKernels', [])

    # Specify type of infill criteria
    #   MGFI: Moment generating function based
    #   EI: Expected Improvement
    #   PV: prediction values
    ICtype = kwargs.get('ICtype', 'MGFI')
    ### Need further check, sometimes it incurred errors/exceptions if using random forest without infill criterion
    if ICtype == 'PV':
        RFConfig = []
        print("Random forest cannot work with non-probabilistic based infill criteria.")
    ## MGFI_tt = kwargs.get('MGFI_tt',1)
    # The size of sliding window for online model selection
    SlidingWLen = kwargs.get('SelectionSize', 1)
    # The number of historical losses overlooked in model selection
    NhistoryLoss = kwargs.get('NumHistoryLoss', 1)
    history_loss_sorted_ind = [[]]
    # Specify online model selection criterion
    criterion = kwargs.get('SelectionCriterion', 'mse')
    if criterion.lower() == 'mse':
        # Mean squared Error
        fCrit = mean_squared_error
    elif criterion.lower() == 'mae':
        # Mean squared Error
        fCrit = mean_absolute_error
    elif criterion.lower() == 'r2' and SlidingWLen > 1:
        # R squared
        fCrit = r2_score
    else:
        raise NotImplementedError("This criterion is not supported!")

    # Type of backend optimizer
    Opttype = kwargs.get('Optimizer', 'mies')
    if Opttype == 'mies':
        from .SearchSpace import OrdinalSpace, NominalSpace

        if isinstance(problemCall, IOH_int_function_new):
            Spctype = kwargs.get('Space', 'O')
            if Spctype == 'O':
                search_space = createMIESIntInputSpace(problemCall)
            else:
                raise NotImplementedError("Currently, only pure integer/ordinal (O) mixed BBOB case is supported")
        elif getattr(problemCall, '__module__', None) == IOH_PBO_problem.__name__:
                from .SearchSpace import OrdinalSpace, NominalSpace
                Spctype = kwargs.get('Space', 'O')
                if Spctype == 'N':
                    search_space = NominalSpace([0, 1]) * nVar
                elif Spctype == 'O':
                    search_space = OrdinalSpace([0, 1], 'I') * nVar
                else:
                    raise ValueError("IOH PBO problems can only be illustrated with categorical (N) or ordinal (O) search space.")
        else:
            raise NotImplementedError("Currently, only pure BBOB and IOH PBO problems are supported")

    # Minization or Maximization
    minimization = kwargs.get('minimization', True)
    bestX = np.empty((1, nVar))
    if minimization:
        bestY = np.inf
    else:
        bestY = -np.inf

    # Early stopping criteria
    earlyEpsilon = 1e-12
    earlyStopGap = maxEval

    # Global optimum
    bestHit = kwargs.get('GlobalOptimum', np.inf)
    if isinstance(problemCall, IOH_int_function_new):
        bestHit = problemCall.get_target_y()
    if bestHit == np.inf:
        if minimization:
            bestHit = -np.inf
        else:
            bestHit = np.inf
    bestHitX = kwargs.get('GlobalOptimumInput', None)
    if isinstance(problemCall, IOH_int_function_new):
        bestHitX = problemCall.get_target_x()
    # Place to store results
    filePath = kwargs.get('filePath', './Results')
    runNo = kwargs.get('runNo', 1)

    outPath = os.path.join(filePath, str(runNo))
    Path(outPath).mkdir(parents=True, exist_ok=True)
    outputFile = os.path.join(outPath, 'record.csv')

    outconfig = os.path.join(outPath, 'Config.csv')
    outlog = os.path.join(outPath, 'log.txt')
    optimizer_log = kwargs.get('optimizer_log', None)
    if optimizer_log is not None:
        out_optimizer_log = os.path.join(outPath, optimizer_log)
    else:
        out_optimizer_log = None
    # Declare data
    X = np.empty((maxEval, nVar))
    Y = np.empty((maxEval, 1))
    new_x = np.empty((SlidingWLen, nVar))
    new_y = np.empty((SlidingWLen, 1))

    # Initialization
    mixint = MixedIntegerContext(dtype, drange)
    if initType == 'LHS':
        init_x, init_y, initSize = LHSSampling(problemCall, mixint, initSize, lhs_citerion)
    else:
        init_x, init_y, initSize = RandomSampling(problemCall, mixint, initSize)
    X[:initSize, :] = init_x
    Y[:initSize, :] = init_y
    X[initSize:, :] = np.NAN
    Y[initSize:, :] = np.NaN

    # Random sample generator in case of duplicated samples from back-end optimizer
    ## This is to say, when the data sample found by backend optimizer exist in obtained samples,
    ## a new random sample will be given
    randSampleGenerator = RandomSampleGenerator(mixint)

    nKriging = len(KrgTypes)
    nRBF = len(RBFKernels)
    if nRBF < 1 and nKriging < 1:
        raise ValueError('At least one meta-model should be specified')
    # nModels = len(KrgTypes) + len(RBFKernels)

    start = time.time()
    evall = initSize
    nEarlyStopGap = 0
    prevBest = np.inf if minimization else -np.inf

    # Do normalization
    normalization = kwargs.get('normalized', True)
    if normalization:
        std = np.std(init_y, axis=0)
        mean = np.mean(init_y, axis=0)
    else:
        std = 1
        mean = 0
    outPd = pd.DataFrame(columns=['Num func evals', 'X', 'Y', 'Best_so_far_Y', 'Is Alg Found?',
                                  'Type of model', 'Index', 'Description', 'Is Model Found?', 'Global Optimum'])
    new_pd = pd.DataFrame({'Num func evals': list(range(1, evall + 1, 1)), 'X': list(init_x), 'Y': list(init_y),
                           'Is Alg Found?': [False] * evall, 'Global Optimum': [bestHit] * evall})
    outPd = outPd.append(new_pd)
    outPd.to_csv(outputFile, mode='w', header=True, index=False)

    init_y = (init_y - mean) / std
    # Path to save logs
    console_log = open(outlog, 'w+')
    # console_log.write("The LOG of " + functionName + " " + str(runNo) + "\n")
    console_log.write("The LOG of " + str(runNo) + "\n")
    if abs(bestHit) != np.inf:
        print('Global optimum is ', bestHit)
        console_log.write("Global optimum is " + str(bestHit) + "\n")
    else:
        print('Global optimum is not known')
        console_log.write("Global optimum is not known\n")
    if bestHitX is not None:
        print('Global optimum solution is', bestHitX)
        console_log.write("Global optimum solution is " + str(list(bestHitX)) + "\n")

    print("Start verifying models...")
    console_log.write("Start verifying models...\n")
    nCPU = kwargs.get('nSingleModel', None)
    nCPU = multiprocessing.cpu_count() - 2 if nCPU is None else nCPU
    nCPU = nCPU if nCPU > 0 else multiprocessing.cpu_count()
    nTotal = kwargs.get('nTotal', nCPU)
    nTotal = multiprocessing.cpu_count() - 2 if nTotal is None else nTotal
    nTotal = nTotal if nTotal > 0 else multiprocessing.cpu_count()

    # Multistart optimization:
    nMultistart = kwargs.get('Multistart', nTotal)
    nMultistart = nMultistart if nMultistart < nTotal else nTotal
    print("Max number of models", nTotal)

    # Do model verification, see ModelVerification.py
    KrgTypes, KrgModels, nKriging, RBFKernels, RBFModels, nRBF, SVMConfig, SVMModels, nSVM, RFconfig, RFModel, nRF = \
        ModelVerification(mixint, init_x, init_y, KrgTypes, RBFconfigs, RBFKernels, SVMConfig, RFConfig,
                          fCrit, 0, nCPU=nCPU, nTotal=nTotal)

    print("Time to verfiy models ", time.time() - start)

    console_log.write('Time to verfiy models ' + str(time.time() - start) + ' seconds\n')

    # Number of CPUs (processes) of your computer
    # nCPU = multiprocessing.cpu_count() - 2
    # nCPU = nCPU if nCPU > 0 else multiprocessing.cpu_count()
    config_log = open(outconfig, "w+")
    w = csv.writer(config_log)
    dargs = {'drange': drange, 'dtype': dtype, 'initType': initType,
             'initSize': initSize, 'maxEval': maxEval,
             'krgTypes': KrgTypes, 'RBFconfigs': RBFconfigs, 'RBFKernels': RBFKernels,
             'ICtype': ICtype, 'Optimizer': Opttype, 'SVMconfigs': SVMConfig, 'RFconfig': RFconfig,
             'ModelSelectionCriterion': criterion, 'Normalization': normalization, 'nCPU': nCPU, 'nTotal': nTotal,
             'nMultistart': nMultistart, 'KrgGap': KrgGap, 'UseKrgGap': useGap}
    for key, val in dargs.items():
        w.writerow([key, val])
    config_log.close()
    console_log.close()

    # Mapping input RBF kernel choices to full configurations
    RBFconfigs = list(itertools.product([RBFconfigs[:3]], RBFKernels))
    for idx, k in enumerate(KrgTypes):
        new_k = [mixint] + list(k)
        KrgTypes[idx] = new_k

    # Concatenating all configurations
    Allconfig = KrgTypes + RBFconfigs + SVMConfig + RFconfig
    AllType = ['Kriging'] * nKriging + ['RBF'] * nRBF + ['SVM'] * nSVM + ['RF'] * nRF
    assert len(Allconfig) == len(AllType)
    Allmodels = KrgModels + RBFModels + SVMModels + RFModel
    assert len(Allmodels) == len(AllType)

    # Occupy/Declare computational resources using Joblib
    par = Parallel(n_jobs=nTotal)
    while evall <= maxEval:
        console_log = open(outlog, 'a+')
        iterationTime = time.time()
        print('# ', (evall - initSize + 1), 'iteration...')
        console_log.write('#' + str((evall - initSize + 1)) + 'iteration...' + '\n')
        # Evaluate different models and find the best with respect to new sample(s)
        # modelType: 'RBF' or 'Kriging'
        #############################################################
        # Select the best model
        if evall == initSize:
            sortedModelInd, modelLoss = ModelRank(X[:evall, :], init_y, fCrit, Allmodels, AllType, nTotal)
            sorted_best_ind = np.argsort(modelLoss)

            # Find best-so-far
            if minimization:
                bestInd = np.argmin(Y[:evall].flatten())
            else:
                bestInd = np.argmax(Y[:evall].flatten())
            bestX = X[bestInd, :]
            bestY = Y[bestInd]
        else:
            if (evall - 1) % KrgGap == 0 and useGap:
                sortedModelInd, modelLoss = ModelRank(new_x, new_normalized_y, fCrit, Allmodels[:nKriging],
                                                      AllType[:nKriging], nTotal)
            else:
                sortedModelInd, modelLoss = ModelRank(new_x, new_normalized_y, fCrit, Allmodels, AllType, nTotal)
            # sortedModelInd, modelLoss = ModelRank(new_x, new_normalized_y, fCrit, Allmodels, AllType, nTotal)
            if len(history_loss_sorted_ind) >= NhistoryLoss or len(history_loss_sorted_ind[0]) == 0:
                history_loss_sorted_ind.pop(0)
            history_loss_sorted_ind.append(sortedModelInd)
            sum_loss_ind = np.sum(history_loss_sorted_ind, axis=0)

            # best_ind = np.argmin(sum_loss_ind)
            sorted_best_ind = np.argsort(sum_loss_ind)

            # Update best-so-far
            if bestY - bestHit == 0:
                print('Hit the global optimum!')
                console_log.write("Hit the global optimum!--Code\n")
                break
            elif hasattr(problemCall, 'state'):
                if problemCall.state.optimum_found:
                    print('Hit the global optimum!')
                    console_log.write("Hit the global optimum!--IOH\n")
                    break
            elif hasattr(problemCall, 'f'):
                if problemCall.f.state.optimum_found:
                    print('Hit the global optimum!')
                    console_log.write("Hit the global optimum!--IOH\n")
                    break

            nEarlyStopGap = nEarlyStopGap + 1
            if nEarlyStopGap == earlyStopGap:
                epsilon = abs(bestY - prevBest) / bestY
                if epsilon < earlyEpsilon:
                    print('Early stopping criteria has been met!')
                    console_log.write("Early stopping criteria has been met!\n")
                    break
                nEarlyStopGap = 0
                prevBest = bestY
        # print("Time to select the best model  ", time.time() - iterationTime)
        console_log.write("Time to select the best model  " + str(time.time() - iterationTime) + '\n')
        t_sel = time.time()
        #############################################################
        # Find the next promising point
        # Infill criterion + backend optimization
        for ind in sorted_best_ind:
            best_ind = ind
            errorFlag = False
            try:
                bestModel = Allmodels[best_ind]
                modelType = AllType[best_ind]
                model_name = Allconfig[best_ind]
                bestModelInd = best_ind
                console_log.write(
                    'Current best surrogate is a ' + str(modelType) + '. Parameter ' + str(model_name) + '\n')

                # Infill Criteria/ Acquisition function
                if ICtype == 'MGFI':
                    tt = np.exp(0.5 * np.random.randn())
                    normalized_bestY = (bestY - mean) / std
                    acquisition_func = MGFI(bestModel, modelType, normalized_bestY, minimize=minimization, t=tt)
                    acquisition_func = partial(acquisition_func, dx=False)
                    acquisition_type = 'MGFI'
                elif ICtype == 'EI':
                    normalized_bestY = (bestY - mean) / std
                    acquisition_func = EI(bestModel, modelType, normalized_bestY, minimize=minimization)
                    acquisition_func = partial(acquisition_func, dx=False)
                    acquisition_type = 'EI'
                else:
                    acquisition_func = bestModel
                    acquisition_type = modelType

                # Optimize IC
                if Opttype == 'mies':
                    if nMultistart > 1:
                        hist_c = int(nMultistart / 3)
                        last_c = 1
                        last_x = X[evall - last_c:evall, :]
                        temp_ind = np.argsort(Y[:evall].ravel(), axis=0)
                        hist_x = X[temp_ind[:hist_c], :] if minimization else X[temp_ind[-hist_c:], :]
                        hist_x = np.vstack((hist_x, last_x))

                        new_xs = MIES_multistart(acquisition_func, acquisition_type, nVar, search_space, hist_x=hist_x,
                                                 nRandStart=nMultistart - (hist_c + last_c),
                                                 minimization=minimization, nTotal=nTotal)
                    else:
                        raise NotImplementedError
                elif Opttype == 'twoRateEA':
                    if nMultistart > 1:
                        hist_c = int(nMultistart / 3)
                        last_c = 1
                        last_x = X[evall-last_c:evall, :]
                        temp_ind = np.argsort(Y[:evall].ravel(), axis=0)
                        hist_x = X[temp_ind[:hist_c], :] if minimization else X[temp_ind[-hist_c:], :]
                        hist_x = np.vstack((hist_x, last_x))
                        new_xs = onePlusLambdaEA_twoRate_multistart(acquisition_func, acquisition_type, nVar, nPop=10,
                                                                    r=2, hist_x=hist_x, nRandStart=nMultistart - (hist_c+last_c),
                                                                    minimization=minimization,
                                                                    nTotal=nTotal)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            except FloatingPointError:
                errorFlag = True
                print("Overflow encountered in optimizing IC, shift to next model.")
                console_log.write("Overflow encountered in optimizing IC, shifting to the next model.\n")
                pass
            if not errorFlag:
                break
            elif ind == sorted_best_ind[-1]:
                console_log.write("All meta-models incur overflow on the problem.")
                raise RuntimeError("All meta-models incur overflow on the problem.")

        x = X[:evall, :]
        #############################################################
        # Check the new x to avoid duplication of samples
        # Evaluate the new x
        nDuplicate = 0
        Flag = False
        isFoundNew = True
        for xxx in new_xs:
            if not (x == xxx).all(1).any():
                Flag = True
                new_x = xxx
                break
        if not Flag:
            isFoundNew = False
            new_x = randSampleGenerator(1)[0]
            new_x = new_x.astype(int, copy=False)
            print("Find duplicates.")
            console_log.write("Find duplicates.\n")
            while (x == new_x).all(1).all():
                new_x = randSampleGenerator(1)[0]
                new_x = new_x.astype(int, copy=False)
                nDuplicate = nDuplicate + 1
                if nDuplicate > nVar:
                    print('NO NEW SAMPLES are found. DUPLICATE samples.')
                    console_log.write("NO NEW SAMPLES are found. DUPLICATE samples.\n")
                    break
        print("Time to optimize infill criteria  ", time.time() - t_sel)
        console_log.write('Time to optimize infill criteria: ' + str(time.time() - t_sel) + ' seconds\n')
        # model_log.writerow([])
        new_y = problemCall(new_x)
        X[evall, :] = new_x
        Y[evall, :] = new_y

        # Normalize the objective values to make sure that RBFs work properly
        y = Y[:evall]
        #############################################################
        # Update the mean and standard deviation of data samples
        if normalization:
            std = np.std(y, axis=0)
            mean = np.mean(y, axis=0)
        else:
            std = 1
            mean = 0
        # # Normalize the training samples
        y = (y - mean) / std
        new_normalized_y = (new_y - mean) / std
        #############################################################
        # Train meta models
        if evall != initSize:
            s = time.time()
            process = [(x, y,
                        Allconfig[modi], AllType[modi], evall, KrgGap) for modi in range(len(AllType))]
            Allmodels = par(delayed(Parallel_train)(p) for p in process)
            print("Time to train meta-models  ", time.time() - s, 'seconds')
            console_log.write('Time to train meta-models: ' + str(time.time() - s) + ' seconds\n')
        else:
            print("Skip the training stage in first iteration...")
            console_log.write("Skip the training stage in first iteration...\n")
        evall = evall + 1
        if minimization:
            if new_y < bestY:
                bestX, bestY = new_x, new_y
        else:
            if new_y > bestY:
                bestX, bestY = new_x, new_y
        #############################################################
        # Do some summary here, write logs
        print('This iteration', new_y, 'Current best', bestY, 'global optimum', bestHit)
        console_log.write('This iteration: ' + str(new_y) + ' ,best so far: ' + str(bestY) + ' ,global optimum: ' + str(
            bestHit) + '\n')
        # print('Iteration time', (time.time() - iterationTime))
        console_log.write('Iteration time: ' + str(time.time() - iterationTime) + ' seconds\n')
        console_log.write("\n")
        console_log.close()

        new_pd = pd.DataFrame(
            {'Num func evals': evall, 'X': [new_x], 'Y': new_y, 'Best_so_far_Y': bestY, 'Is Alg Found?': True,
             'Type of model': modelType, 'Index': str(bestModelInd), 'Parameter': str(model_name),
             'Is Model Found?': isFoundNew, 'Global Optimum': bestHit})
        new_pd.to_csv(outputFile, mode='a', header=False, index=False)
        #############################################################
    # end while

    end = time.time()
    print("Run number", runNo, "is completed!")
    print('Total running time: ', end - start)
    return bestX, bestY
