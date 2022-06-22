import itertools
import timeit

import numpy as np
from smt.applications.mixed_integer import MixedIntegerContext
from smt.sampling_methods import LHS, Random
from smt.surrogate_models import KRG, KPLS, KPLSK

from .utils import batchEvaulateProblem


# Do Latin Hypercube Sampling
def LHSSampling(problemCall, mixint, initSize, lhs_citerion):
    # problemCall, mixint, initSize, lhs_citerion = arguments
    if not isinstance(mixint, MixedIntegerContext):
        # mixint = MixedIntegerContext(dtype, drange)
        raise TypeError('The sample space shall be represented by a mixIntegerContext object.')
    # LHS sampling criterion: [‘center’, ‘maximin’, ‘centermaximin’, ‘correlation’, ‘c’, ‘m’, ‘cm’,
    # ‘corr’, ‘ese’] criterion used to construct the LHS design c, m, cm and corr are abbreviation of center,
    # maximin, centermaximin and correlation, respectively

    # if initSize < mixint.get_unfolded_dimension() * 50:
    #     initSize = mixint.get_unfolded_dimension() * 50
    # ese: Enhanced Stochastic Evolutionary algorithm
    if lhs_citerion is None:
        lhs_citerion = "ese"

    lhs = mixint.build_sampling_method(LHS, criterion=lhs_citerion, )
    print("DOE sampling size = {}".format(initSize))
    xt = lhs(initSize)
    xt = xt.astype(int, copy=False)
    yt = batchEvaulateProblem(problemCall, xt)
    # yt = problemCall(xt)
    return xt, yt, initSize


# Do uniform random sampling
def RandomSampling(problemCall, mixint, initSize):
    # problemCall, mixint, initSize, lhs_citerion = arguments
    if ~isinstance(mixint, MixedIntegerContext):
        # mixint = MixedIntegerContext(dtype, drange)
        raise TypeError('The sample space shall be represented by a mixIntegerContext object.')
    # LHS sampling criterion: [‘center’, ‘maximin’, ‘centermaximin’, ‘correlation’, ‘c’, ‘m’, ‘cm’,
    # ‘corr’, ‘ese’] criterion used to construct the LHS design c, m, cm and corr are abbreviation of center,
    # maximin, centermaximin and correlation, respectively

    ran = mixint.build_sampling_method(Random)
    print("DOE sampling size = {}".format(initSize))
    xt = ran(initSize)
    xt = xt.astype(int, copy=False)
    yt = problemCall(xt)
    return xt, yt


def RandomSampleGenerator(mixint):
    ran = mixint.build_sampling_method(Random)
    return ran


# @DEPRECATED
def createKrgConfig(nVar):
    if nVar <= 30:
        KrgSpec = ['Kriging']
        KrgReg = ['constant', 'linear', 'quadratic']
        # KrgReg = ['constant', 'linear']
        # KrgKernel = ['abs_exp', 'squar_exp', 'act_exp', 'matern52', 'matern32', 'gower']
        KrgKernel = ['abs_exp', 'squar_exp', 'matern52', 'matern32', 'gower']
        Krg = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
    else:
        KrgSpec = ['KPLS']
        KrgReg = ['constant', 'linear', 'quadratic']
        # KrgReg = ['constant', 'linear']
        KrgKernel = ['abs_exp', 'squar_exp']
        tmp1 = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
        KrgSpec = ['KPLSK']
        KrgReg = ['constant', 'linear', 'quadratic']
        # KrgReg = ['constant', 'linear']
        KrgKernel = ['squar_exp']
        tmp2 = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
        Krg = tmp1 + tmp2
    return Krg


# Create all available Kriging models, 24 in total (15+6+3)
def createAllKrgConfig():
    KrgSpec = ['Kriging']
    KrgReg = ['constant', 'linear', 'quadratic']
    # KrgReg = ['constant', 'linear']
    # KrgKernel = ['abs_exp', 'squar_exp', 'act_exp', 'matern52', 'matern32', 'gower']
    KrgKernel = ['abs_exp', 'squar_exp', 'matern52', 'matern32', 'gower']
    tmp0 = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
    KrgSpec = ['KPLS']
    KrgReg = ['constant', 'linear', 'quadratic']
    # KrgReg = ['constant', 'linear']
    KrgKernel = ['abs_exp', 'squar_exp']
    tmp1 = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
    KrgSpec = ['KPLSK']
    KrgReg = ['constant', 'linear', 'quadratic']
    # KrgReg = ['constant', 'linear']
    KrgKernel = ['squar_exp']
    tmp2 = list(itertools.product(KrgSpec, KrgReg, KrgKernel))
    Krg = tmp0 + tmp1 + tmp2
    return Krg


############################################################################################
# Recommended (personally) Kriging Models for BBOB and IOH PBO problems (w.r.t execution time)
# Just Qi's suggestion LOL
def createBBOBKrgConfig():
    config = [('KPLSK', 'constant', 'squar_exp'),
              ('KPLSK', 'linear', 'squar_exp'),
              ('KPLS', 'linear', 'abs_exp'),
              ('KPLS', 'quadratic', 'squar_exp'),
              ('KPLS', 'quadratic', 'abs_exp'),
              ('KPLS', 'constant', 'abs_exp'),
              ('KPLS', 'linear', 'squar_exp'),
              ('KPLS', 'constant', 'squar_exp'),
              ('Kriging', 'constant', 'squar_exp'),
              ('Kriging', 'linear', 'squar_exp'),
              ('Kriging', 'constant', 'abs_exp'),
              ('Kriging', 'constant', 'gower'),
              ('Kriging', 'linear', 'abs_exp'),
              ('Kriging', 'linear', 'gower')]
    return config


# These are the ones recorded in slides/thesis
def createIOHPBOKrgConfig():
    KrgReg = ['constant', 'linear', 'quadratic']
    KrgKernel = ['abs_exp', 'squar_exp', 'matern52', 'matern32', 'gower']
    Base = ['Kriging']
    config = list(itertools.product(Base, KrgReg, KrgKernel))
    return config
############################################################################################


###################################################################
# Fit Kriging models
def createSMTKrigingModel(arguments):
    mixint, x, y, regType, kernel, evall = arguments
    # regType : [‘constant’, ‘linear’, ‘quadratic’]
    # kernel : [‘abs_exp’, ‘squar_exp’, ‘act_exp’, ‘matern52’, ‘matern32’, ‘gower’]
    time = np.inf
    if regType is None:
        regType = 'linear'
    if kernel is None:
        kernel = 'squar_exp'
    try:
        model = mixint.build_surrogate_model(KRG(poly=regType, corr=kernel, print_global=False, n_start=5))
        model.set_training_values(x, y)
        model.train()
    except Exception:
        return None
    # if __debug__:
    # print("Kriging", regType, kernel, "is done!")
    # Example of making prediction
    # rand = mixint.build_sampling_method(Random)
    # xt = rand(50)
    # yt = probelmCall(xt)
    # yt = model.predict_values(xt)
    return model


def createSMTKPLSModel(arguments):
    mixint, x, y, regType, kernel, evall = arguments
    # regType : [‘constant’, ‘linear’, ‘quadratic’]
    # kernel : [‘abs_exp’, ‘squar_exp’]
    if regType is None:
        regType = 'constant'
    if kernel is None:
        kernel = 'squar_exp'
    try:
        model = mixint.build_surrogate_model(KPLS(poly=regType, corr=kernel, n_comp=2, print_global=False, n_start=5))
        model.set_training_values(x, y)
        model.train()
    except Exception:
        return None
    # print("KPLS", regType, kernel, "is done!")
    return model
    # Example of making prediction
    # rand = mixint.build_sampling_method(Random)
    # xt = rand(50)
    # yt = probelmCall(xt)
    # yt = model.predict_values(xt)


def createSMTKPLSKModel(arguments):
    mixint, x, y, regType, kernel, evall = arguments
    # regType : [‘constant’, ‘linear’, ‘quadratic’]
    # kernel : [‘squar_exp’]
    if regType is None:
        regType = 'constant'
    if kernel is None:
        kernel = 'squar_exp'
    try:
        model = mixint.build_surrogate_model(KPLSK(poly=regType, corr=kernel, n_comp=2, print_global=False, n_start=5))
        model.set_training_values(x, y)
        model.train()
    except Exception:
        return None
    # print("KPLSK", regType, kernel, "is done!")
    return model
    # Example of making prediction using this model
    # rand = mixint.build_sampling_method(Random)
    # xt = rand(50)
    # yt = probelmCall(xt)
    # yt = model.predict_values(xt)


# Fitting Kriging models
def createKriging(arguments):
    mixint, x, y, spec, regType, kernel, evall = arguments
    if spec == 'Kriging':
        return createSMTKrigingModel((mixint, x, y, regType, kernel, evall))
    elif spec == 'KPLS':
        return createSMTKPLSModel((mixint, x, y, regType, kernel, evall))
    elif spec == 'KPLSK':
        return createSMTKPLSKModel((mixint, x, y, regType, kernel, evall))
    else:
        raise NotImplementedError


###################################################################

###################################################################
# Verifying Kriging models (only for model verification)
# The same as createKriging but with timer
def verifyKriging(arguments):
    mixint, x, y, spec, regType, kernel, evall = arguments
    if spec == 'Kriging':
        start = timeit.default_timer()
        result = createSMTKrigingModel((mixint, x, y, regType, kernel, evall))
        end = timeit.default_timer()
        if result is not None:
            return result, end - start
        else:
            return None
    elif spec == 'KPLS':
        start = timeit.default_timer()
        result = createSMTKPLSModel((mixint, x, y, regType, kernel, evall))
        end = timeit.default_timer()
        if result is not None:
            return result, end - start
        else:
            return None
    elif spec == 'KPLSK':
        start = timeit.default_timer()
        result = createSMTKPLSModel((mixint, x, y, regType, kernel, evall))
        end = timeit.default_timer()
        if result is not None:
            return result, end - start
        else:
            return None
    else:
        raise NotImplementedError


# Making predictions using Kriging models
def predictKriging(arguments):
    # new_x: size × nVar
    new_x, KrgModel, uncertainty = arguments
    new_y = KrgModel.predict_values(new_x)
    if uncertainty:
        variance = KrgModel.predict_variances(new_x)
        return new_y, variance
    else:
        return new_y
