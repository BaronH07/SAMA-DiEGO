import timeit

import numpy as np

from .RbfInter import trainRBF, predictRBFinter


# Create RBF models
def createRoyRBFmodel(arguments):
    X, Y, RBFconfig, evall = arguments
    ptail = RBFconfig[0]
    squares = RBFconfig[1]
    smooth = RBFconfig[2]
    rbfkernel = RBFconfig[3]
    try:
        rbfModel = trainRBF(X, Y, ptail, squares, smooth, rbfkernel)
    except Exception:
        return None
    return rbfModel


# Verify RBF models
# The same as createRoyRBFmodel but with timer
def verifyRoyRBFmodel(arguments):
    X, Y, RBFconfig, evall = arguments
    ptail = RBFconfig[0]
    squares = RBFconfig[1]
    smooth = RBFconfig[2]
    rbfkernel = RBFconfig[3]
    try:
        start = timeit.default_timer()
        rbfModel = trainRBF(X, Y, ptail, squares, smooth, rbfkernel)
        end = timeit.default_timer()
        time = end - start
    except Exception:
        return None
    return rbfModel, time


# Making predictions using RBF models
def predictRoyRBFmodel(arguments):
    new_x, rbfModel, uncertainty = arguments
    # new_x: size × nVar
    if uncertainty:  # For probabilistic based infill criteria
        results = predictRBFinter(rbfModel, new_x, uncertainty)
        results = np.array(results)
        new_y = np.reshape(results[:, 0], (new_x.shape[0],))
        prob = np.reshape(results[:, 1], (new_x.shape[0],))
        return new_y, prob
    else:
        new_y = predictRBFinter(rbfModel, new_x, uncertainty)
        new_y = np.array(new_y)
        return np.reshape(new_y, (new_x.shape[0],))


# Replace kernels in configurations (list), a util function
def replaceKernel(RBFconfig, RBFKernels, ind):
    RBFconfig[-1] = RBFKernels[ind]
    return RBFconfig


# The suggested kernels for BBOB problems
def createBBOBRBFkernels():
    RBFKernels = ["CUBIC", "THINPLATESPLINE",
                  "POLYHARMONIC1", "POLYHARMONIC4", "MULTIQUADRIC"]
    return RBFKernels


# All kernels of RBF
# ~@Copyright goes to MSc. Roy de Winter
def createIOHPBORBFkernels():
    RBFKernels = ["CUBIC", "THINPLATESPLINE", "INVQUADRIC",
                                           "POLYHARMONIC1", "POLYHARMONIC4", "POLYHARMONIC5",
                                           "MULTIQUADRIC", "GAUSSIAN", "INVMULTIQUADRIC"]
    return RBFKernels