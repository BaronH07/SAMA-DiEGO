import itertools
import timeit

from sklearn.svm import SVR

# SVM regressor
"""
Epsilon-Support Vector Regression.

The free parameters in the model are C and epsilon.

The implementation is based on libsvm. The fit time complexity
is more than quadratic with the number of samples which makes it hard
to scale to datasets with more than a couple of 10000 samples. For large
datasets consider using :class:`~sklearn.svm.LinearSVR` or
:class:`~sklearn.linear_model.SGDRegressor` instead, possibly after a
:class:`~sklearn.kernel_approximation.Nystroem` transformer.
"""


# The default settings for SVM regression
def defaultSVMsettings(nVar):
    # k: choice from ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # d: degree of poly kernel, 2, 3 or 5
    # c: Regularization parameter, default is 1
    # e: float, default=0.5 (config obtained from https://dl.acm.org/doi/pdf/10.1145/3321707.3321801)
    #       Epsilon in the epsilon-SVR model.
    #       It specifies the epsilon-tube within which no penalty is associated in the training loss function with
    #       points predicted within a distance epsilon from the actual value.
    # cache_size: float, default=400. Specify the size of the kernel cache (in MB).
    # max_iter: int, default=-1. Hard limit on iterations within solver, or -1 for no limit.

    k = ['linear', 'rbf', 'sigmoid']
    normal_d = [3]
    k_poly = ['poly']
    poly_d = [2, 3, 5]
    c = [1]  # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    e = [0.5]
    cache_size = [1024]  # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    if nVar <= 20:
        max_iter = [-1]  # unlimited iterations
    else:
        max_iter = [5e3 * nVar]
    config = list(itertools.product(k, normal_d, c, e, cache_size, max_iter))
    poly_config = list(itertools.product(k_poly, poly_d, c, e, cache_size, max_iter))
    config = config + poly_config
    return config


# Create SVM regression
def createSVM(arguments):
    # X, Y, k='rbf', d=3, c=1.0, e=0.5, cache_size=500, max_iter=-1
    X, Y, k, d, c, e, cache_size, max_iter = arguments
    # X: data samples
    # Y: target values
    # k: choice from ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # d: degree of poly kernel, 2, 3 or 5
    # c: Regularization parameter, default is 1
    # e: float, default=0.5 (config in https://dl.acm.org/doi/pdf/10.1145/3321707.3321801)
    #       Epsilon in the epsilon-SVR model.
    #       It specifies the epsilon-tube within which no penalty is associated in the training loss function with
    #       points predicted within a distance epsilon from the actual value.
    # cache_size: float, default=500. Specify the size of the kernel cache (in MB).
    # max_iter: int, default=-1. Hard limit on iterations within solver, or -1 for no limit.
    svr = SVR(kernel=k, degree=d, C=c, epsilon=e, cache_size=cache_size, max_iter=max_iter)
    svr.fit(X, Y)
    return svr


# Verify SVM regression, only for model verification
def verifySVM(arguments):
    # X, Y, k='rbf', d=3, c=1.0, e=0.5, cache_size=500, max_iter=-1
    X, Y, k, d, c, e, cache_size, max_iter = arguments
    # X: data samples
    # Y: target values
    # k: choice from ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # d: degree of poly kernel, 2, 3 or 5
    # c: Regularization parameter, default is 1
    # e: float, default=0.5 (config in https://dl.acm.org/doi/pdf/10.1145/3321707.3321801)
    #       Epsilon in the epsilon-SVR model.
    #       It specifies the epsilon-tube within which no penalty is associated in the training loss function with
    #       points predicted within a distance epsilon from the actual value.
    # cache_size: float, default=500. Specify the size of the kernel cache (in MB).
    # max_iter: int, default=-1. Hard limit on iterations within solver, or -1 for no limit.
    start = timeit.default_timer()
    svr = SVR(kernel=k, degree=d, C=c, epsilon=e, cache_size=cache_size, max_iter=max_iter)
    svr.fit(X, Y)
    end = timeit.default_timer()
    return svr, end - start


def predictSVM(arguments):
    x, svr = arguments
    return svr.predict(x)
