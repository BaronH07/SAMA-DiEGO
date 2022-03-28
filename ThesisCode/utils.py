import numpy as np
from IOHexperimenter import IOH_function
from ioh import get_problem
from numpy import isfinite, mod, floor, shape, bitwise_and, zeros, newaxis
from smt.applications.mixed_integer import INT


def createBinInputSpace(dim):
    dtype = [INT] * dim
    range = [0, 1]
    drange = [range] * dim
    return dtype, drange


def eval(problem, x):
    return problem(x)


def batchEvaulateProblem(problemCall, xt):
    results = np.empty((xt.shape[0], 1))
    for i, x in enumerate(xt):
        results[i] = problemCall(x)
    return results


# For MI-ES, created by Hao and Bas
# TODO: double check this one. It causes the explosion of step-sizes in MIES
def boundary_handling(x, lb, ub):
    """

    This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image
    Analysis" as alorithm 6.

    """
    x = np.asarray(x, dtype='float')
    shape_ori = x.shape
    x = np.atleast_2d(x)
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)

    transpose = False
    if x.shape[0] != len(lb):
        x = x.T
        transpose = True

    lb, ub = lb.flatten(), ub.flatten()

    lb_index = isfinite(lb)
    up_index = isfinite(ub)

    valid = bitwise_and(lb_index, up_index)

    LB = lb[valid][:, newaxis]
    UB = ub[valid][:, newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = mod(floor(y), 2) == 0
    yprime = zeros(shape(y))
    yprime[I] = np.abs(y[I] - floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime

    if transpose:
        x = x.T
    return x.reshape(shape_ori)


# For BBOB discretization, created by Bartosz Piaskowski
## INTEGER FUNCTION UPDATED WITH NEW IOH ##
class IOH_int_function_new(IOH_function):
    '''A wrapper around the BBOB functions with mixed-integer inputs - continuous, integer and categorical
    '''

    def __init__(self, fid, dim, iid, int_mask=None):
        '''Instantiate a mixed-integer problem based on its function ID, dimension, instance, the number
        of integer variables of certain range

        Parameters
        ----------
        fid:
            The function ID of the problem in the suite, or the name of the function as string
        dim:
            The dimension (number of variables) of the problem
        iid:
            The instance ID of the problem
        int_mask:

            An array determining the type of function variables and in case of integer type, also it's range.
            For continuous variables, declare them as "0", for those who are supposed to be integer type, declare them as
            any positive integer, which will automatically set the bounds of this particular variable to be [0, chosen value].
                EXAMPLE:
            Function of 5 dimension: input array [0,0,40,0,30]:
            - 1st, 2nd and 4th variables are continuous with standard bounds of [-5, 5],
            - 3rd an 5th variables are integers with respective bounds of [0, 40] and [0, 30].

            The int_mask array has to match with the desired dimension of the function.

            Default: None - in this case only the first variable is of integer type with bounds [0,50]

        '''
        super().__init__(fid, dim, iid, target_precision=0, suite="BBOB")

        self.mixint_lb = self.lowerbound
        self.mixint_ub = self.upperbound

        if int_mask != None:
            if len(int_mask) != dim:
                raise Exception("Dimension of the mask doesn't match the dimension of the declared problem")
            # if 0 not in int_mask:
            #     raise Exception("The integer mask require at least one continuous variable, represented by 0")
            # else:
            self.int_mask = np.append(np.empty([dim, 0]), int_mask).astype(int)
        else:
            self.int_mask = np.append(np.empty([dim, 0]), 50).astype(int)

        self.mask_ind = np.where(self.int_mask)

        self.mixint_lb[self.mask_ind] = 0
        self.mixint_ub[self.mask_ind] = [i for i in self.int_mask[self.mask_ind]]

        # Create the transformation mappings dictionary
        self.trans_list = []
        for len_i in self.int_mask[self.mask_ind]:
            self.I_keys = list(range(int(len_i) + 1))
            self.C_values = np.linspace(-5, 5, len_i + 1)
            self.trans_list.append(dict(zip(self.I_keys, self.C_values)))

        self.f = get_problem(fid, iid, dim)

    def get_target_y(self):
        return self.f.objective.y

    def get_target_x(self):
        a = self.f.objective.x
        for i in self.mask_ind[0]:
            a[i] = list(self.trans_list[0].keys())[
                list(self.trans_list[0].values()).index(min(self.trans_list[0].values(),
                                                            key=lambda x: abs(x - a[i])))]
        return a

    def __call__(self, x):
        '''Evaluates the function in point x and deals with logging if needed

        Parameters
        ----------
        x:
            The point to evaluate

        Returns
        ------
        The value f(x)
        '''
        # The transformation of integer variables into corresponding equidistant continuous values from [-5,5] range
        # error = False
        # err_ind = []

        # Loop over the integer variables
        try:
            x_transformed = np.copy(x)
            if type(x) == dict: x_transformed = np.copy((list(x.values())))
        except:
            print("Input error: x has to be either an array, list, tuple or dict format")

        for list_ind, int_ind in enumerate(self.mask_ind[0]):
            x_transformed[int_ind] = self.trans_list[list_ind].get(x_transformed[int_ind])

        y = self.f(x_transformed)

        if self.y_comparison(y, self.yopt):
            self.yopt = y
            self.xopt = x
        if self.logger is not None:
            self.logger.process_parameters()
            self.logger.process_evaluation(self.f.loggerCOCOInfo())
        return y


############################################################
# For BBOB only
def createIntInputSpace(func_class):
    # func_class is an instance of the IOH_int_function_new
    lb = func_class.mixint_lb.astype(int)
    ub = func_class.mixint_ub.astype(int)
    dim = len(lb)
    dtype = [INT] * dim
    drange = np.vstack((lb, ub)).T
    drange = drange.tolist()
    return dtype, drange


def createMIESIntInputSpace(func_class):
    from .SearchSpace import OrdinalSpace
    lb = func_class.mixint_lb.astype(int)
    ub = func_class.mixint_ub.astype(int)
    dim = len(lb)
    drange = np.vstack((lb, ub)).T
    drange = drange.tolist()
    search_space = OrdinalSpace(drange)
    return search_space
############################################################
