import numpy as np
from IOHexperimenter import IOH_function
from ioh import get_problem


## MIXINTEGER FUNCTION UPDATED WITH NEW IOH ##
class IOH_mixint_function_new(IOH_function):
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
            if 0 not in int_mask:
                raise Exception("The integer mask require at least one continuous variable, represented by 0")
            else:
                self.int_mask = np.append(np.empty([dim, 0]), int_mask)
        else:
            self.int_mask = np.append(np.empty([dim, 0]), 50)

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
            a[i] = list(f.trans_list[0].keys())[list(f.trans_list[0].values()).index(min(f.trans_list[0].values(),
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


## COMPOSITE FUNCTION ##

class IOH_composite_function(IOH_mixint_function_new):
    '''A wrapper around the BBOB functions with mixed-integer inputs - continuous, integer and categorical
    '''

    def __init__(self, fid, dim, iid, int_mask=None, cat_mode=None, cat_fid=None, cat_iid=None):
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

            In case of the function being mix integer + categorical, the last variable is declared as the categorical one,
            regardless of the provided int_mask.

            DEFAULT: None - in this case only the first variable is of integer type with bounds [0,50]

            The int_mask array has to match with the desired dimension of the function!

        cat_mode:
            One of the three available modes of IOH benchmark categorical functions, the available options are:
            - "iid" - makes the categorical variable introduce certain transformations to the chosen function,
            - "fid" - makes the categorical variable variate between the chosen functions,
            - "fid+iid" - make the categorical variable variate between the chosen functions and their instances.
            Choosing any of this modes means overriding the previously chosen fid or idd, depending on the mode.

            DEFAULT: None - in this case the function is a simple mixed integer function with the provided int_mask.

        cat_fid:
            The mask of function ids to be used in the categorical variable. There are 24 available IOH functions available.
            EXAMPLE: A mask [1,2,3,5] means the categorical variable will alternate between the 4 provided functions
            with fid 1,2,3 and 5.

        cat_iid:
            The mask of instance ids to be used in the categorical variable.
            EXAMPLE: A mask [1,2,3,5] means the categorical variable will alternate between the 4 provided instance id
            1,2,3 and 5 of the function with fid and dim provided.



        '''
        super().__init__(fid, dim, iid, int_mask)
        self.cat_mode = cat_mode
        self.cat_fid = sorted(cat_fid) if cat_fid != None else cat_fid
        self.cat_iid = sorted(cat_iid) if cat_iid != None else cat_iid

        if int_mask != None:
            if len(int_mask) == dim:
                self.int_mask = int_mask[:-1]
            else:
                raise Exception("The dimensions of the provided mask doesn't match the dimensions of the function.")

        if self.cat_mode == "fid":
            self.cat_values = [i + 1 for i in range(len(self.cat_fid))]
            self.cat_list = [IOH_mixint_function_new(i, dim - 1, iid, self.int_mask) for i in self.cat_fid]
            self.ytargets = [self.cat_list[i - 1].f.objective.y for i in self.cat_values]
            self.xtargets = [self.cat_list[i - 1].get_target_x() for i in self.cat_values]
            print("Composite function with instance id {} with categories alteranting between functions {}".format(iid,
                                                                                                                   self.cat_fid))


        elif self.cat_mode == "iid":
            self.cat_values = [i + 1 for i in range(len(self.cat_iid))]
            self.cat_list = [IOH_mixint_function_new(fid, dim - 1, i, self.int_mask) for i in self.cat_iid]
            self.ytargets = [self.cat_list[i - 1].f.objective.y for i in self.cat_values]
            self.xtargets = [self.cat_list[i - 1].get_target_x() for i in self.cat_values]
            print("Composite function with function id {} with categories alternating between instances {}".format(fid,
                                                                                                                   self.cat_iid))

        elif self.cat_mode == "fid_iid":
            if self.cat_fid and self.cat_iid != None:
                self.combinations = [[i, j] for i in self.cat_fid for j in self.cat_iid]
                self.cat_values = [i + 1 for i in range(len(self.combinations))]
                self.cat_list = [IOH_mixint_function_new(i, dim - 1, j, self.int_mask) for i, j in self.combinations]
                self.ytargets = [self.cat_list[i - 1].f.objective.y for i in self.cat_values]
                self.xtargets = [self.cat_list[i - 1].get_target_x() for i in self.cat_values]
                print(
                    "Composite function with categorical variable alternating between functions {} and their instances {}".format(
                        self.cat_fid, self.cat_iid))
            else:
                raise Exception("Provide list of fid and iid for this categorical mode")
        else:
            self.mixint_func = IOH_mixint_function_new(fid, dim, iid, int_mask)
            self.ftarget = self.mixint_func.f.objective.y
            print("Regular mixed-integer function with the default [50,0..] mask or the provided int_mask")

        if self.cat_mode != None:
            self.comp_lb = np.append(self.mixint_lb[:-1], 1)
            self.comp_ub = np.append(self.mixint_ub[:-1], self.cat_values[-1])
        else:
            self.comb_lb = self.mixint_lb
            self.comb_ub = self.mixint_ub

    def get_target_y(self):
        return min(self.ytargets)

    def get_target_x(self):
        self.y_id = [i for i, value in enumerate(self.ytargets) if value == min(self.ytargets)]
        xtarget = np.copy(self.xtargets[y_id[0]])
        xtarget = np.append(xtarget, self.cat_values[self.y_id[0]])
        return xtarget

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
        if self.cat_mode != None:
            try:
                x[-1] not in self.cat_values == True
            except Custom_error:
                print("Categorical variable out of range")
            y = self.cat_list[x[-1] - 1](x[:-1])
        else:
            y = self.mixint_func(x)

        if self.y_comparison(y, self.yopt):
            self.yopt = y
            self.xopt = x
        if self.logger is not None:
            self.logger.process_parameters()
            self.logger.process_evaluation(
                [sum([self.cat_list[i].f.state.evaluations for i in range(len(self.cat_list))]),
                 y - self.get_target_y(), self.yopt - self.get_target_y(), y, self.yopt])
        return y
