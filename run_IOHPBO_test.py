# Function group: 1,2,3,[4~17] are variations of 1,2, 18, [19,20,21], 22, 23, 24, 25
import argparse
import json
import os.path
from pathlib import Path

from ioh import get_problem

from ThesisCode.KrigingSMT import createIOHPBOKrgConfig
from ThesisCode.Pipeline import pipeline
from ThesisCode.RBFRoy import createIOHPBORBFkernels
from ThesisCode.SVMSklearn import defaultSVMsettings
from ThesisCode.utils import createBinInputSpace


def create_func(f_id, iid, func_dim):
    func = get_problem(f_id, iid, func_dim, 'PBO')
    return func


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments according to IOHPBO_config.json')
    parser.add_argument('-F', metavar='Functions', type=int,
                        help='1~6, the index of function groups', required=True)
    parser.add_argument('-I', metavar='Instance', type=int,
                        help='1~2, the index of instance', default=1)
    parser.add_argument('-D', metavar='Dimensionality', type=int, default=2,
                        help="1~3, dimensionality of the problem, 25 or 64 or 100")
    parser.add_argument('-A', metavar='Algorithms', type=int,
                        help='1~12, benchmark functions, 9(PV-ES), 10(EI-ES), 11(PV-EA) and 12(EI-EA) is our algorithms.', required=True)
    # parser.add_argument('-W', metavar='Weights', type=int, nargs=1, default=1, help='1, for half-half')
    parser.add_argument('-R', metavar='Run', type=int, required=True, help='the index of an independent run')
    args = parser.parse_args()
    input_configs = vars(args)
    try:
        with open('./IOHPBO_config.json') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print("Create all_config.json using Make_settings.py")
        raise FileNotFoundError
    # Function ID
    Fid = input_configs['F'] - 1
    # Dimensionality ID
    Did = input_configs['D'] - 1
    # Instance ID
    Iid = input_configs['I'] - 1
    # Algorithm ID
    Aid = input_configs['A'] - 1
    func_id = config['Func_ids'][Fid]
    dims = config['Dims'][Did]
    iid = config['Inst'][Iid]
    if Aid < 8:
        Alg_names = config['Algorithms'][Aid]
    else:
        Alg_names = 'Ours'
    Space = 'O'
    func = create_func(func_id, iid, dims)
    func1 = create_func(func_id, iid, dims)
    runNo = input_configs['R']
    caseName = './Case'+str(Fid+1)+'/'
    logName = 'run_num_'+str(runNo)+'.csv'
    newPath = os.path.join(caseName, Alg_names)
    Path(newPath).mkdir(parents=True, exist_ok=True)
    logs = os.path.join(newPath, logName)
    print(func1)
    Krg = createIOHPBOKrgConfig()
    RBF = createIOHPBORBFkernels()
    svm = defaultSVMsettings(dims)
    initSize = dims + 1
    if Aid == 8: #SAMA-DiEGO-A
        dtype, drange = createBinInputSpace(dims)
        fp = 'case1'
        fp = os.path.join(newPath, fp)
        if dims == 25:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='mies', KrgGap=1, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        elif 25 < dims < 100:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='mies', KrgGap=10, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        else:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='mies', KrgGap=20, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
    elif Aid == 9: #SAMA-DiEGO-B
        dtype, drange = createBinInputSpace(dims)
        fp = 'case2'
        fp = os.path.join(newPath, fp)
        if dims == 25:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='mies', KrgGap=1, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        elif 25 < dims < 100:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='mies', KrgGap=10, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        else:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='mies', KrgGap=20, Space=Space,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)

    elif Aid == 10: #SAMA-DiEGO-C
        dtype, drange = createBinInputSpace(dims)
        fp = 'case3'
        fp = os.path.join(newPath, fp)
        if dims == 25:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='twoRateEA', KrgGap=1,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        elif 25 < dims < 100:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='twoRateEA', KrgGap=10,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        else:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=3, nTotal=7,
                                      optimizer_log=None, filePath=fp, ICtype='PV', Optimizer='twoRateEA', KrgGap=20,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
    elif Aid == 11: #SAMA-DiEGO-D
        dtype, drange = createBinInputSpace(dims)
        fp = 'case4'
        fp = os.path.join(newPath, fp)
        if dims == 25:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='twoRateEA', KrgGap=1,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        elif 25 < dims < 100:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='twoRateEA', KrgGap=10,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)
        else:
            foundX, foundY = pipeline(func, drange, dtype, initSize=initSize, maxEval=500, minimization=False, runNo=runNo, nSingleModel=2, nTotal=7,
                                      optimizer_log=None, ICtype='EI', filePath=fp, Optimizer='twoRateEA', KrgGap=20,
                                      KrgType=Krg, RBFKernels=RBF, SVMConfig=svm)


