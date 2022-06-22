from .RandomForestHao import RandomForest


def createRandomForest(arguments):
    X, y = arguments
    rf = RandomForest()
    rf.fit(X, y)
    return rf


def predictRandomForest(arguments):
    X, rfmodel, uncertainty = arguments
    if uncertainty:
        new_y, variance = rfmodel.predict(X, eval_MSE=uncertainty)
        return new_y, variance
    else:
        new_y = rfmodel.predict(X, eval_MSE=uncertainty)
        return new_y

