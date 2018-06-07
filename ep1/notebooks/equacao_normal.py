#5619562

import numpy as np

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    """
    X = np.append(np.ones((len(X), 1), dtype=int), X, axis=1)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    prediction = np.dot(X, w)
    return prediction