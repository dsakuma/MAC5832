import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    # YOUR CODE HERE:
    mean = X.mean(axis=0)
    std_dev = X.std(axis=0)
    X_out = (X-mean)/std_dev
    # END YOUR CODE

    return X_out


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """
    
    # YOUR CODE HERE:
    N = X.shape[0]
    J = ((X.dot(w) - y).T).dot(X.dot(w) - y) / N
    J = J[0][0]
    # END YOUR CODE

    return J


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    # YOUR CODE HERE:
    N = X.shape[0]
    grad = ((X.dot(w) - y).T.dot(X))*2/N
    grad = grad[0]
    # END YOUR CODE

    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]

    # YOUR CODE HERE:
    for i in range(1, num_iters):
        #compute grad
        grad = compute_wgrad(X, y, w)
        grad = grad.reshape((2,1))
        #apply weights
        w = w-(learning_rate*grad)
        weights_history.append(w)
        #compute cost
        Jw =  compute_cost(X, y, w)
        #save cost 
        cost_history.append(Jw)
        num_iters += 1
    # END YOUR CODE

    return w, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    
    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]

    # YOUR CODE HERE:
    for i in range(1, num_iters):
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        Xbatch = X[idx]
        ybatch = y[idx]
        #compute grad
        grad = compute_wgrad(Xbatch, ybatch, w)
        grad = grad.reshape((2,1))
        #apply weights
        w = w-(learning_rate*grad)
        weights_history.append(w)
        #compute cost
        Jw =  compute_cost(X, y, w)
        #save cost 
        cost_history.append(Jw)
        num_iters += 1
    # END YOUR CODE

    return w, weights_history, cost_history
