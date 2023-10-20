from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0.0
    else:
        return (1 / (1 + np.exp(-x)))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return (sigmoid(x) * (1 - sigmoid(x)))


def perceptron(x: np.ndarray, w: np.ndarray) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum = np.dot(x, w)
    output = sigmoid(weighted_sum)
    return weighted_sum, output


def ffnn(x: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1.0)
    
    # hidden layer z1
    a1, z1 = [], []
    for i in range(M):
        weighted_sum, output = perceptron(z0, W1[:, i])
        a1.append(weighted_sum)
        z1.append(output)
        
    z1 = [1.0] + z1
    a2, y = [], []
    for i in range(K):
        weighted_sum, output = perceptron(z1, W2[:, i])
        a2.append(weighted_sum)
        y.append(output)
    
    return np.array(y), np.array(z0), np.array(z1), np.array(a1), np.array(a2)



def backprop(x: np.ndarray, target_y: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    delta2 = y - target_y
    delta1 = np.dot(W2[1:, :], delta2) * a1 * (1 - a1)
    
    dE1 = np.zeros_like(W1)
    dE2 = np.zeros_like(W2)
    dE1 += np.dot(delta1, z0).T  # Transpose the outer product
    dE2 += np.dot(delta2, z1).T   # Transpose the outer product

    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    ...


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


if __name__ == "__main__":
    print("[+] Part 1.1")
    print(sigmoid(0.5))
    print(d_sigmoid(0.2))
    
    print("\n[+] Part 1.2")
    print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))
    
    print("\n[+] Part 1.3")
    # initialize the random generator to get repeatable results
    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    
    # initialize the random generator to get repeatable results
    np.random.seed(1234)
    
    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    print(y)
    print(z0)
    print(z1)
    print(a1)
    print(a2)
    
    print("\n[+] Part 1.4")
    # initialize random generator to get predictable results
    np.random.seed(42)
    
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    x = features[0, :]
    
    # create one-hot target for the feature
    target_y = np.zeros(K)
    target_y[targets[0]] = 1.0
    
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    
    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print(y)
    print(dE1)
    print(dE2)