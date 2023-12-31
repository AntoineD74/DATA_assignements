from typing import Union
import numpy as np

from tools import load_iris, split_train_test
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    x = np.clip(x, -100, None)  # Will return 0.0 in case of overflow
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

    deltak = y - target_y
    deltaj= np.dot(W2[1:, :], deltak) * d_sigmoid(a1)
    
    dE1 = np.zeros_like(W1)
    dE2 = np.zeros_like(W2)
    dE1 += np.outer(z0, deltaj)
    dE2 += np.outer(z1, deltak)

    return y, dE1, dE2


def train_nn(X_train: np.ndarray, t_train: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray, iterations: int, eta: float) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    E_total = []
    misclassification_rate = []
    N = X_train.shape[0]
    last_guesses = []
    for _ in range(iterations):
        dE1_total = np.zeros_like(W1)
        dE2_total = np.zeros_like(W2)
        total_error = 0
        misclassifications = 0

        for i in range(N):
            x = X_train[i, :]
            target_y = np.zeros(K)
            target_y[t_train[i]] = 1.0
            
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            total_error -= np.sum(target_y * np.log(y) + (1 - target_y) * np.log(1 - y))

            predicted_class = np.argmax(y)
            
            if predicted_class != t_train[i]:
                misclassifications += 1

        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        misclassification_rate.append(misclassifications / N)
        E_total.append(total_error / N)

    for j in range(N):
        y, z0, z1, a1, a2 = ffnn(X_train[j, :], M, K, W1, W2)
        last_guesses.append(np.argmax(y))

    return W1, W2, E_total, misclassification_rate, np.array(last_guesses, dtype=float)


def test_nn(X: np.ndarray,M: int, K: int, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = X.shape[0]
    guesses = np.zeros(N, dtype=int)
    
    for i in range(N):
      y, _, _, _, _ = ffnn(X[i, :], M, K, W1, W2)
      guess = np.argmax(y)
      guesses[i] = guess
   
    return guesses


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
    
    print("\n[+] Part 2.1")
    np.random.seed(1234)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    print(W1tr)
    print(W2tr)
    print(Etotal)
    print(misclassification_rate)
    print(last_guesses)
    
    print("\n[+] Part 2.3")
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:, :], train_targets[:], M, K, W1, W2, 500, 0.1)  #training on 80% of the dataset
    
    predictions = test_nn(test_features, M, K, W1tr, W2tr)
    
    #print(misclassification_rate)
    #print(Etotal)
    print(predictions)
    print(test_targets)
    confusion = confusion_matrix(test_targets, np.array(predictions))
    
    print(f"\nAccuracy: {np.count_nonzero(last_guesses == train_targets) / len(train_targets) * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion)
    
    plt.plot(Etotal)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()
    
    plt.plot(misclassification_rate)
    plt.xlabel("Epochs")
    plt.ylabel("Misclassification rate")
    plt.show()
    