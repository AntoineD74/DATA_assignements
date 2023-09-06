from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(n: int, k: int, mean: np.ndarray, var: float) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov = np.power(var, 2)*np.identity(k)
    x_array = np.array(np.random.multivariate_normal(mean, cov, size=n)) 
    return x_array

def update_sequence_mean(mu: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    pass


def _plot_sequence_estimate():
    data = None # Set this as the data
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        """
            your code here
        """
    plt.plot([e[0] for e in estimates], label='First dimension')
    """
        your code here
    """
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    pass


def _plot_mean_square_error():
    pass


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass

if __name__ == '__main__':
    
    print("\n[+]Part 1.1")
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    
