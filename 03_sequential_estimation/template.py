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
    return ((n * mu + x) / (n + 1))


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
    return np.power((y-y_hat), 2)


def _plot_mean_square_error():
    initial_estimates = np.array([0, 0, 0])
    dataX = gen_data(100, 3, initial_estimates, 1)
    
    actual_mean = np.mean(dataX, axis=0)
    
    estimates = []
    errors = []
    current_mean = initial_estimates
    for i in range(dataX.shape[0]):
        current_mean = update_sequence_mean(current_mean, dataX[i], i)
        estimates.append(current_mean)
        
        errors.append(np.mean(_square_error(current_mean, actual_mean)))
    
    errors = np.array(errors)
    
    #Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(100), errors)
    plt.xlabel('Data Point')
    plt.ylabel('Average Squared Error')
    plt.show()


if __name__ == '__main__':
   
    print("\n[+]Part 1.1")
    np.random.seed(1234)
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    np.random.seed(1234)
    print(gen_data(5, 1, np.array([0.5]), 0.5))
    
    print("\n[+]Part 1.2: Plotting...")
    np.random.seed(1234)
    X = gen_data(300, 3, np.array([0, 1, -1]), 1.73)
    scatter_3d_data(X)
    bar_per_axis(X)
    
    print("\n[+]Part 1.4: Plotting...")
    mean = np.mean(X, 0)
    #np.random.seed(1234)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    print(update_sequence_mean(mean, new_x, X.shape[0]))
    
    print("\n[+]Part 1.5: Plotting...")
    np.random.seed(1234)
    initial_estimates = np.array([0, 0, 0])
    data = gen_data(100, 3, initial_estimates, 1)
    
    estimates = []
    current_mean = initial_estimates
    for i in range(data.shape[0]):
        current_mean = update_sequence_mean(initial_estimates, data[i], i)
        estimates.append(current_mean)
        
    estimates = np.array(estimates)
    plt.figure(figsize=(12, 6))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()
    
    print("\n[+]Part 1.6: Plotting...")
    np.random.seed(1234)
    _plot_mean_square_error()