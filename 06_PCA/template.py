# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X)
    std_dev = np.std(X)    
    
    return ((X - mean) / std_dev)


def scatter_standardized_dims(X: np.ndarray, i: int, j: int):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_standardized = standardize(X)
    x_i = X_standardized[:, i]
    x_j = X_standardized[:, j]
    
    plt.scatter(x_i, x_j)


def _scatter_cancer():
    X, y = load_cancer()
    
    plt.figure(figsize=(25, 20))
    for i in range(30):
        plt.subplot(5, 6, i+1)
        scatter_standardized_dims(X, 0, i)
    
    plt.tight_layout()
    plt.show()
    

def _plot_pca_components():
    ...
    X, y = load_cancer()
    for i in range(...):
        plt.subplot(5, 6, ...)
        ...
    plt.show()


def _plot_eigen_values():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    print("[+]Part 1.1")
    print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))
    
    
    print("\n[+]Part 1.2: Plotting ...")
    X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
    scatter_standardized_dims(X, 0, 2)
    plt.show()
    
    print("\n[+]Part 1.3")
    _scatter_cancer()
    plt.show()