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
    X, _ = load_cancer()
    X = standardize(X)
    
    pca = PCA(n_components=30)
    pca.fit(X)
    
    plt.figure(figsize=(25, 20))
    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.plot(X[:, i])
        plt.title(f'PCA {i + 1}')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def _plot_eigen_values():
   X, _ = load_cancer()
   X = standardize(X)
   
   pca = PCA(n_components=30)
   pca.fit(X)
   
   eigenvalues = pca.explained_variance_
   plt.plot(eigenvalues)
   plt.xlabel('Eigenvalue index')
   plt.ylabel('Eigenvalue')
   plt.grid()
   plt.show()


def _plot_log_eigen_values():
    X, _ = load_cancer()
    X = standardize(X)
    
    pca = PCA(n_components=30)
    pca.fit(X)
    
    eigenvalues = pca.explained_variance_
    log_eigenvalues = np.log10(eigenvalues)
    
    plt.plot(log_eigenvalues)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    X, _ = load_cancer()
    X = standardize(X)
    
    pca = PCA(n_components=30)
    pca.fit(X)
    
    eigenvalues = pca.explained_variance_
    cum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    plt.plot(cum_variance)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.xticks(np.arange(1, 30, step=5))
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
    
    print("\n[+]Part 1.3: Plotting ...")
    _scatter_cancer()
    plt.show()
    
    print("\n[+]Part 2.1 Plotting ...")
    _plot_pca_components()
    
    print("\n[+]Part 3.1: Plotting ...")
    _plot_eigen_values()
    
    print("\n[+]Part 3.2: Plotting ...")
    _plot_log_eigen_values()
    
    print("\n[+]Part 3.3: Plotting ...")
    _plot_cum_variance()