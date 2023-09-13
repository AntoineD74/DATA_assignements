import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(features: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    width_f, height_f = features.shape
    width_mu = mu.shape[0]
    fi = np.zeros((width_f, width_mu))
    covariance = sigma * np.eye(height_f)

    for i in range(width_mu):
        mvn = multivariate_normal(mean=mu[i], cov=covariance)
        fi[:, i] = mvn.pdf(features)

    return fi

def _plot_mvn(features: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    plt.figure(figsize=(12, 6))
    fi = mvn_basis(features, mu, sigma)
    for j in range(fi.shape[1]):
        plt.plot(fi[:, j])
    plt.show()

def max_likelihood_linreg(fi: np.ndarray, targets: np.ndarray, lamda: float) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    k = np.matmul(fi.T, fi) #fi*fi^T
    dim_M = np.identity(fi.shape[1])
    
    k_lambda = np.linalg.inv(k + lamda * dim_M)
    k_lambda_fi = np.dot(k_lambda, fi.T)
    wml = np.dot(k_lambda_fi, targets)
    return wml

def linear_model(features: np.ndarray, mu: np.ndarray, sigma: float, w: np.ndarray) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    basis_fi = np.array(mvn_basis(features, mu, sigma))
    prediction = np.dot(basis_fi, w)
    return prediction

def _square_error(y, y_hat):
    return np.power((y-y_hat), 2)

def update_mean(mu: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    return ((n * mu + x) / (n + 1))

if __name__ == '__main__':
    
    print('[+]Part 1.1')
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    
    fi = mvn_basis(X, mu, sigma)
    print(f'fi: {fi}')
    
    print('\n[+]Part 1.2: Plotting...')
    _plot_mvn(X, mu, sigma)
    
    print('\n[+]Part 1.3')
    fi = mvn_basis(X, mu, sigma) # same as before
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print(f'wml: {wml}')
    
    print('\n[+]Part 1.4')
    prediction = linear_model(X, mu, sigma, wml)
    print(f'prediction = {prediction}')
    
    print('\n[+]Part 1.5: Plotting...')
    estimates = []
    errors = []
    current_mean = np.array([0, 0, 0])
    actual_mean = np.mean(prediction, axis=0)
    for i in range(prediction.shape[0]):
        current_mean = update_mean(current_mean, prediction[i], i)
        estimates.append(current_mean)
        
        errors.append(np.mean(_square_error(current_mean, actual_mean)))
    
    errors = np.array(errors)
    
    plt.figure(figsize=(12, 6))
    plt.plot(errors)
    plt.xlabel('Data Point')
    plt.ylabel('Average Squared Error')
    plt.show()


