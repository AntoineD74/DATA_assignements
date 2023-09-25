from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(features: np.ndarray, targets: np.ndarray, selected_class: int) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    feature_for_specific_class = []
    for i in range(len(features)):
        if(targets[i]==selected_class):
             feature_for_specific_class.append(features[i])
             
    means = []
    feature_for_specific_class = np.array(feature_for_specific_class)
    for j in range(features.shape[1]):
        means.append(np.mean(feature_for_specific_class[:, j]))
        
    return np.array(means)


def covar_of_class(features: np.ndarray, targets: np.ndarray, selected_class: int) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    feature_for_specific_class = []
    for i in range(len(features)):
        if(targets[i]==selected_class):
             feature_for_specific_class.append(features[i])
             
    return np.cov(feature_for_specific_class, rowvar=False)
     

def likelihood_of_class(feature: np.ndarray, class_mean: np.ndarray, class_covar: np.ndarray) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    ...
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)



def maximum_likelihood(train_features: np.ndarray, train_targets: np.ndarray, test_features: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))        
        
    likelihoods = np.zeros((test_features.shape[0], len(classes)))
    for i in range(test_features.shape[0]):
        feature = test_features[i, :]
        
        for j in range(len(classes)):
            likelihood = multivariate_normal.pdf(feature, mean=means[j], cov=covs[j])
            likelihoods[i, j] = likelihood
        
    return likelihoods


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(train_features: np.ndarray, train_targets: np.ndarray, test_features: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    prior_probabilities = [] 
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))  
        prior_probabilities.append(np.sum(train_targets == class_label) / len(train_targets))
        
    print(prior_probabilities)
    likelihoods = np.zeros((test_features.shape[0], len(classes)))
    for i in range(test_features.shape[0]):
        feature = test_features[i, :]
        
        for j in range(len(classes)):
            likelihood = multivariate_normal.pdf(feature, mean=means[j], cov=covs[j])
            likelihoods[i, j] = likelihood * prior_probabilities[j] #aposteriori likelihood
        
    return likelihoods

def confusion_matrix(prediction, actual_targets):
    confusion_matrix = np.zeros((len(prediction), len(prediction)), dtype=int)
    
    for actual, predicted in zip(actual_targets, prediction):
       confusion_matrix[actual, predicted] += 1
        
    return confusion_matrix


'''
if __name__ == '__main__':
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)
    
    print("[+]Part 1.1")
    print(mean_of_class(train_features, train_targets, 0))
    
    print("\n[+]Part 1.2")
    print(covar_of_class(train_features, train_targets, 0))
    
    print("\n[+]Part 1.3")
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    print(likelihood_of_class(test_features[0, :], class_mean, class_cov))
    
    print("\n[+]Part 1.4")
    print(maximum_likelihood(train_features, train_targets, test_features, classes))
    
    print("\n[+]Part 1.5")
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    prediction = predict(likelihoods)
    print(prediction)
    correct_predictions = np.sum(prediction == test_targets)
    accuracy = 100*correct_predictions / len(prediction)
    print(f'Accuracy maximum_likelihood: {accuracy}%')
    print(confusion_matrix(prediction, test_targets))
    
    print("\n[+]Part 2.1")
    likelihoods = maximum_aposteriori(train_features, train_targets, test_features, classes)
    prediction = predict(likelihoods)
    print(prediction)
    correct_predictions = np.sum(prediction == test_targets)
    accuracy = 100*correct_predictions / len(prediction)
    print(f'Accuracy maximum_aposteriori: {accuracy}%')
    print(confusion_matrix(prediction, test_targets))
'''