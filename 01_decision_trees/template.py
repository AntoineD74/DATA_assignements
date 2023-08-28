# Author: Antoine DUPUIS
# Date: 27/08/2023
# Project: DATA lecture assignement
# Acknowledgements: /
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test

#Part 1.1
def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    numberTrues = 0
    output = [0] * len(classes)
    total = len(targets)
    for i in range(len(targets)):
        output[targets[i]]+=1
        
    for j in range(len(output)):
        output[j]/=total
        
    return output


def split_data(features: np.ndarray, targets: np.ndarray, split_feature_index: int, theta: float) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    split_feature = features[:, split_feature_index]
    
    mask_1 = split_feature <= theta #Conditional masks applied
    mask_2 = split_feature > theta
    
    features_1 = features[mask_1]   
    targets_1 = targets[mask_1]

    features_2 = features[mask_2]
    targets_2 = targets[mask_2]
    
    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    nb_samples = len(targets)
    sumPC = 0
    for i in classes:
        current_s = np.count_nonzero(targets == i)/nb_samples
        sumPC += np.power(current_s, 2)
    
    return (0.5*(1-sumPC))


def weighted_impurity(t1: np.ndarray, t2: np.ndarray, classes: list) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    
    numerator1 = t1.shape[0] * g1
    numerator2 = t2.shape[0] * g2
    
    n = t1.shape[0] + t2.shape[0]
    return ((numerator1 + numerator2)/n)


def total_gini_impurity(features: np.ndarray, targets: np.ndarray, classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (feat_1, targ_1), (feat_2, targ_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(targ_1, targ_2, classes)


def brute_best_split(features: np.ndarray, targets: np.ndarray, classes: list, num_tries: int) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    
    # iterate feature dimensions
    for i in range(features.shape[1]):

        min_value = features[:, i].min()
        max_value = features[:, i].max()
        thetas = np.linspace(min_value, max_value, num_tries+2)[1:-1]   # generate tresholds     
        
        # iterate thresholds
        for theta in thetas:
            gini_impurity = total_gini_impurity(features, targets, classes, i, theta)
            
            if gini_impurity < best_gini:
                best_gini = gini_impurity
                best_threshold = theta
                best_dim = i
    
    return best_gini, best_dim, best_threshold


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        ...

    def accuracy(self):
        ...

    def plot(self):
        ...

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        ...

    def confusion_matrix(self):
        ...
        
if __name__ == '__main__':
    
    #Test 1.1
    print("[+]Part 1.1")
    print(prior([0, 0, 1], [0, 1]))
    print(prior([0, 2, 3, 3], [0, 1, 2, 3]))
    
    #Test 1.2
    print("\n[+]Part 1.2")
    features, targets, classes = load_iris()
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    print("Samples in f_1:", f_1.shape[0])
    print("Samples in f_2:", f_2.shape[0])
    
    #Test 1.3
    print("\n[+]Part 1.3")
    print(gini_impurity(t_1, classes))
    print(gini_impurity(t_2, classes))
    
    #Test 1.4
    print("\n[+]Part 1.4")
    print(weighted_impurity(t_1, t_2, classes))
    
    #Test 1.5
    print("\n[+]Part 1.5")
    print(total_gini_impurity(features, targets, classes, 2, 4.65))
    
    #Test 1.6
    print("\n[+]Part 1.6")
    print(brute_best_split(features, targets, classes, 30))
    
    
    
    
    
    
    
    
