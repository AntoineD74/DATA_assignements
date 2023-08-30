# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum = 0
    for i in range(len(x)):
        sum += np.power((x[i] - y[i]), 2)
    
    sum = np.power(sum, 0.5)
    return sum

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
        
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
        
    distances = np.argsort(distances)
    return distances[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    classes_counts = np.bincount(targets)
    most_common = np.argmax(classes_counts)
    return most_common
    


def knn(x: np.ndarray, points: np.ndarray,point_targets: np.ndarray, classes: list, k: int) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_closest = k_nearest(x, points, k)
    k_closest_array = np.zeros(k_closest.shape[0], dtype=int)
    for i in range(len(k_closest)):
        k_closest_array[i] = point_targets[k_closest[i]]
        
    return vote(k_closest_array, np.array(classes))
    
        
def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    ...


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    ...


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    ...


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    ...


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    ...


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...
    
if __name__ == '__main__':
    d, t, classes = load_iris()
    plot_points(d, t)
    
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]
    
    print("\n[+]Part 1.1")
    print(euclidian_distance(x, points[0]))
    print(euclidian_distance(x, points[50]))
    
    print("\n[+]Part 1.2")
    print(euclidian_distances(x, points))
    
    print("\n[+]Part 1.3")
    print(k_nearest(x, points, 1))
    print(k_nearest(x, points, 3))
    
    print("\n[+]Part 1.4")
    print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    print(vote(np.array([1,1,1,1]), np.array([0,1])))
    print(vote(np.array([0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2 ,1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ,2 ,2 ,2 ,2]), np.array([0,1,2])))

    print("\n[+]Part 1.5")
    print(knn(x, points, point_targets, classes, 1))
    print(knn(x, points, point_targets, classes, 5))
    print(knn(x, points, point_targets, classes, 150))

    