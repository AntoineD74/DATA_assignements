# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

import help as util


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
    nearest_indices = k_nearest(x, points, k)
    nearest_targets = point_targets[nearest_indices]
    return vote(nearest_targets, classes)
    
        
def knn_predict(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> np.ndarray:

    predictions = []
    for i in range(len(points)):
        point_minus_current = util.remove_one(points, i)
        point_targets_minus_current = util.remove_one(point_targets, i)
        predictions.append(knn(points[i], point_minus_current, point_targets_minus_current, classes, k))
    return np.array(predictions)


def knn_accuracy(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    accuracy = np.mean(predictions == point_targets)
    return accuracy
    

def knn_confusion_matrix(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> np.ndarray:
    predictions = knn_predict(points, point_targets, classes, k)
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, predicted in zip(point_targets, predictions):
       confusion_matrix[predicted, actual] += 1
        
    return confusion_matrix


def best_k(points: np.ndarray, point_targets: np.ndarray, classes: list) -> int:
    nb_points = len(points)
    accuracies = []

    for k in range(1, nb_points - 1):
        accuracy = knn_accuracy(points, point_targets, classes, k)
        accuracies.append(accuracy)

    best_k_value = np.argmax(accuracies) + 1
    return best_k_value


def knn_plot_points(points: np.ndarray,point_targets: np.ndarray,classes: list,k: int):
  
    colors = ['yellow', 'purple', 'blue']
    edge_colors = ['green', 'red']

    for i in range(len(points)):
       x = points[i]
       y = point_targets[i]
       predicted_class = knn(x, points, point_targets, classes, k)

       edge = edge_colors[0] if predicted_class == y else edge_colors[1]

       [x, y] = points[i,:2]
       plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edge, linewidths=2)
    
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()

    
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
    
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    
    print("\n[+]Part 2.1")
    print(knn_predict(d_test, t_test, classes, 10))
    print(knn_predict(d_test, t_test, classes, 5))
    
    print("\n[+]Part 2.2")
    print(knn_accuracy(d_test, t_test, classes, 10))
    print(knn_accuracy(d_test, t_test, classes, 5))
    
    print("\n[+]Part 2.3")
    print(knn_confusion_matrix(d_test, t_test, classes, 10))
    print(knn_confusion_matrix(d_test, t_test, classes, 20))
    
    print("\n[+]Part 2.4")
    print(best_k(d_train, t_train, classes))
    
    print("\n[+]Part 2.5: plotting...")
    knn_plot_points(d, t, classes, 3)
 

    