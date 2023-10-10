import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(X: np.ndarray, Mu: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    n = X.shape[0]
    k = Mu.shape[0]
    distances = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            distances[i, j] = np.linalg.norm(X[i] - Mu[j])

    return distances


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    n, k = dist.shape
    r = np.zeros((n, k), dtype=int)
    closest_prototypes = np.argmin(dist, axis=1)
    for i in range(n):
        r[i, closest_prototypes[i]] = 1

    return r


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    n, k = R.shape
    J = 0.0
    for i in range(n):
        for j in range(k):
            J += R[i, j] * dist[i, j]
    J /= n

    return J


def update_Mu(Mu: np.ndarray, X: np.ndarray, R: np.ndarray) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    k, d = Mu.shape
    new_Mu = np.zeros((k, d))
    for i in range(k):
        numerator = np.sum(R[:, i].reshape(-1, 1) * X, axis=0)
        denominator = np.sum(R[:, i])
        new_Mu[i] = numerator / denominator

    return new_Mu


def k_means(X: np.ndarray, k: int, num_its: int) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    js = []
    for iteration in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        r = determine_r(dist)
        J = determine_j(r, dist)
        
        js.append(J)
        Mu = update_Mu(Mu, X_standard, r)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, r, js


def _plot_j():
    X, y, c = load_iris()
    _, _, js = k_means(X, 4, 10)
    plt.plot(js)
    plt.show()


def _plot_multi_j():
    k_values = [2, 3, 5, 10]
    counter = 1
    for i, k in enumerate(k_values):
        _, _, js = k_means(X, k, num_its=10)
        plt.subplot(2, 2, counter)
        plt.title(f'k={k}')
        plt.plot(js)
        counter+=1
    plt.tight_layout()
    plt.show()


def k_means_predict(X: np.ndarray, t: np.ndarray, classes: list, num_its: int) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, r, _ = k_means(X, len(classes), num_its)
    cluster_targets = np.argmax(r, axis=1)

    cluster_class = {}
    for class_label in classes:
        number_cluster = np.bincount(cluster_targets[t == class_label])
        most_common_cluster = np.argmax(number_cluster)
        cluster_class[most_common_cluster] = class_label

    k_predictions = []
    for cluster in cluster_targets:
        prediction = cluster_class.get(cluster, None)
        if prediction is not None:
            k_predictions.append(prediction)
        
    return np.array(k_predictions)



def _iris_kmeans_accuracy():
    X, y, _ = load_iris()
    class_predictions = k_means_predict(X, y, np.unique(y), 5)
    accuracy = accuracy_score(y, class_predictions)
    confusion = confusion_matrix(y, class_predictions)
    return accuracy, confusion


def _my_kmeans_on_image():
    image, (w, h) = image_to_numpy()
    print(k_means(image, 7, 5))



def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    # Load the image and convert it to a compatible numpy array
    image, _ = image_to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5, max_iter=100)
    cluster_labels = kmeans.fit_predict(image)

    print(cluster_labels)

    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(image)
    plot_gmm_results(image, cluster_labels, gmm.means_, gmm.covariances_)
    

if __name__ == '__main__':
    
    print("[+]Part 1.1")
    a = np.array([
        [1, 0, 0],
        [4, 4, 4],
        [2, 2, 2]])
    b = np.array([
        [0, 0, 0],
        [4, 4, 4]])
    print(distance_matrix(a, b))
    
    print("\n[+]Part 1.2")
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    print(determine_r(dist))
    
    print("\n[+]Part 1.3")
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    R = determine_r(dist)
    print(determine_j(R, dist))
    
    print("\n[+]Part 1.4")
    X = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]])
    Mu = np.array([
        [0.0, 0.5, 0.1],
        [0.8, 0.2, 0.3]])
    R = np.array([
        [1, 0],
        [0, 1],
        [1, 0]])
    print(update_Mu(Mu, X, R))
    
    print("\n[+]Part 1.5")
    X, y, c = load_iris()
    print(k_means(X, 4, 10))
    
    print("\n[+]Part 1.6: Plotting ...")
    _plot_j()
    
    print("\n[+]Part 1.7: Plotting ...")
    _plot_multi_j()
    
    print("\n[+]Part 1.9")
    X, y, c = load_iris()
    print(k_means_predict(X, y, c, 5))
    
    print("\n[+]Part 1.10")
    accuracy, confusion = _iris_kmeans_accuracy()
    print(accuracy)
    print(confusion)
      
    print("\n[+]Part 2.1")
    #print(_my_kmeans_on_image())
    
    print("\n[+]Part 2.1.1")
    num_clusters = [2, 5, 10, 20]
    for num in num_clusters:
        plot_image_clusters(num)
    