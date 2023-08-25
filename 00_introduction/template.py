import numpy as np
import matplotlib.pyplot as plt

# Part 1.1
def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    elem1 = 1/(np.power((2*np.pi*np.power(sigma, 2)), 0.5))
    beforeExp = -np.power((x - mu), 2)/(2*np.power(sigma, 2))
    elem2 = np.exp(beforeExp)
    return (elem1*elem2)
    
    
# Part 1.2
def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    plt.clf()    
    x_range = np.linspace(x_end, x_start, 500)
    
    elem1 = 1/(np.power((2*np.pi*np.power(sigma, 2)), 0.5))
    beforeExp = -np.power((x_range - mu), 2)/(2*np.power(sigma, 2))
    elem2 = np.exp(beforeExp)
        
    plt.plot(x_range, (elem1*elem2))


# Part 1.2
def _plot_three_normals():
    def plot_normal2(sigma: float, mu: float, x_range):   
        elem1 = 1 / (np.power((2 * np.pi * np.power(sigma, 2)), 0.5))
        beforeExp = -np.power((x_range - mu), 2) / (2 * np.power(sigma, 2))
        elem2 = np.exp(beforeExp)
            
        plt.plot(x_range, elem1 * elem2, label=f'Sigma {sigma} Mu {mu}')
        
    plt.clf()  # Clear the current figure if there is one

    x_range = np.linspace(-5, 5, 500)
    plot_normal2(0.5, 0, x_range)
    plot_normal2(0.25, 1, x_range)
    plot_normal2(1, 1.5, x_range)

    plt.legend()
    plt.title('Three Normal Distribution Curves')
    plt.show()

# Part 2.1
def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    sum = 0
    for i in range(len(weights)):
        elem1 = weights[i] / (np.sqrt(2 * np.pi * np.power(sigmas[i], 2)))
        toExp = -(np.power((x - mus[i]), 2)) / (2 * np.power(sigmas[i], 2))
        elem2 = np.exp(toExp)
        sum += elem1*elem2
    return sum
    
    
# Part 2.2
def _compare_components_and_mixture():
    
    def plot_normal2(sigma: float, mu: float, x_range):   
        elem1 = 1 / (np.power((2 * np.pi * np.power(sigma, 2)), 0.5))
        beforeExp = -np.power((x_range - mu), 2) / (2 * np.power(sigma, 2))
        elem2 = np.exp(beforeExp)
            
        plt.plot(x_range, elem1 * elem2, label=f'Sigma {sigma} Mu {mu}')
    
    
    def plot_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
        sum = 0
        for i in range(len(weights)):
            elem1 = weights[i] / (np.sqrt(2 * np.pi * np.power(sigmas[i], 2)))
            toExp = -(np.power((x - mus[i]), 2)) / (2 * np.power(sigmas[i], 2))
            elem2 = np.exp(toExp)
            sum += elem1*elem2
        plt.plot(x_range, sum, label='Mixture')
       
    
    plt.clf()  # Clear the current figure if there is one
    
    x_range = np.linspace(-5, 5, 500)
    plot_normal2(0.5, 0, x_range)
    plot_normal2(1.5, -0.5, x_range)
    plot_normal2(0.25, 1.5, x_range)
    plot_mixture(x_range, [0.5, 1.5, 0.25], [0, -0.5, 1.5], [1/3, 1/3, 1/3])

    plt.legend()
    plt.title('Three Normal Distribution Curves + mixture')
    plt.show()
    
   
# Part 3.1
def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    np.random.seed(0)
    nb_normal_components = np.random.multinomial(n_samples, weights)
    allRandoms = []
    for i in range(len(nb_normal_components)):
        for j in range(nb_normal_components[i]):
            random_normal = np.random.normal(mus[i], sigmas[i])
            allRandoms.append(random_normal)
    randoms_array = np.array(allRandoms)
    return randoms_array
    

# Part 3.2   
def _plot_mixture_and_samples():
    def plot_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
        sum = 0
        for i in range(len(weights)):
            elem1 = weights[i] / (np.sqrt(2 * np.pi * np.power(sigmas[i], 2)))
            toExp = -(np.power((x - mus[i]), 2)) / (2 * np.power(sigmas[i], 2))
            elem2 = np.exp(toExp)
            sum += elem1*elem2
        plt.plot(x_range, sum, label='Mixture')
    sigmas = [0.3, 0.5, 1]
    mus = [0, -1, 1.5]
    weights = [0.2, 0.3, 0.5]
    x_range = np.linspace(-5, 5, 10)
    
    plt.figure(figsize=(16, 6))
    plt.subplot(141)
    plt.hist(sample_gaussian_mixture(sigmas, mus, weights, 10), density=True)
    plot_mixture(np.linspace(-2, 3, 10), sigmas, mus, weights)    #plt.plot(normal_mixture(np.linspace(-2, 3, 10), sigmas, mus, weights))
    
    plt.subplot(142)
    plt.hist(sample_gaussian_mixture(sigmas, mus, weights, 100), 100, density=True)
    plot_mixture(np.linspace(-2, 3, 10), sigmas, mus, weights)
    
    plt.subplot(143)
    plt.hist(sample_gaussian_mixture(sigmas, mus, weights, 500), 100, density=True)
    plot_mixture(np.linspace(-2, 3, 10), sigmas, mus, weights)
    
    plt.subplot(144)
    plt.hist(sample_gaussian_mixture(sigmas, mus, weights, 1000), 100, density=True)
    plot_mixture(np.linspace(-2, 3, 10), sigmas, mus, weights)


if __name__ == '__main__':
    
    #Test 1.1
    print("[+]Results part 1.1")
    print(normal(0, 1, 0))
    print(normal(3, 1, 5))
    print(normal(np.array([-1,0,1]), 1, 0))
    
    #Tests 1.2
    print("\n[+]Results part 1.2 : plotting...")
    plot_normal(0.5, 0, -2, 2)
    _plot_three_normals()
    
    #Tests 2.1
    print("\n[+]Results part 2.1")
    print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))
    print(normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1]))
    
    #Test 2.2
    print("[+]Results part 2.2")
    _compare_components_and_mixture()
    
    #Test 3.1
    print("\n[+]Results part 3.1")
    print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3))
    print(sample_gaussian_mixture([0.1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10))
    
    #Test 3.2
    print("\n[+]Results part 3.2 : plotting...")
    _plot_mixture_and_samples()