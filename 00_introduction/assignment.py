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

    
"""
def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1

def _compare_components_and_mixture():
    # Part 2.2

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

def _plot_mixture_and_samples():
    # Part 3.2
"""

if __name__ == '__main__':
    
    #Test 1.1
    print(normal(0, 1, 0))
    print(normal(3, 1, 5))
    print(normal(np.array([-1,0,1]), 1, 0))
    
    #Tests 1.2
    plot_normal(0.5, 0, -2, 2)
    _plot_three_normals()