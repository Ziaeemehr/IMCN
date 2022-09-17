import time
import numba
import random
import numpy as np
from numba import jit
import networkx as nx


@jit(nopython=True)
def correlation_matrix(x):

    n = len(x)
    C = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            z0 = np.exp(complex(0, x[i]-x[j]))
            C[i, j] = z0

    return C


@jit(nopython=False)
def time_average_correlation_matrix(x, step=1):
    """

    .. math::
        Cij = \\frac{1}{nstep} | \sum_{t=1}^{nstep} \exp^{i(x_i[t]-x_j[t])}| 

    Parameters
    ---------------
    x : numpy.ndarray
        [nstep by n] time course of coordinates of nodes
    step: int
        calculate correlations every 'step'

    """

    nstep, n = x.shape
    C = np.zeros((n, n), dtype=complex)
    C_real = np.zeros_like(C)

    for it in range(0, nstep, step):
        for i in range(n):
            for j in range(i, n):
                C[i, j] = C[i, j] + np.exp(complex(0, x[it, i]-x[it, j]))

    C_real = 1.0/(nstep/step) * np.abs(C)

    return C_real


def display_time(time, message=""):
    ''' 
    show real time elapsed

    Parameters
    ---------------
    time : float
        time in seconds
    '''
    hour = int(time/3600)
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print(f"{message:s} Done in {hour:d} hours {minute:d} minutes {second:.4f} seconds")


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    '''
    check if a matrix is symmetric

    Parameters
    ---------------
    a : numpy.ndarray
        matrix to be checked
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    '''
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def select_random_edges(adj, num_links, directed=True, seed=None):
    '''
    select random edges from a graph

    Parameters
    ---------------
    adj : numpy.ndarray
        adjacency matrix of the graph
    num_links : int
        number of links to be selected
    directed : bool
        if True, the graph is directed
    seed : int
        random seed

    Returns
    ---------------
    list of tuples

    '''

    if seed is not None:
        random.seed(seed)

    if directed:
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_array(adj, create_using=None)
    edges = list(G.edges())
    links = random.sample(edges, num_links)

    return links


def timer(func):
    '''
    decorator to measure elapsed time
    Parameters
    -----------
    func: function
        function to be decorated
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end-start, message="{:s}".format(func.__name__))
        return result
    return wrapper
