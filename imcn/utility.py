import numba
import numpy as np
from numba import jit
import networkx as nx
import random


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
    Cij = 1/nstep * |sum_{t=1}^{nstep} exp(i*(x_i[t]-x_j[t]))| 

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


def display_time(time):
    ''' 
    show real time elapsed
    '''
    hour = time//3600
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print("Done in %d hours %d minutes %.4f seconds"
          % (hour, minute, second))


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def select_random_edges(adj, num_links, directed=True, seed=None):

    if seed is not None:
        random.seed(seed)

    if directed:
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_array(adj, create_using=None)
    edges = list(G.edges())
    links = random.sample(edges, num_links)

    return links
