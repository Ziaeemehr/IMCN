import numba 
import numpy as np 
from numba import jit 



@jit(nopython=True)
def correlation_matrix(x):

    n = len(x)
    C = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            z0 = np.exp(complex(0, x[i]-x[j]))
            C[i,j] = z0 
    
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
                C[i,j] = C[i,j] + np.exp(complex(0, x[it, i]-x[it, j]))
    
    
    C_real = 1.0/(nstep/step) * np.abs(C)
    
    return C_real

    