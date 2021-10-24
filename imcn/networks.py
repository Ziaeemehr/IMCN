import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import networkx as nx
from copy import copy
from sys import exit


class make_graph:
    ''' 
    make different graphs ans return their adjacency matrices
    '''

    def __init__(self, seed=None):
        self.G = 0

        if seed:
            np.random.seed(seed)

    #---------------------------------------------------------------#
    def complete_graph(self, N):
        ''' 
        returns all to all graph 
        '''

        self.N = N
        self.G = nx.complete_graph(N)

        return nx.to_numpy_array(self.G, dtype=int)

    #---------------------------------------------------------------#
    def erdos_renyi_graph(self, N, p, seed=None, directed=False):
        ''' 
        returns Erdos Renyi graph 
        '''

        self.N = N
        self.G = nx.erdos_renyi_graph(N, p, seed=seed,
                                      directed=directed)

        return nx.to_numpy_array(self.G, dtype=int)
    #---------------------------------------------------------------#

    def barabasi(self, N, m, seed=None):
        ''' 
        returns BA graph 
        '''

        self.N = N
        self.G = nx.barabasi_albert_graph(N, m, seed=seed)

        return nx.to_numpy_array(self.G, dtype=int)
    #---------------------------------------------------------------#

    def fgc_network(self,
                    N,
                    omega,
                    k,
                    gamma=0.4):
        """
        Frequency Gap-conditioned (FGC) network
        @param N: the number of oscillators in the system (int)
        @param k: degree of the network (int)
        @param gamma: minimal frequency gap (float)
        """

        # the number of links in the network
        L = N*k//2

        # the natural frequencies follow a uniform distribution
        # omega = np.random.uniform(low=minOmega, high=maxOmega, size=N)

        # initialize the adjacency matrix
        A = np.zeros((N, N), dtype=int)

        # construct FGC random network
        counter = 0
        while counter < L:
            i, j = np.random.choice(range(N), size=2, replace=False)
            if (abs(omega[i]-omega[j]) > gamma) and (A[i][j] == 0):
                A[i][j] = 1
                counter += 1
        A = list(np.asarray(A).reshape(-1))
        A = [a.item() for a in A]

        return A
        #---------------------------------------------------------------#
