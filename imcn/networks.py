import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import networkx as nx
from copy import copy
from sys import exit
from imcn.utility import is_symmetric


class make_graph:
    ''' 
    make different graphs ans return their adjacency matrices
    '''

    def __init__(self, seed=None):
        self.G = 0
        self.seed = seed

        if seed is not None:
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
    def erdos_renyi_graph(self, N, p, directed=False):
        ''' 
        returns Erdos Renyi graph 
        '''

        self.N = N
        self.G = nx.erdos_renyi_graph(N, p, seed=self.seed,
                                      directed=directed)

        return nx.to_numpy_array(self.G, dtype=int)
    #---------------------------------------------------------------#

    def barabasi(self, N, m):
        ''' 
        returns BA graph 
        '''

        self.N = N
        self.G = nx.barabasi_albert_graph(N, m, seed=self.seed)

        return nx.to_numpy_array(self.G, dtype=int)
    #---------------------------------------------------------------#

    def fgc_network(self,
                    N,
                    omega,
                    k,
                    gamma=0.4,
                    verbose=False):
        """
        Frequency Gap-conditioned (FGC) network
        @param N: the number of oscillators in the system (int)
        @param k: degree of the network (int)
        @param gamma: minimal frequency gap (float)
        """

        # the number of links in the network
        L = N*k//2

        # initialize the adjacency matrix
        A = np.zeros((N, N), dtype=int)
        node_list = list(range(N))

        # construct FGC random network
        counter = 0
        num_trial = 0
        while counter < L:
            i, j = np.random.choice(node_list, size=2, replace=False)
            if ((abs(omega[i]-omega[j]) > gamma) and
                    (A[i][j] == 0) and (A[j][i] == 0)):
                A[i][j] = A[j][i] = 1
                counter += 1
                num_trial = 0

            if (num_trial > 1000):
                print("adding edge stuck!")
                exit(0)

        if verbose:
            G = nx.from_numpy_array(A)
            if nx.is_connected(G):
                print("network is connected.")
            if is_symmetric(A):
                print("network is symmetric.")

        return A
        #---------------------------------------------------------------#
