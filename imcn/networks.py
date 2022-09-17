import numpy as np
from sys import exit
import networkx as nx
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
        returns adjacency matrix of a complete graph 

        Parameters
        ---------------
        N : int
            number of nodes

        Returns
        ---------------
        numpy.ndarray

        '''

        self.N = N
        self.G = nx.complete_graph(N)

        return nx.to_numpy_array(self.G, dtype=int)

    #---------------------------------------------------------------#
    def erdos_renyi_graph(self, N, p, directed=False):
        ''' 
        returns Erdos Renyi graph 

        Parameters
        ---------------
        N : int
            number of nodes
        p : float
            Probability for edge creation.
        directed : bool, optional (default=False)
            if True, the graph is directed

        Returns
        ---------------
        numpy.ndarray
        '''

        self.N = N
        self.G = nx.erdos_renyi_graph(N, p, seed=self.seed,
                                      directed=directed)

        return nx.to_numpy_array(self.G, dtype=int)
    #---------------------------------------------------------------#

    def barabasi(self, N, m):
        ''' 
        returns adjacency matrix of a Barabasi-Albert graph

        Parameters
        ---------------
        N : int
            number of nodes
        m : int
            number of edges to attach from a new node to existing nodes

        Returns
        ---------------
        numpy.ndarray

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
        return adjacency matrix of a Frequency Gap-conditioned (FGC) network

        Parameters
        ---------------
        N : int
            number of nodes
        omega : float
            frequency gap
        k : int
            average degree of the network
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
