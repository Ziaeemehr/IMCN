import numpy as np
import pylab as plt
import networkx as nx


def plot_degree_omega_distribution(adj, omega,
                                   file_name="omega_k.png"):
    """
    plot omega distribution 
    """

    adj = np.asarray(adj)
    assert (len(adj.shape) == 2)

    G = nx.from_numpy_array(adj)

    N = adj.shape[0]
    omegaNeighborsBar = [0] * N
    omegaBar = omega - np.mean(omega)

    degrees = list(dict(G.degree()).values())

    for i in range(N):
        neighbors = [n for n in G[i]]
        omegaNeighborsBar[i] = np.mean(omegaBar[neighbors])

    fig, ax = plt.subplots(2, figsize=(8, 5))

    ax[0].plot(omegaBar, degrees, "ro")
    ax[1].plot(omegaBar, omegaNeighborsBar, "ko")

    ax[0].set_ylabel(r"$k_i$", fontsize=16)
    ax[1].set_ylabel(r"$\langle \omega_j \rangle$", fontsize=16)
    ax[1].set_xlabel(r"$\omega_i$", fontsize=16)

    for i in range(2):
        ax[i].tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
