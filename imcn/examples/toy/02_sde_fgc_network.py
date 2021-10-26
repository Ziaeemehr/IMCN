import warnings
import numpy as np
import pylab as plt
from numpy import pi
from time import time
from jitcsim import plot_order
from multiprocessing import Pool
from numpy.random import uniform
from imcn import calc_TE, calc_MI
from jitcsim.models.kuramoto_sde import Kuramoto_II
from imcn import (time_average_correlation_matrix,
                  select_random_edges, make_graph,
                  plot_degree_omega_distribution)
warnings.filterwarnings("ignore")  # to hide all warnings


# -------------------------------------------------------------------
# -------------------------------------------------------------------

def run_for_each(coupl):
    """
    run the simulation for each coupling
    The initial state and frequencies are changed.

    Parameters
    -----------
    coupl : float
        coupling strength 

    return: dict("g"=coupl, "R"=R, "te"=te, "cor"=cor)
        - g : coupling strength
        - R : time average order parameter
        - te: time average transfer entropy
        - mi: time average mutual information
        - cor: selected elements of time average correlation matrix
    """

    print(f"coupling : {coupl}")

    controls = [coupl]
    I = Kuramoto_II(parameters)
    I.set_initial_state(uniform(-pi, pi, N))
    I.set_integrator_parameters(atol=1e-6, rtol=1e-3)
    data = I.simulate(controls, mode_2pi=False)
    x = data['x']

    corr = time_average_correlation_matrix(x, step=10)
    cor = np.zeros(len(links))
    for i in range(len(links)):
        e = links[i]
        cor[i] = corr[e[0], e[1]]

    te = np.zeros(len(links))
    mi = np.zeros(len(links))

    for i in range(len(links)):
        source_id = links[i][0]
        target_id = links[i][1]
        source = x[:-1, source_id]
        target = np.diff(x[:, target_id])
        if parameters["CALCUALTE_TE"]:
            te[i] = calc_TE(source,
                            target,
                            num_threads=num_threads)
        if parameters["CALCULATE_MI"]:
            mi[i] = calc_MI(source,
                            target,
                            TIME_DIFF=0,
                            NUM_THREADS=num_threads
                            )

    # time average order parameter
    R = np.mean(I.order_parameter(x))

    return {"g": coupl,
            "R": R,
            "te": te,
            "mi": mi,
            "cor": cor}

# -------------------------------------------------------------------
# -------------------------------------------------------------------


def batch_run(couplings, num_ensembles, num_links):
    """
    run simulations in parallel using multiprocessing

    Parameters
    -------------

    couplings : list, array of float
        list of couplings
    num_ensembles: int
        number of ensembles for simulation
    num_links: int
        number of links for calculation of TE and Correlations

    return: dict("R"=R, "TE"=TE, "Cor"=Cor)
        R: array of time average order parameters for each coupling strength
        TE: 2d array of TE for each link and for each coupling strength [num_couplings by num_links]
        Cor: 2d array of correlations for each link and for each coupling strength [num_couplings by num_links]

    """

    par = []
    for i in range(len(couplings)):
        for j in range(num_ensembles):
            par.append(couplings[i])

    with Pool(processes=num_processes) as pool:
        data = (pool.map(run_for_each, par))

    R = [d['R'] for d in data]
    TE = [d['te'] for d in data]
    MI = [d['mi'] for d in data]
    Cor = [d["cor"] for d in data]

    results = {}
    R = np.reshape(R, (len(couplings), num_ensembles))
    R = np.mean(R, axis=1)
    results["R"] = R

    Cor = np.reshape(Cor, (len(couplings), num_ensembles, num_links))
    Cor = np.mean(Cor, axis=1)  # average on ensebles
    results["Cor"] = Cor

    if parameters["CALCUALTE_TE"]:
        TE = np.reshape(TE, (len(couplings), num_ensembles, num_links))
        TE = np.mean(TE, axis=1)  # average on ensebles
        results["TE"] = TE

    if parameters["CALCULATE_MI"]:
        MI = np.reshape(MI, (len(couplings), num_ensembles, num_links))
        MI = np.mean(MI, axis=1)  # average on ensebles
        results["MI"] = MI

    return results


if __name__ == "__main__":

    seed = 2
    np.random.seed(seed)

    # SETTING PARAMETERS --------------------------------------------
    # ---------------------------------------------------------------
    N = 60
    omega = np.random.uniform(low=0, high=1, size=N)
    initial_state = uniform(-pi, pi, N)
    couplings = np.linspace(0.0, 0.05, 6)
    num_links = 1
    noise_amplitude = 0.001
    num_ensembles = 1
    num_processes = 4
    num_threads = 1

    # Frequency Gap-conditioned (FGC) network
    obj = make_graph(seed=seed)
    adj = obj.fgc_network(N=N, k=10, gamma=0.5, omega=omega)
    links = select_random_edges(adj, num_links=num_links, directed=True, seed=seed)
    plot_degree_omega_distribution(adj, omega, "data/omega_k.png")


    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 500,                     # final time of integration
        't_transition': 100.0,              # transition time
        "interval": 0.01,                   # time interval for sampling

        "alpha": 0.0,                       # frustration
        "sigma": noise_amplitude,           # noise amplitude
        "omega": omega,                     # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators
        'control': ["coupling"],            # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
        "CALCUALTE_TE": True,
        "CALCULATE_MI": True
    }

    # compilings ----------------------------------------------------
    # ---------------------------------------------------------------
    sol = Kuramoto_II(parameters)
    sol.compile()

    # running the simulation in parallel ----------------------------
    # ---------------------------------------------------------------
    start_time = time()
    data = batch_run(couplings, num_ensembles, num_links=num_links)
    print("Simulation time: {:.3f} seconds".format(time()-start_time))

    # saving to file ------------------------------------------------
    # ---------------------------------------------------------------
    np.savez("data/data",
             g=couplings,
             R=data['R'],
             Cor=data['Cor'],
             TE=data['TE'],
             MI=data['MI']
             )

    # ploting -------------------------------------------------------
    # ---------------------------------------------------------------
    plt.style.use("ggplot")
    fig, ax = plt.subplots(3, figsize=(8, 10), sharex=True)
    plot_order(couplings,
               data['R'],
               ax=ax[0],
               xlabel="couplings",
               ylabel="R",
               label="R",
               figsize=(8, 5),
               close_fig=False
               )

    Cor = data["Cor"]
    TE = data["TE"]
    MI = data["MI"]

    for i in range(num_links):
        ax[0].plot(couplings, Cor[:, i], marker="s")

    for i in range(num_links):
        ax[1].plot(couplings, TE[:, i], marker="o", )

    for i in range(num_links):
        ax[2].plot(couplings, MI[:, i], marker="*")

    # for i in range(3):
    #     ax[i].legend(frameon=False)
    ax[-1].set_xlabel("coupling")
    ax[1].set_ylabel("TE", fontsize=14)
    ax[2].set_ylabel("MI", fontsize=14)

    plt.tight_layout()
    plt.savefig("data/fgc_fig.png", dpi=150)
    plt.close()
