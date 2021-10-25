import warnings
import numpy as np
import pylab as plt
from numpy import pi
from time import time
from imcn import calc_TE
from jitcsim import plot_order
from multiprocessing import Pool
from numpy.random import uniform, normal
from jitcsim.models.kuramoto_sde import Kuramoto_II
from imcn import time_average_correlation_matrix
warnings.filterwarnings("ignore") # to hide all warnings


def run_for_each(coupl):
    """
    run the simulation for each coupling
    The initial state and frequencies are changed.
    """

    print(f"coupling : {coupl}")

    controls = [coupl]
    I = Kuramoto_II(parameters)
    I.set_initial_state(uniform(-pi, pi, N))
    I.set_integrator_parameters(atol=1e-6, rtol=1e-3)
    data = I.simulate(controls, mode_2pi=False)
    x = data['x']

    links = [(0, 1), (0, 2), (1, 2)]
    corr = time_average_correlation_matrix(x, step=10)
    cor = np.zeros(len(links))
    for i in range(len(links)):
        e = links[i]
        cor[i] = corr[e[0], e[1]]

    te = np.zeros(len(links))

    for i in range(len(links)):
        source_id = links[i][0]
        target_id = links[i][1]
        source = x[:-1, source_id]
        target = np.diff(x[:, target_id])
        te[i] = calc_TE(source, target, num_threads=num_threads)

    # # time average order parameter
    R = np.mean(I.order_parameter(x))

    return {"g": coupl, "R": R, "te": te, "cor": cor}


def batch_run(couplings, num_ensembles, num_links):

    par = []
    for i in range(len(couplings)):
        for j in range(num_ensembles):
            par.append(couplings[i])

    with Pool(processes=num_processes) as pool:
        data = (pool.map(run_for_each, par))

    R = [d['R'] for d in data]
    TE = [d['te'] for d in data]
    Cor = [d["cor"] for d in data]

    R = np.reshape(R, (len(couplings), num_ensembles))
    R = np.mean(R, axis=1)

    TE = np.reshape(TE, (len(couplings), num_ensembles, num_links))
    TE = np.sum(TE, axis=1)  # average on ensebles

    Cor = np.reshape(Cor, (len(couplings), num_ensembles, num_links))
    Cor = np.mean(Cor, axis=1)

    return {"R": R, "TE": TE, "Cor": Cor}


if __name__ == "__main__":

    np.random.seed(2)

    N = 3
    omega = [0.3, 0.4, 0.5]
    initial_state = uniform(-pi, pi, N)
    noise_amplitude = 0.001
    num_ensembles = 2
    num_processes = 4
    num_threads = 1

    adj = np.asarray([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 500,                    # final time of integration
        't_transition': 100.0,              # transition time
        "interval": 0.01,                   # time interval for sampling

        # "coupling": coupling0,              # coupling strength
        "alpha": 0.0,                       # frustration
        "sigma": noise_amplitude,
        "omega": omega,                     # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'integration_method': 'RK23',     # integration method
        'control': ["coupling"],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = Kuramoto_II(parameters)
    sol.compile()

    start_time = time()
    couplings = np.linspace(0, 0.4, 21)
    data = batch_run(couplings, num_ensembles, num_links=3)

    print("Simulation time: {:.3f} seconds".format(time()-start_time))

    fig, ax = plt.subplots(2, figsize=(8, 10), sharex=True)

    plot_order(couplings,
               data['R'],
               ax=ax[0],
               xlabel="couplings",
               ylabel="R",
               label="R",
               figsize=(8, 5),
               close_fig=False
               )

    labels = ["1->2", "1->3", "2->3"]

    Cor = data['Cor']
    TE = data['TE']

    for i in range(3):
        ax[0].plot(couplings, Cor[:, i], marker="s", label=labels[i])

    for i in range(3):
        ax[1].plot(couplings, TE[:, i], marker="o", label=labels[i])

    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    ax[1].set_xlabel("coupling")

    plt.tight_layout()
    plt.savefig("data/FIG.png", dpi=150)
    plt.close()
