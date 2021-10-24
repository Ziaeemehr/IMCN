import numpy as np
import numpy as np
from numpy import pi
from time import time
from multiprocessing import Pool
from numpy.random import uniform, normal
from jitcsim.models.kuramoto import Kuramoto_II
from jitcsim.visualization import plot_order
# from jitcsim.networks import make_network


def run_for_each(coupl):
    """
    run the simulation for each coupling
    The initial state and frequencies are changed.
    """

    controls = [coupl]
    I = Kuramoto_II(parameters)
    I.set_initial_state(uniform(-pi, pi, N))
    data = I.simulate(controls)
    x = data['x']
    order = np.mean(I.order_parameter(x))

    return order


if __name__ == "__main__":

    np.random.seed(2)

    N = 3
    omega = [0.3, 0.4, 0.5]
    initial_state = uniform(-pi, pi, N)
    num_ensembles = 5
    num_processes = 4

    adj = np.asarray([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])

    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 500,                    # final time of integration
        't_transition': 100.0,              # transition time
        "interval": 0.1,                   # time interval for sampling

        # "coupling": coupling0,              # coupling strength
        "alpha": 0.0,                       # frustration
        "omega": omega,                     # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'integration_method': 'dopri5',     # integration method
        'control': ["coupling"],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = Kuramoto_II(parameters)
    sol.compile()

    start_time = time()
    couplings = np.linspace(0, 0.4, 20)
    par = []
    for i in range(len(couplings)):
        for j in range(num_ensembles):
            par.append(couplings[i])

    with Pool(processes=num_processes) as pool:
        orders = (pool.map(run_for_each, par))
    orders = np.reshape(orders, (len(couplings), num_ensembles))

    print("Simulation time: {:.3f} seconds".format(time()-start_time))
    plot_order(couplings, np.mean(orders, axis=1),
               filename="data/R.png",
               xlabel="couplings",
               ylabel="R",
               figsize=(8, 5)
               )

    
    
    # R = np.zeros_like(couplings)

    # for i in range(len(couplings)):
    #     for j in range(num_ensembles):
    #         controls = [couplings[i]]
    #         data = sol.simulate(controls)
    #         x = data['x']
    #         t = data['t']
    #         R[i] = np.mean(sol.order_parameter(x))

    # plot_order(t,
    #            order,
    #            filename="data/00.png",
    #            xlabel="time",
    #            ylabel="r(t)")
