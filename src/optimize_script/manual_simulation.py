import time
import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8

import warnings

# Silence pandas during training
warnings.simplefilter(action='ignore', category=FutureWarning)

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm

from optimize_script.generate_simulation import simulate
from optimize_script.draw_test import draw_figure
import matplotlib.pyplot as plt


def load_mirko_param():
    q_gamma = 1e3  # 448.90170243967316
    control_param = {
        'q_bis': 1,
        'q_gamma_propofol': q_gamma,  # 27
        'q_gamma_remi': q_gamma,
        'r_induction': 100,  # 37.92109188569586,
        'r_maintenance': 10,  # 14.419328154245125,
        'q_ratio': 0,
        'N': 32,
    }
    return control_param


def load_range_param():
    control_param = {
        'q_bis': 1,
        'q_range': 0,
        'r_induction': 200,
        'r_maintenance': 96,  # 96 // 6,
        'q_ratio': 0,
        'N': 30,
        'k1': 0,
        'k2': 0,
    }
    return control_param


def main():

    filename = "manual_test_static_gain_estimation"
    category = "manual"
    root_folder = "./data/{}".format(category)
    os.makedirs(root_folder, exist_ok=True)

    np.random.seed(2)
    # restored_sampler = pickle.load(open("trash/np_sampler.pkl", "rb"))
    # np.random.set_state(restored_sampler)

    np.set_printoptions(formatter={'all': np.format_float_scientific})

    # run the best parameter on the test set
    """
    Hyper parameters

    """
    Te = 1  # Sampling time
    control_sampling_time = 5

    """    
    Definition of the patient

    """
    age = 53
    height = 164
    weight = 97
    gender = 1

    """
    Controller definition

    """
    # control_param = load_mirko_param()
    control_param = load_range_param()

    cost = "range"

    model_param = {'lambda': 1}

    """
    Simulation
    
    """

    start = time.time()

    results = simulate(
        0, control_ts=control_sampling_time, Te=Te, N_mhe=15, Patient_info=[age, height, weight, gender], control_param=control_param, model_param=model_param, model="Eleveld", cost=cost
    )
    results.insert(0, 'caseid', 0)  # for compatibility purpose with the script draw_test

    print(f"Simulation time: {time.time() - start:.2f} s")
    # save the result
    print("Saving results...")
    os.makedirs("{}/signals".format(root_folder), exist_ok=True)
    results.to_csv("{}/signals/{}".format(root_folder, filename), quoting=csv.QUOTE_ALL)

    print("Done!")

    draw_figure({'test': results})
    plt.show()


if __name__ == '__main__':
    main()
