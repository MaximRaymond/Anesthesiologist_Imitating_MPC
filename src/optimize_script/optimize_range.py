from functools import partial
import multiprocessing as mp
import time
import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings

# Silence pandas during training
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from optimize_script.generate_simulation import generate_simulation


def main():
    # define the parameter of the study
    control_type = 'range'
    root_folder = "./data/{}".format(control_type)
    os.makedirs(root_folder, exist_ok=True)

    test_patient_number = 8

    np.random.seed(2)

    # normalize formatting of number for csv saving
    np.set_printoptions(formatter={'all': np.format_float_scientific})

    study_name = 'paper_final_range_realistic'

    dict = {
        'control_type': control_type,
        'filename': "{}.csv".format(study_name),
    }

    control_param = {
        'q_bis': 1,
        'q_range': 0,
        'r_induction': 200,
        'r_maintenance': 96,
        'N': 120,
    }

    start = time.time()
    test_func = partial(generate_simulation, control_param=control_param, cost=control_type)
    patient_list = np.arange(test_patient_number) + 4000
    nb_cpu = min(mp.cpu_count(), test_patient_number)
    with mp.Pool(nb_cpu) as p:
        res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Study Range'))

    print(f"Simulation time: {time.time() - start:.2f} s")
    # save the results
    print("Saving results...")
    final_df = pd.concat(res)
    os.makedirs("{}/signals".format(root_folder), exist_ok=True)
    final_df.to_csv("{}/signals/{}".format(root_folder, dict['filename']), quoting=csv.QUOTE_ALL)

    print("Done!")


if __name__ == '__main__':
    main()
