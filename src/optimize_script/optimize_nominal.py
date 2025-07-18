from functools import partial
import multiprocessing as mp
import time
import sys
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings

# Silence pandas during training
warnings.simplefilter(action='ignore', category=FutureWarning)

from optimize_script.generate_simulation import generate_simulation


def main():
    # define the parameter of the study
    control_type = 'nominal'
    root_folder = "./data/{}".format(control_type)
    os.makedirs(root_folder, exist_ok=True)

    test_patient_number = 8

    np.random.seed(2)
    # normalize formatting of number for csv saving
    np.set_printoptions(formatter={'all': np.format_float_scientific})

    # save the parameter of the study as json file
    study_name = 'paper_final_nominal_step'

    dict = {
        'control_type': control_type,
        'filename': "{}.csv".format(study_name),
    }

    R_matrix = np.diag([4, 1])

    control_param = {
        'R': 200 * R_matrix,
        'R_maintenance': 50 * R_matrix,
        'N': 120,
        'Nu': 120,
    }

    start = time.time()
    test_func = partial(generate_simulation, control_param=control_param, cost="nominal")
    patient_list = np.arange(test_patient_number) + 4000
    nb_cpu = min(mp.cpu_count(), test_patient_number)
    with mp.Pool(nb_cpu) as p:
        # imap return an iterator on the results, map return directly the list of the results
        # tqdm is used to show a progress bar using an iterable, and total to show the number of iteration expected, desc is the prefix of the progress bar
        res = list(tqdm(p.imap(test_func, patient_list), total=len(patient_list), desc='Study nominal'))

    print(f"Simulation time: {time.time() - start:.2f} s")
    # save the result of the test set
    print("Saving results...")
    final_df = pd.concat(res)
    os.makedirs("{}/signals".format(root_folder), exist_ok=True)
    final_df.to_csv("{}/signals/{}".format(root_folder, dict['filename']), quoting=csv.QUOTE_ALL)

    print("Done!")


if __name__ == '__main__':
    main()
