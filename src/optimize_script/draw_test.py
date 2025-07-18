import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


def draw_figure_separate(results_dict) -> tuple:
    # Draw bis
    _, ax_bis = plt.subplots(1)

    for key in results_dict.keys():
        results = results_dict[key]
        ax_bis.step(results['Time'] / 60, results['BIS'])  # , label="bis_{}".format(key))

    ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'], linewidth=1, label="center")
    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] - 10, 'r--', linewidth=1.5, label="safe range")
    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] + 10, 'r--', linewidth=1.5)

    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] - 8, 'g--', linewidth=1.5, label="mpc range")
    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] + 8, 'g--', linewidth=1.5)

    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] - 5, 'b--', linewidth=1.5, label="equilibrium range")
    # ax_bis.plot(results['Time'] / 60, results['BIS_REFERENCE'] + 5, 'b--', linewidth=1.5)

    # ax_bis.legend()
    ax_bis.set_xlabel("Time (min)")
    ax_bis.set_ylabel("BIS")
    ax_bis.grid(True, color='black', linewidth=1)
    ax_bis.axis(ymin=0, ymax=100, xmin=0, xmax=20)

    # Draw control
    _, ax_control = plt.subplots(2)
    u_name = ["Propofol rate", "Remifentanil rate"]

    for key in results_dict.keys():
        results = results_dict[key]
        inputs_array = np.reshape(np.concatenate(results['inputs'].explode().to_numpy()), (-1, 2))
        for i in range(len(ax_control)):
            ax_control[i].step(results['Time'] / 60, inputs_array[:, i])  # , label="{}-{}".format(key, u_name[i]))
            # ax_control[i].legend()
            ax_control[i].set_xlabel("Time (min)")
            ax_control[i].set_ylabel("{}".format(u_name[i]))
            ax_control[i].grid(True, color='black', linewidth=1)
            ax_control[i].axis(ymin=0, ymax=2.7, xmin=0, xmax=40)  # 2.7 or 3.5

    # Draw state_x
    _, ax_state = plt.subplots(6, 2)
    state_name = ["x1r", "x2r", "x3r", "Cer", "x1p", "x2p", "x3p", "Cep", "C50p", "C50r", "gamma", "Disturbance"]

    for key in results_dict.keys():
        results = results_dict[key]
        states_array = np.reshape(results['x'].explode().to_numpy(), (-1, 12))
        for i in range(ax_state.shape[0]):
            for j in range(ax_state.shape[1]):
                ax_state[i, j].step(
                    results['Time'] / 60,
                    states_array[:, ax_state.shape[1] * i + j],
                    label="{}.{}".format(key, state_name[ax_state.shape[1] * i + j]),
                )
                ax_state[i, j].set_xlabel("Time (min)")
                ax_state[i, j].grid(True, color='black', linewidth=1)
                ax_state[i, j].legend()


def draw_figure(results_dict, time_limit) -> tuple:
    # Draw bis
    _, ax = plt.subplots(3, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=[6.4, 7.5])

    for key in results_dict.keys():
        results = results_dict[key]
        ax[0].step(results['Time'] / 60, results['BIS'])  # , label="bis_{}".format(key))

    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'], linewidth=1, label="center")
    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] - 10, 'r--', linewidth=1.5, label="safe range")
    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] + 10, 'r--', linewidth=1.5)

    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] - 8, 'g--', linewidth=1.5, label="mpc range")
    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] + 8, 'g--', linewidth=1.5)

    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] - 5, 'b--', linewidth=1.5, label="equilibrium range")
    ax[0].plot(results['Time'] / 60, results['BIS_REFERENCE'] + 5, 'b--', linewidth=1.5)

    ax[0].legend()
    ax[0].set_ylabel("BIS")
    ax[0].grid(True, color='black', linewidth=1)
    ax[0].axis(ymin=20, ymax=100, xmin=0, xmax=time_limit)

    # Draw control
    u_name = ["Propofol rate", "Remifentanil rate"]

    for key in results_dict.keys():
        results = results_dict[key]
        inputs_array = np.reshape(np.concatenate(results['inputs'].explode().to_numpy()), (-1, 2))
        for i in range(len(ax) - 1):
            ax[i + 1].step(results['Time'] / 60, inputs_array[:, i])  # , label="{}-{}".format(key, u_name[i]))
            # ax_control[i].legend()
            ax[i + 1].set_ylabel("{}".format(u_name[i]))
            ax[i + 1].grid(True, color='black', linewidth=1)
            ax[i + 1].axis(ymin=0, ymax=3.5, xmin=0, xmax=time_limit)  # 2.7 or 3.5
        ax[i + 1].set_xlabel("Time (min)")


def get_csv_results(control_type, study_name):
    root_folder = "./data/{}".format(control_type)

    results = pd.read_csv("{}/signals/{}.csv".format(root_folder, study_name))

    # CSV formatting problems
    results.inputs = results.inputs.apply(lambda x: x.replace('\n', ','))
    results.inputs = results.inputs.apply(ast.literal_eval)

    results.x = results.x.apply(lambda x: x.replace(' ', ','))
    results.x = results.x.apply(lambda x: x.replace('\n', ''))
    results.x = results.x.apply(ast.literal_eval)

    return results


def get_result_by_patient(results):
    by_group = results.groupby('caseid')
    dict_result = {}

    for group in by_group.groups:

        dict_result[group] = by_group.get_group(group)
        # break

    return dict_result


def main():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option("max_colwidth", 200)

    plt.rcParams.update({"text.usetex": True})

    plt.rc('font', size=14)  # , weight='bold')
    # plt.rc('axes', labelweight='bold')

    # last study range
    control_type = 'range'
    study_name = 'paper_final_range_realistic'

    # last study nominal
    # control_type = 'nominal'
    # study_name = 'paper_final_nominal_step'

    results = get_csv_results(control_type, study_name)
    dict_result = get_result_by_patient(results)
    draw_figure(dict_result, 60)
    plt.show()


if __name__ == '__main__':
    main()
