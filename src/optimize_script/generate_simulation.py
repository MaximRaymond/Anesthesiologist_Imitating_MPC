import numpy as np
import pandas as pd
import casadi as ca
from time import perf_counter
import python_anesthesia_simulator as pas

import mpc as myMpc
import estimator_only_disturbance as distEst
from close_loop_anesth.mpc import NMPC_integrator_multi_shooting

use_real_data = False  # use real data no estimation

inter_variability = True  # simulate with uncertainties
noise = True  # simulate with noise

block_u = True  # Don't change u for small variations


def generate_simulation(caseid: int, control_param: dict, cost: str):
    ###########################################
    # Hyper parameters
    ###########################################

    Te = 1  # Sampling time
    control_sampling_time = 5

    ###########################################
    # Definition of the patient
    ###########################################
    np.random.seed(caseid)

    # Generate random patient information with uniform distribution
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    results = simulate(control_ts=control_sampling_time, Te=Te, N_mhe=15, Patient_info=[age, height, weight, gender], control_param=control_param, model="Eleveld", cost=cost)

    results.insert(0, 'caseid', caseid)
    return results


def load_nominal_control_param(control_param, model_param):
    control_param['A'] = model_param['A']
    control_param['B'] = model_param['B']
    control_param['BIS_param'] = model_param['hill_param']
    control_param['bool_u_eq'] = True
    control_param['bool_non_linear'] = True

    return control_param


def load_control_param(control_type: str, model_param, control_param, control_sample_time):

    up_max = 6.67
    ur_max = 16.67
    control_param['ts'] = control_sample_time
    control_param['umax'] = [up_max, ur_max]
    control_param['umin'] = [0, 0]

    if control_type == 'nominal':
        control_param = load_nominal_control_param(control_param, model_param)

    else:
        control_param['cost_type'] = control_type

    return control_param


def load_model(simulator):
    Ap = simulator.propo_pk.continuous_sys.A[:4, :4]
    Bp = simulator.propo_pk.continuous_sys.B[:4]
    Ar = simulator.remi_pk.continuous_sys.A[:4, :4]
    Br = simulator.remi_pk.continuous_sys.B[:4]

    A = np.block([[Ap, np.zeros((4, 4))], [np.zeros((4, 4)), Ar]])
    B = np.block([[Bp, np.zeros((4, 1))], [np.zeros((4, 1)), Br]])

    model_param = {
        'A': A,
        'B': B,
        'hill_param': simulator.hill_param,  # list [C50p_BIS, C50r_BIS, gamma_BIS, beta_BIS, E0_BIS, Emax_BIS]
    }
    return model_param


def load_disturbance_mhe_param(R: float, N_mhe: int, q: float, sampling_time):
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1] + [1e3] * 1) * q
    P = np.diag([1, 550, 550, 1, 1, 50, 750, 1])

    MHE_std = {
        'R': R,
        'Q': Q,
        'P': P,
        'horizon_length': N_mhe,
        'ts': sampling_time,
    }
    return MHE_std


def simulate(control_ts: int, Te: int, N_mhe: int, Patient_info: list, control_param: dict, model: str, cost: str) -> dict:

    ###########################################
    # Retrieve the corresponding model
    ###########################################

    nominal_simulator = pas.Patient(Patient_info, model_propo=model, model_remi=model, ts=Te)

    model_param = load_model(nominal_simulator)

    real_simulator = pas.Patient(Patient_info, model_propo=model, model_remi=model, ts=Te, random_PD=inter_variability, random_PK=inter_variability)

    ###########################################
    # Define controller
    ###########################################

    control_param = load_control_param(cost, model_param, control_param, control_ts)

    if cost == 'nominal':
        if 'R_maintenance' in control_param.keys():
            R_maintenance = control_param['R_maintenance']
            control_param.pop('R_maintenance')
        else:
            R_maintenance = None

        R_mpc = control_param['R']
        controller = NMPC_integrator_multi_shooting(**control_param)
    else:
        controller = myMpc.NMPC_extended_single_shooting(model_param, control_param)

    ###########################################
    # Define estimator
    ###########################################

    mhe_param = load_disturbance_mhe_param(R=0.0002, N_mhe=N_mhe, q=1e3, sampling_time=Te)
    estimator = distEst.MHE(model_param, mhe_param)

    ###########################################
    # Run the simulation
    ###########################################

    # simulation loops
    t0 = 0
    N_max_simu = 60 * 60 // Te

    # define dataframe to return
    line_list = []

    # initial conditions
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # reference for the robots
    bis_ref = 50

    # contains time history
    t = []
    u0 = ca.DM.zeros(2, 1)

    # contains history of states x
    xx = ca.DM.zeros(12, N_max_simu)
    xx[:, 0] = x0

    # contains history of bis
    bisbis = ca.DM.zeros(N_max_simu)

    u_cl = ca.DM.zeros(2, N_max_simu)

    for mpciter in range(N_max_simu):

        dist_type = 'step'
        # dist_type = 'realistic'

        disturbance = pas.compute_disturbances(mpciter * Te, dist_type, 60 * 10, 15 * 60)
        if (mpciter * Te == (60 * 5 // Te)) and (cost == 'nominal') and (R_maintenance is not None):
            R_mpc = R_maintenance

        disturbance = np.array(disturbance)
        actualBis, _, _, _ = real_simulator.one_step(u0[0], u0[1], 0, 0, disturbance, noise)

        u_cl[:, mpciter] = u0
        t.append(t0)
        bisbis[mpciter] = actualBis
        xx[:, mpciter] = x0

        dict_to_save = {
            'Time': mpciter * Te,
            'BIS': actualBis,
            'BIS_REFERENCE': bis_ref,
            'inputs': [np.array(u0)],
            'x': [np.array(x0)],
            'age': Patient_info[0],
            'height': Patient_info[1],
            'weight': Patient_info[2],
            'gender': Patient_info[3],
        }

        start = perf_counter()

        # compute the new initialisation for next step
        x0, _ = estimator.one_step(u0, actualBis)

        if mpciter == (60 * 5 // Te) and (cost != 'nominal'):
            controller.switch_phase()

        ###########################################
        # Event based u_eq
        ###########################################

        maintenance_phase = mpciter >= (60 * 5 // Te)
        each_10_starting_5 = (mpciter - (60 * 5 // Te)) % (60 * 10 // Te) == 0
        control_time = (mpciter) % (5 // Te) == 0
        outside_safe_range = control_time and (abs(actualBis - 50) > 5)

        if maintenance_phase and (each_10_starting_5 or outside_safe_range) and (cost == 'range'):
            # mean dist
            start = 1 * 60 // Te
            dist = np.mean(xx[-1, mpciter - start : mpciter])

            # last dist
            # dist = np.array(xx[-1, mpciter]).squeeze()

            est_hill_param = np.array(xx[8:-1, mpciter]).squeeze()
            x_slow = np.array(xx[[0, 3, 4, 7], mpciter]).squeeze()
            controller.update_ueq(dist, est_hill_param, x_slow)

        ###########################################
        # Update control law
        ###########################################

        if mpciter * Te % control_ts == 0:
            if use_real_data:
                xpropo = real_simulator.propo_pk.x
                xremi = real_simulator.remi_pk.x
                pd_param = real_simulator.bis_pd
                xcontrol = np.hstack((xpropo[:4], xremi[:4], pd_param.c50p, pd_param.c50r, pd_param.gamma, disturbance[0]))

            else:
                xcontrol = x0

            if cost == 'nominal':
                new_u = controller.one_step(xcontrol, bis_ref, R_mpc)

            else:
                new_u = controller.one_step(xcontrol[:8], bis_ref, xcontrol[8:])

            if block_u:

                if abs(u0[0] - new_u[0]) > 0.015:
                    u0[0] = new_u[0]

                if abs(u0[1] - new_u[1]) > 0.015:
                    u0[1] = new_u[1]

            else:

                u0 = new_u

        t0 = t0 + Te

        end = perf_counter()

        dict_to_save['computation_time'] = end - start
        line = pd.DataFrame(dict_to_save)
        line_list.append(line)

    results = pd.concat(line_list)

    return results
