import casadi as ca
import numpy as np
from close_loop_anesth.utils import discretize


def range_function(y, ref, range) -> float:

    return ((y - ref) / range) ** 6


class NMPC_extended_single_shooting:

    def __init__(self, model_param, mpc_param) -> None:
        self.hill_param = model_param['hill_param']
        self.N = mpc_param['N']
        self.cost_type = mpc_param['cost_type']
        self.ts = mpc_param['ts']
        self.u_max = mpc_param['umax']
        self.u_min = mpc_param['umin']
        self.q_bis = mpc_param['q_bis']
        self.r = mpc_param['r_induction']
        self.r_maintenance = mpc_param['r_maintenance']

        if self.cost_type == "range":
            self.q_range = mpc_param['q_range']

        self.define_symbolic_function(model_param['A'], model_param['B'])

        self.u_eq = [0] * 2
        self.x_eq = [2] * 4
        self.define_equilibrium(model_param['A'], model_param['B'])
        self.solve_equilibrium_pb([2] * 4, model_param['hill_param'][:3], 0)

        self.define_mpc()
        self.previous_u = ca.DM.zeros(self.n_controls, self.N)

    def define_symbolic_function(self, A_model, B_model) -> None:
        # Declare full system propofol/remi
        A = ca.DM.zeros(8, 8)
        B = ca.DM.zeros(8, 2)
        A[0:8, 0:8], B[0:8, 0:2] = discretize(A_model, B_model, self.ts)

        ###########################################
        # Stability test
        ###########################################

        if False:

            eigA, _ = np.linalg.eig(A[:8, :8])
            print("     A eigenvalues : ", eigA)

            if True in (abs(eigA) >= 1):
                print("########################################################")
                print("######### ALERT MARGINALLY OR UNSTABLE PATIENT #########")
                print("########################################################")

        ###########################################
        # End stability test
        ###########################################

        # Declare CASADI variables

        # x1p, x2p, x3p, xep, x1r, x2r, x3r, xer [mg/ml]
        x = ca.MX.sym("x", 8)
        self.n_states_x = x.size1()

        u_prop = ca.MX.sym("prop")  # Propofol infusion rate [mg/s]
        u_rem = ca.MX.sym("rem")  # Remifentanil infusion rate [µg/s]
        u = ca.vertcat(u_prop, u_rem)
        self.n_controls = u.size1()

        # Compute states at next timestamp

        state_next_x = A @ x + B @ u

        # Compute state at next timestamp
        self.state_next = ca.Function("state_next", [x, u], [state_next_x])

        # Compute BIS for this timestamp
        estimated_hill_param = ca.MX.sym('constant_param', 4)  # C50p, C50r, gamma, disturbance

        # pharmaco dynamic model from : Optimized PID control of propofol and remifentanil coadministration for general anesthesia  (Luca Merigo, Fabrizio Padula, Nicola Latronico, Massimiliano Paltenghi, Antonio Visioli)
        u_prop_dyn = x[3] / estimated_hill_param[0]
        u_remi_dyn = x[7] / estimated_hill_param[1]
        phi = u_prop_dyn / (u_prop_dyn + u_remi_dyn + 1e-6)  # to avoid division by 0
        u_50 = 1 - self.hill_param[3] * phi + self.hill_param[3] * (phi**2)  # BIS_param[3] = beta
        to_the_power = ((u_prop_dyn + u_remi_dyn) / u_50) ** estimated_hill_param[2]

        compute_bis = self.hill_param[4] - self.hill_param[5] * (to_the_power / (1 + to_the_power)) + estimated_hill_param[3]
        # Compute BIS
        self.output = ca.Function("bis", [x, estimated_hill_param], [compute_bis])

    def define_mpc(self) -> None:

        ###########################################
        # Create symbolic function over the horizon
        ###########################################

        U = ca.MX.sym("U", self.n_controls, self.N)
        X = ca.MX.zeros(self.n_states_x, (self.N + 1))
        Y = ca.MX.zeros((self.N + 1))

        # [X0, BISref]
        P = ca.MX.sym("P", self.n_states_x + 1)

        estimated_hill_param = ca.MX.sym('constant_param', 4)

        ###########################################
        # Compute solution symbolically over the horizon
        ###########################################

        # Initialisation
        X[:, 0] = ca.vertcat(P[0 : self.n_states_x])

        for k in range(self.N):
            st_x = X[:, k]
            con = U[:, k]
            Y[k] = self.output(st_x, estimated_hill_param)
            X[:, k + 1] = self.state_next(st_x, con)

        Y[-1] = self.output(X[:, self.N], estimated_hill_param)

        ###########################################
        # Compute the objective function to minimize
        ###########################################

        obj = 0
        R = ca.MX.sym("R")
        r_matrix = ca.MX.zeros(2, 2)
        r_matrix[0, 0] = R
        r_matrix[1, 1] = R

        RANGE_WEIGHT = ca.MX.sym("RANGE_WEIGHT")
        BIS_WEIGHT = ca.MX.sym("BIS_WEIGHT")

        U_EQ = ca.MX.sym("U_EQ", 2, 1)

        # Compute objective function from X(0) to X(n-1)
        for k in range(self.N):
            output = Y[k]
            con = U[:, k]

            # compute cost
            obj = (
                obj
                + (output - P[self.n_states_x]) * BIS_WEIGHT * (output - P[self.n_states_x])
                + (con.T @ r_matrix @ con) * BIS_WEIGHT
                + ((con - U_EQ).T @ r_matrix @ (con - U_EQ)) * RANGE_WEIGHT
                + range_function(output, P[self.n_states_x], 8) * RANGE_WEIGHT
            )

        # Make decision variables one colum vector
        OPT_variables = ca.reshape(U, self.n_controls * self.N, 1)
        nlp_prob = {"f": obj, "x": OPT_variables, "p": ca.vertcat(P, estimated_hill_param, R, RANGE_WEIGHT, BIS_WEIGHT, U_EQ)}

        opts = {
            "ipopt": {
                "max_iter": 100,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
                "sb": 'yes',
            },
            "print_time": 0,
        }

        # Generate solver
        self.mpc_solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

        # Constraints on U
        lbx = ca.DM.zeros((2 * self.N, 1))
        ubx = ca.DM.zeros((2 * self.N, 1))

        lbx[0 : 2 * (self.N - 1) : 2] = self.u_min[0]
        lbx[1 : 2 * self.N : 2] = self.u_min[1]

        ubx[0 : 2 * (self.N - 1) : 2] = self.u_max[0]
        ubx[1 : 2 * self.N : 2] = self.u_max[1]

        self.lbx = lbx
        self.ubx = ubx

    def switch_phase(self):
        self.r = self.r_maintenance
        self.q_range = 1
        self.q_bis = 0

    def update_ueq(self, dist, est_hill_param, x_slow):
        self.solve_equilibrium_pb(x_slow, est_hill_param, dist)

    def define_equilibrium(self, A_model, B_model):
        Ad, Bd = discretize(A_model, B_model, self.ts)
        Aeq = np.zeros((4, 4))
        Aeq[:2, :2] = np.array([[Ad[0, 0], Ad[0, 3]], [Ad[3, 0], Ad[3, 3]]])
        Aeq[2:, 2:] = np.array([[Ad[4, 4], Ad[4, 7]], [Ad[7, 4], Ad[7, 7]]])

        Beq = Bd[[0, 3, 4, 7], :]

        Eeq = np.zeros((4, 4))
        Eeq[:2, :2] = np.array([[Ad[0, 1], Ad[0, 2]], [Ad[3, 1], Ad[3, 2]]])
        Eeq[2:, 2:] = np.array([[Ad[4, 5], Ad[4, 6]], [Ad[7, 5], Ad[7, 6]]])

        x_eq = ca.MX.sym('x_eq', 4)
        u_eq = ca.MX.sym('u', 2)
        x_slow = ca.MX.sym('x_slow', 4)
        est_hill_param = ca.MX.sym('hill_param', 3)
        disturbance = ca.MX.sym('disturbance')

        h = self.output(ca.vertcat(x_eq[0], x_slow[0], x_slow[1], x_eq[1], x_eq[2], x_slow[2], x_slow[3], x_eq[3]), ca.vertcat(est_hill_param, disturbance))

        J = (h - 50) ** 2

        x_plus = ca.MX(Aeq) @ x_eq + ca.MX(Beq) @ u_eq + ca.MX(Eeq) @ x_slow

        # equality constraint xk = xk+1
        g = (x_plus - x_eq).T @ (x_plus - x_eq)

        opts = {
            "ipopt": {
                "max_iter": 300,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
                "sb": 'yes',
            },
            "print_time": 0,
            "show_eval_warnings": False,
        }

        nlp_prob = {"f": J, "x": ca.vertcat(u_eq, x_eq), "g": g, "p": ca.vertcat(x_slow, est_hill_param, disturbance)}
        self.u_eq_solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def one_step(self, state_x: list, bisRef: float, bis_param: list) -> list:

        u0 = ca.horzcat(
            self.previous_u[:, 1 : self.previous_u.shape[1]],
            self.previous_u[:, self.previous_u.shape[1] - 1],
        )
        u0 = ca.reshape(u0, 2 * self.N, 1)
        mpc_parameters = ca.vertcat(state_x, bisRef, bis_param, self.r, self.q_range, self.q_bis, self.u_eq)

        sol = self.mpc_solver(
            x0=u0,
            lbx=self.lbx,
            ubx=self.ubx,
            p=mpc_parameters,
        )
        u = ca.reshape(sol['x'], 2, self.N)
        self.previous_u = u
        return u[:, 0]

    def solve_equilibrium_pb(self, x_slow, est_hill_param, dist):
        p = ca.vertcat(x_slow, est_hill_param, dist)
        x0 = ca.vertcat(self.u_eq, self.x_eq)

        sol = self.u_eq_solver(x0=x0, lbx=[0] * 6, ubx=[20] * 6, lbg=[0], ubg=[0], p=p)
        self.u_eq = sol['x'][:2]
        self.x_eq = sol['x'][2:]
