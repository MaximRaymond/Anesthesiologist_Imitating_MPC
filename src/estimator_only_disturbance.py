import casadi as ca
import numpy as np
from close_loop_anesth.utils import discretize


class MHE:
    """Implementation of the Moving Horizon Estimator for the Coadministration of propofol and remifentanil in Anesthesia.

    x = [x1p, x2p, x3p, xep, x1r, x2r, x3r, xer, disturbance]

    Parameters
    ----------
    A : list
        Dynamic matric of the continuous system dx/dt = Ax + Bu.
    B : list
        Input matric of the continuous system dx/dt = Ax + Bu.
    BIS_param : list
        Contains parameters of the non-linear function output BIS_param = [C50p, C50r, gamma, beta, E0, Emax]
    ts : float, optional
        Sampling time of the system. The default is 1.
    x0 : list, optional
        Initial state of the system. The default is np.zeros((8, 1)).
    Q : list, optional
        Covariance matrix of the process uncertainties. The default is np.eye(8).
    R : list, optional
        Covariance matrix of the measurement noises. The default is np.array([1]).
    horizon_length : int, optional
        Number of steps of the horizon. The default is 20.
    theta : list, optional
        Parameters of the Q matrix. The default is np.ones(16).

    Returns
    -------
    None.

    """

    def __init__(self, model_param, mhe_param) -> None:

        self.nb_states = 9
        self.nb_inputs = 2
        self.ts = mhe_param['ts']

        Ad, Bd = discretize(model_param['A'], model_param['B'], self.ts)

        # Add the disturbance to be estimated in the states
        self.A = np.block([[Ad, np.zeros((self.nb_states - 1, 1))], [np.zeros((1, self.nb_states - 1)), np.eye(1)]])
        self.B = np.block([[Bd], [np.zeros((1, 2))]])

        self.BIS_param = model_param['hill_param']
        self.Q = ca.MX(mhe_param['Q'])
        self.P = mhe_param['P']
        self.R = mhe_param['R']
        self.N = mhe_param['horizon_length']

        # system dynamics
        self.declare_dynamic_functions()

        # optimization problem
        self.define_opt_problem()

        # define bound for variables
        self.lbx = ([1e-6] * (self.nb_states - 1) + [-40]) * self.N
        self.ubx = ([20] * (self.nb_states - 1) + [40]) * self.N

        # init state and output
        self.x = np.array([[1e-6] * (self.nb_states - 1) + [0]]).T * np.ones((1, self.N))
        self.y = []
        self.u = np.zeros(2 * self.N)
        self.x_pred = self.x.reshape(self.nb_states * self.N, order='F')
        self.time = 0

    def declare_dynamic_functions(self):
        _, _, _, beta, E0, Emax = self.BIS_param

        # declare CASADI variables
        x = ca.MX.sym('x', self.nb_states)  # x1p, x2p, x3p, x4p, x1r, x2r, x3r, x4r, disturbance
        u = ca.MX.sym('u', self.nb_inputs)  # Propofol and remifentanil infusion rate

        # declare CASADI functions
        xpred = ca.MX(self.A) @ x + ca.MX(self.B) @ u
        self.Pred = ca.Function('Pred', [x, u], [xpred], ['x', 'u'], ['xpred'])

        C50p, C50r, gamma, _, _, _ = self.BIS_param

        up = x[3] / C50p
        ur = x[7] / C50r
        Phi = up / (up + ur + 1e-6)
        U_50 = 1 - beta * (Phi - Phi**2)
        i = (up + ur) / U_50
        y = E0 - Emax * i**gamma / (1 + i**gamma) + x[8]
        self.output = ca.Function('output', [x], [y], ['x'], ['bis'])

    def define_opt_problem(self):
        # optimization problem
        # optimization variables
        x_bar = ca.MX.sym('x0', self.nb_states * self.N)
        # parameters
        x_pred0 = ca.MX.sym('x_pred', self.nb_states)
        u = ca.MX.sym('u', self.nb_inputs * self.N)
        y = ca.MX.sym('y', self.N)
        time = ca.MX.sym('time', 1)

        # objective function
        J = 0
        P_disturbance = 0

        P = ca.blockcat([[self.P, ca.MX(np.zeros((self.nb_states - 1, 1)))], [ca.MX(np.zeros((1, self.nb_states - 1))), ca.diag(ca.vertcat(P_disturbance))]])

        for i in range(0, self.N):
            x_i = x_bar[self.nb_states * i : self.nb_states * (i + 1)]
            u_i = u[self.nb_inputs * (i) : self.nb_inputs * (i + 1)]
            # cost function
            J += (y[i] - self.output(x_i)) ** 2 * self.R
            if i < self.N - 1:
                x_i_plus = x_bar[self.nb_states * (i + 1) : self.nb_states * (i + 2)]
                x_bar_plus = self.Pred(x=x_i, u=u_i)['xpred']
                J += (x_i_plus - x_bar_plus).T @ self.Q @ (x_i_plus - x_bar_plus)
            if i == 0:
                J += (x_i - x_pred0).T @ P @ (x_i - x_pred0)

        # create solver instance
        opts = {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0, 'ipopt.max_iter': 300}
        prob = {'f': J, 'p': ca.vertcat(*[u, y, x_pred0, time]), 'x': x_bar}
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    def one_step(self, u, Bis) -> tuple[list, list]:
        """solve the MHE problem for one step.

        Parameters
        ----------
        u : list
            propofol and remifentanil infusion rate at time t-1
        Bis : float
            BIS value at time t

        Returns
        -------
        np.array
            state estimation at time t

        """
        if len(self.y) == 0:
            self.y = [Bis] * self.N
        else:
            self.y = self.y[1:] + [Bis]

        self.u = np.hstack((self.u[2:], np.ndarray.flatten(np.array(u))))
        self.time += self.ts
        # init the problem
        x0 = []
        for i in range(self.N):
            if len(self.x_pred[self.nb_states * i : self.nb_states * (i + 1)]) != 9:
                print(0)
            x0 += list(self.Pred(x=self.x_pred[self.nb_states * i : self.nb_states * (i + 1)], u=self.u[2 * i : 2 * (i + 1)])['xpred'].full().flatten())
        x_pred_0 = self.x_pred[self.nb_states : 2 * self.nb_states]
        # solve the problem
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, p=ca.vertcat(*[self.u, self.y, x_pred_0, self.time]))
        x_bar = res['x']

        self.x = np.reshape(x_bar, (self.nb_states, self.N), order='F')
        self.x_pred = np.array(res['x']).reshape(self.nb_states * self.N)
        bis = float(self.output(x=self.x[:, [-1]])['bis'])

        return np.hstack((self.x[:-1, -1], self.BIS_param[:3], self.x[-1, -1])), bis
