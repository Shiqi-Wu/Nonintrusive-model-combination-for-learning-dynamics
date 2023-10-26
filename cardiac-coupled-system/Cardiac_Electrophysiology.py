import numpy as np
from collections import OrderedDict
import scipy

class Cardiac_Electrophysiology_Hodgkin_Huxley_1952(object):
    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model
        """
        self._parameters = self.default_parameters()
        
    @staticmethod
    def default_parameters():
        params = OrderedDict([("g_Na", 120.0),
                              ("g_K", 36.0),
                              ("g_L", 0.3),
                              ("Cm", 1.0),
                              ("E_R", -75.0)])
        return params
    
    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V", -75.0),
                          ("m", 0.052955),
                          ("h", 0.595994),
                          ("n", 0.317732)])
        return ic
    
    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else 0.0
        
        # Assign states
        V = v
        assert(len(s) == 3)
        m, h, n = s
        
        # Assign parameters
        g_Na = self._parameters["g_Na"]
        g_K = self._parameters["g_K"]
        g_L = self._parameters["g_L"]
        Cm = self._parameters["Cm"]
        E_R = self._parameters["E_R"]
        
        # Init return args
        current = [np.zeros(V.shape)]
        
        # Expressions for the sodiun channel componet
        E_Na = 115.0 + E_R
        i_Na = g_Na * (m * m * m) * (-E_Na + V) * h
        
        # Expressions for the Potassium channel component
        E_K = -12.0 + E_R
        i_K = g_K * np.power(n, 4) * (-E_K + V)
        
        # Expressions for the Leakage current component
        E_L = 10.613 + E_R
        i_L = g_L * (-E_L + V)
        
        # Expressions for the Membrane component
        i_Stim = 0.0
        current[0] = (-i_K - i_L - i_Na + i_Stim) / Cm
        
        # Return results
        return current[0]
    
    def I(self, v, s, time = None):
        """
        Transmembrane current
            I = -dV/dt
        """
        return -self._I(v, s, time)
    
    def F(self, v, s, time = None):
        """
        Right hand side for ODE system
        """
        time = time if time else [0.0]
        
        # Assign states
        V = v
        assert(len(s) == 3)
        m, h, n = s
        
        # Init return args
        F_expressions = [np.zeros(V.shape), np.zeros(V.shape), np.zeros(V.shape)]
        
        # Expressions for the m gate component
        alpha_m = (-5.0 - 0.1 * V)/(-1.0 + np.exp(-5.0 - V/10.0))
        beta_m = 4 * np.exp(-25.0/6.0 - V/18.0)
        F_expressions[0] = (1 - m) * alpha_m - beta_m * m
        
       # Expressions for the h gate component
        alpha_h = 0.07*np.exp(-15.0/4.0 - V/20.0)
        beta_h = 1.0/(1 + np.exp(-9.0/2.0 - V/10.0))
        F_expressions[1] = (1 - h)*alpha_h - beta_h*h

        # Expressions for the n gate component
        alpha_n = (-0.65 - 0.01*V)/(-1.0 + np.exp(-13.0/2.0 - V/10.0))
        beta_n = 0.125*np.exp(-15.0/16.0 - V/80.0)
        F_expressions[2] = (1 - n)*alpha_n - beta_n*n
        
        # Return results
        return F_expressions
    
    def num_states(self):
        return 3
    
    def __str__(self):
        return 'Hodgkin_Huxley_1952 cell model'

    """
    def initial_conditions(self):
        "Return initial conditions for v and s as an Expression."
        return Expression(list(self.default_initial_conditions().keys()), degree=1,
                          **self.default_initial_conditions())
    """

class Cardiac_Electrophysiology_DataModel:
    def __init__(self, crossed=False):
        # Create cell model
        self.cellmodel = Cardiac_Electrophysiology_Hodgkin_Huxley_1952()
        self.num_states = self.cellmodel.num_states()
    
    def vs_reshape(self, points, num_x):
        v = points[:num_x**2]
        v = np.reshape(v, (num_x,-1))
        s = points[num_x**2:]
        s = np.reshape(s, (3, num_x, -1))
        return v, s
    
    def BuildDivergence(self, v, M, dlt_v):
        v_1 = np.zeros(np.shape(v))
        v_1[:, 0] = 2 * v[:, 1] - 2 * v[:, 0]
        v_1[:, -1] = 2 * v[:, -2] - 2 * v[:, -1]
        v_2 = np.zeros(np.shape(v))
        v_2[0, :] =  2 * v[1, :] - 2 * v[0, :]
        v_2[-1, :] = 2 * v[-2, :] - 2 * v[-1, :]
        for i in range(1, np.shape(v)[0] - 1):
            v_1[:, i] = v[:, i - 1] + v[:, i + 1] - 2 * v[:, i]
            v_2[i, :] = v[i - 1, :] + v[i + 1, :] - 2 * v[i, :]

        div_v = (M[0] * v_1 + M[1] * v_2)/dlt_v**2
        return div_v
    
    def dx(self, x1, x2):
        dx = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                dx[i,j] = x1[i] + 10 - x2[j]
        return dx
    
    def BuildFunction(self, points, t, M, dlt_v, num_x, x1, x2):
        v, s = self.vs_reshape(points, num_x)
        div_v = self.BuildDivergence(v, M, dlt_v)
        I = self.cellmodel.I(v, s)
        F = self.cellmodel.F(v, s)
        dx = self.dx(x1, x2)
        I = I + dx * np.sin(t)
        OdeFunction = [I + div_v]
        print(t)
        for i in range(self.num_states):
            OdeFunction.append(F[i])
        OdeFunction = np.array(OdeFunction)    
        return np.reshape(OdeFunction, (1,-1))[0]
    
    def TestFunction(self, points, t, M, dlt_v, num_x, x1, x2):
        v, s = self.vs_reshape(points, num_x)
        div_v = self.BuildDivergence(v, M, dlt_v)
        I = self.cellmodel.I(v, s)
        F = self.cellmodel.F(v, s)
        dx = self.dx(x1, x2)
        # u_dx = (dx < 3)
        # if t <= 2:
        #     I = I + u_dx * 5
        I = I + dx * np.cos(t)
        OdeFunction = [I + div_v]
        print(t)
        for i in range(self.num_states):
            OdeFunction.append(F[i])
        OdeFunction = np.array(OdeFunction)    
        return np.reshape(OdeFunction, (1,-1))[0]