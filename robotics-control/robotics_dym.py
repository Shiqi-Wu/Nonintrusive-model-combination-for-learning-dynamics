import numpy as np
import scipy
import math
from parameters import *
from scipy.integrate import solve_ivp

# Define parameters
m_b = 0.725
m_ax = -0.217
m_ay = -0.7888
J_bz = 2.66 * 1e-3
J_az = -7.93 * 1e-4
L = 0.071
d = 0.04
c = 0.105
rho = 1000
S = 0.03
C_D = 0.97
C_L = 3.9047
K_D = 4.5*1e-3
K_f = 0.7
K_m = 0.45
m1 = m_b - m_ax
m2 = m_b - m_ay
m = 1
J3 = J_bz - J_az
c1 = 1/2 * rho * S * C_D
c2 = 1/2 * rho * S * C_L
c4 = 1/J3 * K_D
c5 = 1/(2 * J3) * L**2 * m * c

# Define functions
def f_1(s):
    first_term = m2 / m1 * s[4]
    second_term = - c1 / m1 * s[3] * math.sqrt(s[3]**2 + s[4]**2)
    third_term = c2 / m1 * s[4] * math.sqrt(s[3]**2 + s[4]**2) * math.atan(s[4]/s[3])
    return first_term + second_term + third_term

def f_2(s):
    first_term = - m1 / m2 * s[3] * s[5]
    second_term = - c1 / m2 * s[4] * math.sqrt(s[3]**2 + s[4]**2)
    third_term = - c2 / m1 * s[3] * math.sqrt(s[3]**2 + s[4]**2) * math.atan(s[4]/s[3])
    return first_term + second_term + third_term

def f_3(s):
    return (m1 - m2) * s[3] * s[4] - c4 * np.sign(s[5]) * s[5]**2

def u_1(alpha_0, alpha_a):
    return (alpha_a**2 * (3 - 3/2*alpha_0**2 - 3/8*alpha_a**2))

def u_2(alpha_0, alpha_a):
    return alpha_a**2 * alpha_0

def f_4(alpha_0, alpha_a, omega_a):
    return m / (12 * m1) * L**2 * omega_a**2 * u_1(alpha_0, alpha_a)

def f_5(alpha_0, alpha_a, omega_a):
    return m / (4 * m2) * L**2 * omega_a**2 * u_2(alpha_0, alpha_a)

def f_6(alpha_0, alpha_a, omega_a):
    return -m / (4 * J3) * L**2 * c * omega_a**2 * u_2(alpha_0, alpha_a)
    
def ode_system(t, s, alpha1, alpha2, alpha3):

    # print("t:", t)
    # print("s:", s)
    # print("alpha1:", alpha1)
    # print("alpha2:", alpha2)
    # print("alpha3:", alpha3)

    dsdt = np.zeros(6)
    dsdt[0] = s[3]*np.cos(s[2]) - s[4]*np.sin(s[2])
    dsdt[1] = s[3]*np.cos(s[2]) + s[4]*np.sin(s[2])
    dsdt[2] = s[5]
    dsdt[3] = f_1(s) + K_f * f_4(alpha1, alpha2, alpha3)
    dsdt[4] = f_2(s) + K_f * f_5(alpha1, alpha2, alpha3)
    dsdt[5] = f_3(s) + K_m * f_6(alpha1, alpha2, alpha3)
    # print("f1(s)", f_1(s))
    # print("f4", K_f * f_4(alpha1, alpha2, alpha3))
    # print("dsdt:", dsdt)
    return dsdt

t_span = (0, dlt_t)
t_eval = [0, dlt_t]
def dynamics_solver(s_current, alpha, omega_a):
    sol = solve_ivp(ode_system, t_span, s_current, args=(alpha[0], alpha[1], omega_a[0]), method='Radau', t_eval=t_eval)
    return sol.y[:,1]