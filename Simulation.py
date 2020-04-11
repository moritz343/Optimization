import numpy as np
import structural_dynamics as dyn
import matplotlib.pyplot as plt
import pandas as pd

"""Creation of the the dynamic system"""
xl = pd.ExcelFile('Matrix.xlsx')
df_mass = xl.parse('Mass', header=None)
df_stiffness = xl.parse('Stiffness', header=None)
df_damping = xl.parse('Damping', header=None)

M = np.asarray(df_mass)
K = np.asarray(df_stiffness)
C = np.asarray(df_damping)

coordinates = [[0, 0], [0, 1], [2, 1.2], [0, 2]]
size = [[1, 0.1], [1, 0.1], [1, 0.1], [1, 0.1]]

system = dyn.Structure(M, K, C, coordinates, size)
system.show_geometry()
# system.rayleigh(0.5, 7, 0.1) # for Rayleigh damping

"""Time Integration Parameters"""
T = 20
num_step = 10000
dt = T / num_step
Time = np.arange(dt, T + dt, dt)

# """Parametric frequency sweep for numerical FRF"""
# frf_U1 = []
# frf_U2 = []
# omega_range = np.arange(6, 80, 1.)
# for omega in omega_range:

"""Excitation of Periodic lattice"""
Amp = 0.1
omega = 1 * 2 * np.pi
q = 0.5
ajm1 = -omega**2 * Amp * np.cos(omega*Time - q)
ajp1 = -omega**2 * Amp * np.cos(omega*Time + q)
excit = np.zeros([len(ajm1), len(M)])
excit[:, 0] = ajm1
excit[:, 3] = ajp1

"""Initial conditions of Periodic lattice"""
ujm1_ini = Amp * np.cos(-q)
ujp1_ini = Amp * np.cos(q)
u_ini = np.asarray([ujm1_ini, 0, 0, ujp1_ini])

vjm1_ini = - Amp * omega * np.sin(-q)
vjp1_ini = - Amp * omega * np.sin(q)
v_ini = np.asarray([vjm1_ini, 0, 0, vjp1_ini])

ajm1_ini = -omega**2 * Amp * np.cos(-q)
ajp1_ini = -omega**2 * Amp * np.cos(q)
a_ini = np.asarray([ajm1_ini, 0, 0, ajp1_ini])


"""Time integration and plotting"""
A, V, U, time = system.newmark(excit, dt=dt, a_ini=a_ini, v_ini=v_ini, u_ini=u_ini)
"Maximum Displacements"
max_list = [max(map(abs, item)) for item in U]

# for i in range(max(np.shape(M))):
#     plt.plot(time, system.displacement[i], label='DOF' + str(i+1))
# plt.legend(loc='upper left')
# plt.xlabel('Time [s]')
# plt.ylabel('Displacement U [m]')
# plt.show()

system.time_history_animation(frame_step=10)

    # frf_U1.append(max(U[1][int(-num_step/5): -1]))
    # frf_U2.append(max(U[2][int(-num_step/5): -1]))

# plt.figure()
# plt.title('Waven number q = '+str(q))
# plt.plot(omega_range, frf_U1, label='Main Chain')
# plt.plot(omega_range, frf_U2, label='Resonator')
# plt.legend()
# plt.xlabel('Frequency \\omega [rad/s]')
# plt.ylabel('Response U [m]')
# plt.show()
