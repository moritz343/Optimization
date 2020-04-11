import numpy as np
import structural_dynamics as dyn
import matplotlib.pyplot as plt

"""Parameters"""
m = 50000.
m2 = 5000.
m3 = 000.1

E = 210000. # Elastic modulus Steel N/mm^2
I = 251700000. # Second moment of inertia HEB 300 cm^4
h = 3. # height of 1 floor m
n = 8. # number of columns #

k = 12 * E * I * 10000 * n / (h * 1000)**3 / 1000
k2 = m2 * (10 * 2 * np.pi)**2
k3 = m3 * (1 * 2 * np.pi)**2

"""Spectrum"""
spec_import = np.genfromtxt('spectrum.txt')
wrange_import = np.genfromtxt('FreqRange.txt')

EQ = 121
DT = np.genfromtxt('./Earthquakes/DT.txt')
gm = dyn.Earthquake(np.genfromtxt('./Earthquakes/' + str(EQ) + '.txt'), DT[EQ - 1])

# gm1.plot_signal()
# gm1.plot_fourier_amplitude()
# gm1.plot_fourier_phase()


M = np.array([
    [m, 0, 0, 0, 0,  0, 0],
    [0, m, 0, 0, 0,  0, 0],
    [0, 0, m, 0, 0,  0, 0],
    [0, 0, 0, m, 0,  0, 0],
    [0, 0, 0, 0, m,  0, 0],
    [0, 0, 0, 0, 0, m2, 0],
    [0, 0, 0, 0, 0,  0, m3]
])

K = np.array([
    [k+k, -k,      0,   0,      0,   0,   0],
    [-k, k+k,     -k,   0,      0,   0,   0],
    [0,   -k, k+k+k3,  -k,      0,   0, -k3],
    [0,    0,     -k, k+k,     -k,   0,   0],
    [0,    0,      0,  -k,   k+k2, -k2,   0],
    [0,    0,      0,   0,    -k2,  k2,   0],
    [0,    0,    -k3,   0,      0,   0,  k3]
])

coords = [[0, 3], [0, 6], [0, 9], [0, 12], [0, 15], [1.5, 15], [1.5, 9]]
size = [[1, 0.2], [1, 0.2], [1, 0.2], [1, 0.2], [1, 0.2], [0.3, 0.2], [0.3, 0.2]]

building = dyn.OptimalStructure(M, K, coordinates=coords, size=size, omega_range=wrange_import, spectrum=spec_import)
building.rayleigh(1, 5, 0.03)
C = building.C

# building.show_geometry()
building.newmark(gm.signal, gm.dt)
# building.time_history_animation(frame_step=10, magnification=40)
plt.figure()
plt.plot(building.time, building.displacement[0])
plt.show()

step_size_C = 2 * 0.05 * np.sqrt(k2 * m2) / 10
step_size_K = k2/10
building.optimizationC(step_size=step_size_C, dof1=5, dof2=6, controlDOF=5)
building.optimizationK(step_size=step_size_K, dof1=5, dof2=6, controlDOF=5)

building.newmark(gm.signal, gm.dt)
# building.time_history_animation(frame_step=10, magnification=40)
plt.figure()
plt.plot(building.time, building.displacement[0])
plt.show()

# def optimize_MTMD(self, dofs, step_size_K, step_size_C, controlDOF):
# dofs = [[5, 6], [3, 7]]
# step_size_C = 2 * 0.05 * np.sqrt(k2 * m2) / 10
# step_size_K = k2/10
# Variance_tot = []
# for tmd in dofs:
#     building.optimizationC(step_size=step_size_C, dof1=tmd[0], dof2=tmd[1], controlDOF=5)
#     building.optimizationK(step_size=step_size_K, dof1=tmd[0], dof2=tmd[1], controlDOF=5)
#


#
# structure.optimizationK(step_size=.1, dof1=2, dof2=3, controlDOF=2)
# print(structure.K)
#
# structure.optimizationC(step_size=1., dof1=2, dof2=3, controlDOF=2)
# print(structure.C)
#
# structure.optimizationK(step_size=.1, dof1=2, dof2=3, controlDOF=2)
# print(structure.K)