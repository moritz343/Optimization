import numpy as np
import structural_dynamics as dyn
import matplotlib.pyplot as plt

"""System Parameters"""
m1 = 58836.
m2 = 267907.
mi = 451666.
mc = 85773.

k1 = 850180000.
k2 = (1*2*np.pi)**2 * m2
ki = 835180000.
kc = 386480.

c1 = 2 * 0.05 * np.sqrt(m1 * k1)
c2 = 2 * 0.03 * np.sqrt(m2 * k2)
ci = 1942230.
cc = 1821.

M = np.array([
    [m1, 0, 0, 0],
    [0, m2, 0, 0],
    [0, 0, mi, 0],
    [0, 0, 0, mc]
])

K = np.array([
    [k1 + k2 + ki + kc,     -k2,       -ki,   -kc],
    [-k2,                    k2,         0,     0],
    [-ki,                     0,        ki,     0],
    [-kc,                     0,         0,    kc]
])

C = np.array([
    [c1 + c2 + ci + cc,     -c2,       -ci,   -cc],
    [-c2,                    c2,         0,     0],
    [-ci,                     0,        ci,     0],
    [-cc,                     0,         0,    cc]
])

coords = [[0, 3], [0, 1.5], [0, 6], [1, 9]]
size = [[1, 0.2], [0.5, 0.2], [0.3, 0.3], [0.2, 0.2]]

"""Spectrum"""
spec_import = np.genfromtxt('spectrum.txt')
wrange_import = np.genfromtxt('FreqRange.txt')

DT = np.genfromtxt('./Earthquakes/DT.txt')
# gm = dyn.Earthquake(np.genfromtxt('./Earthquakes/' + str(EQ) + '.txt'), DT[EQ - 1])

# ktcp = dyn.KanaiTajimi(wrange_import, 1., 0.7, 10.5, 0.65)
# ktcp.plot_spectrum()


# metafoundation = dyn.Structure(M, K, C, coordinates=coords, size=size, omega_range=wrange_import, spectrum=spec_import)
metafoundation = dyn.Structure(M, K, C, coordinates=coords, size=size)

tank = dyn.Structure(M[2:, 2:], K[2:, 2:], C[2:, 2:])

nEQ = 20

maxbs_tank = []
for EQ in range(nEQ):
    gm = np.genfromtxt('./Earthquakes/' + str(EQ+1) + '.txt')
    tank.newmark(gm, DT[EQ])
    base_shear_tank = ((np.asarray(tank.displacement[0])) * ki)
    maxbs_tank.append(max(base_shear_tank*1.5))
    print('Earthquake ' + str(EQ+1) + ' of ' + str(nEQ))
total_bs_tank = np.sum(maxbs_tank)

maxbs = []
for EQ in range(nEQ):
    gm = np.genfromtxt('./Earthquakes/' + str(EQ+1) + '.txt')
    metafoundation.newmark(gm, DT[EQ])
    base_shear = ((np.asarray(metafoundation.displacement[0]) - np.asarray(metafoundation.displacement[2])) * ki)
    maxbs.append(max(base_shear))
    print('Earthquake ' + str(EQ+1) + ' of ' + str(nEQ))
total_bs = np.sum(maxbs)

step_size_C = 2 * np.sqrt(k2 * m2) / 10000
df = 0.5
step_size_K = (df * 2 * np.pi)**2 * m2
# step_size_K = k2/100
metafoundation.optimization_abs_acc_C(step_size=step_size_C, dof1=2, dof2=1, controlDOF=3)
print('Damping done')
metafoundation.optimization_abs_acc_K(step_size=step_size_K, dof1=2, dof2=1, controlDOF=3)
print('Stiffness done')


maxbs2 = []
for EQ in range(nEQ):
    gm = np.genfromtxt('./Earthquakes/' + str(EQ + 1) + '.txt')
    metafoundation.newmark(gm, DT[EQ])
    base_shear2 = ((np.asarray(metafoundation.displacement[0]) - np.asarray(metafoundation.displacement[2])) * ki)
    maxbs2.append(max(base_shear2))
    print('Earthquake ' + str(EQ+1) + ' of ' + str(nEQ))

total_bs2 = np.sum(maxbs2)
print('Simulations Tank Base shear ' + str(total_bs_tank))
print('Simulations META non Opt Base shear ' + str(total_bs))
print('Simulations META Opt Base shear ' + str(total_bs2))

k_opt = metafoundation.K[0, 1]
f_opt = np.sqrt(abs(k_opt)/m2)/2/np.pi

c_opt = abs(metafoundation.C[0, 1])
zeta_opt = c_opt/(2*np.sqrt(m2 * k2))

print('Optimal Frequency = ' + str(f_opt) + ' Hz')
print('Optimal Damping = ' + str(zeta_opt))
