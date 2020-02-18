import numpy as np
import response_optimization as opt
import damping as dp

M = np.array([[110., 0., 0.],
              [0.,  70., 0.],
              [0.,  0., 10.]])

K = np.array([[200., -100., 0.0],
              [-100., 110.,  -15.],
              [0.,   -15.,  15.]])

C = dp.Damping(K, M, 0.05, 1, 5).modal()
# C = np.array([[0.43, -0.3, 0.0],
#               [-0.3,  0.4,  -0.1],
#               [0.,   -0.1,  0.1]])

spec_import = np.genfromtxt('spectrum.txt')
wrange_import = np.genfromtxt('FreqRange.txt')

# structure = opt.OptimizationInput(M, C, K, spec_import, wrange_import)
#
# structure.optimizationK(step_size=.1, dof1=2, dof2=3, controlDOF=2)
# print(structure.K)
#
# structure.optimizationC(step_size=1., dof1=2, dof2=3, controlDOF=2)
# print(structure.C)
#
# structure.optimizationK(step_size=.1, dof1=2, dof2=3, controlDOF=2)
# print(structure.K)