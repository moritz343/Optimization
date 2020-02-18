import numpy as np


class Damping:

    def __init__(self, stiffness, mass, damping_ratio, low_frequency, high_frequency):
        self.K = stiffness
        self.M = mass
        self.zeta = damping_ratio
        self.f1 = low_frequency
        self.f2 = high_frequency

    def rayleigh(self):
        omega1 = self.f1 * 2 * np.pi
        omega2 = self.f2 * 2 * np.pi
        alpha = self.zeta * 2 * omega1 * omega2 / (omega1 + omega2)
        beta = self.zeta * 2 / (omega1 * omega2)
        C = alpha * self.M + beta * self.K
        return C

    def modal(self):
        eig_val, eig_vec = np.linalg.eig(self.K-self.M)
        zeta_mat = np.diag(2*eig_val*self.zeta)
        C = np.matmul(np.matmul(self.M, eig_vec), np.matmul(zeta_mat, np.matmul(self.M, eig_vec.transpose())))
        return C