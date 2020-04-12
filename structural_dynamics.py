import numpy as np
"""This class optimizes 1D linear dynamic systems and is based off the structural dynamics class defined above.
The class requires the mass, damping, and stiffness matrices of the classical equations of motion. The matrices
need to be written in the direct form (the mass matrix needs to be diagonal and contain only the masses of the
individual degrees of freedom DOFs). Besides the matrices, also the power spectral density PSD function of the
ground motion and its corresponding frequency range need to be provided to the class."""

class Structure:

    def __init__(self, mass, stiffness, damping=None, coordinates=None, size=None, omega_range=None, spectrum=None):
        self.M = mass
        self.K = stiffness
        if damping is None:
            self.C = self.M * 0
        else:
            self.C = damping
        self.coordinates = coordinates
        self.size = size
        self.omega_range = omega_range
        self.spectrum = spectrum

    def rayleigh(self, low_frequency, high_frequency, zeta):
        """Rayleigh damping model changes the damping matrix of the system"""
        omega1 = low_frequency * 2 * np.pi
        omega2 = high_frequency * 2 * np.pi
        alpha = zeta * 2 * omega1 * omega2 / (omega1 + omega2)
        beta = zeta * 2 / (omega1 * omega2)
        self.C = alpha * self.M + beta * self.K

    def newmark(self, excitation, dt, exc_vec=None, a_ini=None, v_ini=None, u_ini=None, beta=0.5, gamma=0.25):
        """Standard newmark beta scheme for solving linear dynamic systems"""
        self.dt = dt

        """If no instructions are given for the initial conditions, they are assumed to be 0"""
        if exc_vec is None:
            exc_vec = np.diag(self.M)
        if a_ini is None:
            a_ini = np.zeros(len(self.M))
        if v_ini is None:
            v_ini = np.zeros(len(self.M))
        if u_ini is None:
            u_ini = np.zeros(len(self.M))

        """Setting the initial conditions"""
        a = []
        v = []
        u = []
        a.append(a_ini)
        v.append(v_ini)
        u.append(u_ini)
        time = [0.]

        """Inversion of the dynamic stiffness matrix"""
        dynamic_stiffness = np.linalg.inv((self.M + self.C + self.K * dt ** 2 / 4))

        """Time integration"""
        for i in range(max(np.shape(excitation))):
            a.append(
                dynamic_stiffness.dot(- self.C.dot(v[i] + dt * beta * a[i])
                                      - self.K.dot(u[i] + dt * v[i] + dt ** 2 * gamma * a[i])
                                      + (exc_vec * excitation[i]))
            )
            v.append(
                v[i] + (a[i] + a[i + 1]) * dt * beta
            )
            u.append(
                u[i] + v[i] * dt + (a[i] + a[i + 1]) * dt ** 2 * gamma
            )

            time.append(dt * i)

        self.acceleration = list(map(list, zip(*a)))
        self.velocity = list(map(list, zip(*v)))
        self.displacement = list(map(list, zip(*u)))
        self.time = time
        return self.acceleration, self.velocity, self.displacement, self.time

    def show_geometry(self):
        """Function to display the geometry of the system"""
        import matplotlib.pyplot as plt
        fig_geom = plt.figure()
        ax_geom = fig_geom.add_subplot(111)
        rectangle = []
        for i in range(len(self.coordinates)):
            rectangle.append(plt.Rectangle((self.coordinates[i][0],
                                            self.coordinates[i][1]),
                                           self.size[i][0], self.size[i][1], alpha=0.5))
            ax_geom.add_patch(rectangle[i])
        plt.axis('auto')
        plt.show()

    def time_history_animation(self, frame_step=1, magnification=1):
        """Function for animating the results from the newmark solver"""
        import matplotlib.pyplot as plt
        import matplotlib.animation as ani

        """Retrieve maximum displacement for axis limits"""
        max_list = [max(map(abs, item)) * magnification for item in self.displacement]

        """Start figure for animation"""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        """Define the rectangles that represent the DOFs"""
        rectangle = []
        for i in range(len(self.coordinates)):
            rectangle.append(plt.Rectangle((self.coordinates[i][0],
                                            self.coordinates[i][1]),
                                           self.size[i][0], self.size[i][1], alpha=0.5))

        """Init function for animation draws the frame, so that blip can be used and the animation runs faster"""
        def init():
            for i in range(len(self.coordinates)):
                ax.add_patch(rectangle[i])
                plt.axis('auto')
                plt.xlim([-max(max_list) + min(self.coordinates[:][0]),
                          max(max_list) + max([item[0] for item in self.coordinates]) + max(self.size[:][0])])
            return rectangle

        """Animation function: only the coordinates of the rectangles are updated here"""
        def motion(t_step):
            for i in range(len(self.coordinates)):
                rectangle[i].set_xy((float(self.coordinates[i][0]
                                           + self.displacement[i][t_step * frame_step] * magnification),
                                    float(self.coordinates[i][1])))
            return rectangle

        """Animation function: inter gives the time delay between frames in milli seconds"""
        inter = int(1000 * self.dt * frame_step)
        self.anim = ani.FuncAnimation(fig,
                                      motion,
                                      init_func=init,
                                      interval=inter,
                                      blit=True)

        motion(int(len(self.displacement) / frame_step))
        plt.show()

    def incrementK(self, step_size, dof1, dof2):
        """This function increments the stiffness matrix for a spring that acts between dof1 and dof2.
        The step_size controls the amount of stiffness that is added to the spring.
        dof1 and dof2 mark the dofs which are connected by the spring, which is supposed to be incremented."""
        self.K[dof1 - 1, dof1 - 1] += step_size
        self.K[dof1 - 1, dof2 - 1] += -step_size
        self.K[dof2 - 1, dof1 - 1] += -step_size
        self.K[dof2 - 1, dof2 - 1] += step_size

    def incrementC(self, step_size, dof1, dof2):
        """Same function as above for the damping matrix"""
        self.C[dof1 - 1, dof1 - 1] += step_size
        self.C[dof1 - 1, dof2 - 1] += -step_size
        self.C[dof2 - 1, dof1 - 1] += -step_size
        self.C[dof2 - 1, dof2 - 1] += step_size

    def VarianceOfResponse(self):
        """This functions calculates the variance of the response for all DOFs. It uses the matrices of the system,
        computes the transmission matrix H, and integrates it together with the PSD of the spectrum.
        In principle, the the PSD of the response can be calculated with PSD_reponse = abs(H)^2 * PSD_spectrum.
        Here the PSD_spectrum represents a vector that contains the diagonal of the mass matrix (or M*I, with I being
        the identity vector) and multiplies it with the spectrum defined by the user (self.spectrum)."""
        H = []
        for i in range(len(self.omega_range)):
            """Calculation of the Transmission matrix H"""
            H.append(np.linalg.inv((-self.omega_range[i] ** 2 * self.M
                                    - 1j * self.omega_range[i] * self.C
                                    + self.K)))
        """squared absolute of the transmission matrix H multiplied with the diagonal of the mass matrix M (M*I)"""
        Habs2 = [(np.abs(matrix)**2) for matrix in H]
        PSDexc = [np.transpose(np.diagonal(self.M)) * spec_val for spec_val in self.spectrum]
        """Response of all DOFs as PSD"""
        RespPSD = [Habs2[wincr].dot(PSDexc[wincr]) for wincr in range(len(self.spectrum))]
        """The variance of the response can be obtained with the integral of the response PSD. 
        integral(PSD_response)"""
        variance = (np.trapz(RespPSD, self.omega_range, axis=0))
        return variance

    def optimization_rel_disp_K(self, step_size, dof1, dof2, controlDOF):
        """Optimization of the system by minimizing the variance"""
        """First, the initial variance of the system is computed for the matrices given by the user"""
        Variance = []
        """The sign variable keeps track of the direction the algorithm is stepping"""
        sign = [0]
        Variance.append(self.VarianceOfResponse())

        """The matrix K is incremented by one step with size = step_size. for the spring that connects dof1 and dof2"""
        sign.append(1)
        self.incrementK(step_size, dof1, dof2)
        Variance.append(self.VarianceOfResponse())

        """Loop that iteratively steps down the optimal direction of increasing or reducing K"""
        i = 0
        while len(sign) < 4 or np.sum(sign[-4:]) != 0 or i < 100:
            i += 1
            print(abs(Variance[-2][int(controlDOF[0] - 1)] - Variance[-2][int(controlDOF[1] - 1)]))
            """First condition demands a minimum of 4 steps. Second condition sums up the direction of the last 4 steps
            if the sum equals 0, the minimum value for the variance has been found."""
            if abs(Variance[-2][int(controlDOF[0] - 1)] - Variance[-2][int(controlDOF[1] - 1)]) \
                    > abs(Variance[-1][int(controlDOF[0] - 1)] - Variance[-1][int(controlDOF[1] - 1)]):
                """If the last variance is smaller than the variance before, keep going in that direction."""
                self.incrementK(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1])
            elif abs(Variance[-2][int(controlDOF[0] - 1)] - Variance[-2][int(controlDOF[1] - 1)]) \
                    < abs(Variance[-1][int(controlDOF[0] - 1)] - Variance[-1][int(controlDOF[1] - 1)]):
                """If the last variance is bigger than the one before, turn around and go the other way."""
                self.incrementK(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1] * -1)
        return Variance

    def optimization_rel_disp_C(self, step_size, dof1, dof2, controlDOF):
        """Same as for the stiffness matrix K"""
        Variance = []
        sign = []
        sign.append(0)
        Variance.append(self.VarianceOfResponse())

        sign.append(1)
        self.incrementC(step_size, dof1, dof2)
        Variance.append(self.VarianceOfResponse())

        i = 0
        while len(sign) < 4 or np.sum(sign[-4:]) != 0 or i < 100:
            i += 1
            if abs(Variance[-2][int(controlDOF[0] - 1)] - Variance[-2][int(controlDOF[1] - 1)]) \
                    > abs(Variance[-1][int(controlDOF[0] - 1)] - Variance[-1][int(controlDOF[1] - 1)]):
                self.incrementC(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1])
            elif abs(Variance[-2][int(controlDOF[0] - 1)] - Variance[-2][int(controlDOF[1] - 1)]) \
                    < abs(Variance[-1][int(controlDOF[0] - 1)] - Variance[-1][int(controlDOF[1] - 1)]):
                self.incrementC(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1] * -1)
        return Variance

    def VarianceOfAbsAcceleration(self):
        """This functions calculates the variance of the response for all DOFs. It uses the matrices of the system,
        computes the transmission matrix H, and integrates it together with the PSD of the spectrum.
        In principle, the the PSD of the response can be calculated with PSD_reponse = abs(H)^2 * PSD_spectrum.
        Here the PSD_spectrum represents a vector that contains the diagonal of the mass matrix (or M*I, with I being
        the identity vector) and multiplies it with the spectrum defined by the user (self.spectrum)."""
        H = []
        for i in range(len(self.omega_range)):
            """Calculation of the Transmission matrix H"""
            H.append(np.linalg.inv((-self.omega_range[i] ** 2 * self.M
                                    - 1j * self.omega_range[i] * self.C
                                    + self.K)))
        """squared absolute of the transmission matrix H multiplied with the diagonal of the mass matrix M (M*I)"""
        FRFacc = [H[wincr].dot(np.diagonal(self.M)) * self.omega_range[wincr]**2 for wincr in range(len(self.spectrum))]
        Habs2 = [(np.abs(vector)**2) for vector in FRFacc]
        PSDexc = self.spectrum
        """Response of all DOFs as PSD"""
        RespPSD = [Habs2[wincr] * PSDexc[wincr] for wincr in range(len(self.spectrum))]
        AccPSD = [abs(RespPSD[wincr] + 0* PSDexc[wincr]) for wincr in range(len(self.spectrum))]
        """The variance of the response can be obtained with the integral of the response PSD. 
        integral(PSD_response)"""
        variance = (np.trapz(AccPSD, self.omega_range, axis=0))
        return variance

    def optimization_abs_acc_K(self, step_size, dof1, dof2, controlDOF):
        """Optimization of the system by minimizing the variance"""
        """First, the initial variance of the system is computed for the matrices given by the user"""
        Variance = []
        """The sign variable keeps track of the direction the algorithm is stepping"""
        sign = [0]
        Variance.append(self.VarianceOfAbsAcceleration())

        """The matrix K is incremented by one step with size = step_size. for the spring that connects dof1 and dof2"""
        sign.append(1)
        self.incrementK(step_size, dof1, dof2)
        Variance.append(self.VarianceOfAbsAcceleration())

        """Loop that iteratively steps down the optimal direction of increasing or reducing K"""
        i = 0
        while len(sign) < 4 or np.sum(sign[-8:]) != 0:
            i += 1
            if i == 400:
                break
            print(Variance[-2][int(controlDOF - 1)])
            """First condition demands a minimum of 4 steps. Second condition sums up the direction of the last 4 steps
            if the sum equals 0, the minimum value for the variance has been found."""
            if Variance[-2][int(controlDOF - 1)] > Variance[-1][int(controlDOF - 1)]:
                """If the last variance is smaller than the variance before, keep going in that direction."""
                self.incrementK(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfAbsAcceleration())
                sign.append(sign[-1])
            elif Variance[-2][int(controlDOF - 1)] < Variance[-1][int(controlDOF - 1)]:
                """If the last variance is bigger than the one before, turn around and go the other way."""
                self.incrementK(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfAbsAcceleration())
                sign.append(sign[-1] * -1)
        return Variance

    def optimization_abs_acc_C(self, step_size, dof1, dof2, controlDOF):
        """Same as for the stiffness matrix K"""
        Variance = []
        sign = []
        sign.append(0)
        Variance.append(self.VarianceOfAbsAcceleration())

        sign.append(1)
        self.incrementC(step_size, dof1, dof2)
        Variance.append(self.VarianceOfAbsAcceleration())

        i = 0
        while len(sign) < 4 or np.sum(sign[-8:]) != 0:
            i += 1
            if i == 400:
                break
            print(Variance[-2][int(controlDOF - 1)])

            if Variance[-2][int(controlDOF - 1)] > Variance[-1][int(controlDOF - 1)]:
                self.incrementC(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfAbsAcceleration())
                sign.append(sign[-1])
            elif Variance[-2][int(controlDOF - 1)] < Variance[-1][int(controlDOF - 1)]:
                self.incrementC(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfAbsAcceleration())
                sign.append(sign[-1] * -1)
        return Variance

class Earthquake:

    def __init__(self, signal, time_step, unit='m/s2'):
        self.dt = time_step
        if unit == 'm/s2':
            self.signal = signal
        elif unit == 'g':
            self.signal = signal * 9.81
        self.length = len(signal)
        self.duration = self.length * self.dt
        self.time_range = np.arange(0, self.duration, self.dt)
        self.d_f = 1 / self.duration
        self.d_omega = self.d_f * 2 * np.pi
        self.fNiquist = self.length / 2 * self.d_f
        self.f_range = np.arange(0, self.fNiquist, self.d_f)
        self.omega_range = np.arange(0, self.fNiquist * 2 * np.pi, self.d_omega)
        self.fourier = np.fft.fft(self.signal)
        self.psd = abs(self.fourier)**2

    def interpolate(self, dt_new):
        new_time_range = np.arange(0, self.duration, dt_new)
        self.signal = np.interp(new_time_range, self.time_range, self.signal)
        self.dt = dt_new
        self.length = len(self.signal)
        self.duration = self.length * self.dt
        self.time_range = new_time_range
        self.d_f = 1 / self.duration
        self.d_omega = self.d_f * 2 * np.pi
        self.fNiquist = self.length / 2 * self.d_f
        self.f_range = np.arange(0, self.fNiquist, self.d_f)
        self.omega_range = np.arange(0, self.fNiquist * 2 * np.pi, self.d_omega)
        self.fourier = np.fft.fft(self.signal)
        self.psd = abs(self.fourier) ** 2

    def plot_signal(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(self.time_range, self.signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [m/s^2]')
        plt.show()
        return fig

    def plot_fourier_amplitude(self, herz=True):
        import matplotlib.pyplot as plt
        fig_fourier_amplitude = plt.figure()
        if herz is True:
            plt.plot(self.f_range, np.abs(self.fourier[0:len(self.f_range)]))
            plt.xlabel('Frequency [Hz]')
        if herz is False:
            plt.plot(self.omega_range, np.abs(self.fourier[0:len(self.omega_range)]))
            plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Fourier Spectrum [m/s^2]')
        plt.show()
        return fig_fourier_amplitude

    def plot_fourier_phase(self, herz=True):
        import matplotlib.pyplot as plt
        fig_fourier_phase = plt.figure()
        if herz is True:
            plt.plot(self.f_range, np.angle(self.fourier[0:len(self.f_range)]))
            plt.xlabel('Frequency [Hz]')
        if herz is False:
            plt.plot(self.omega_range, np.anlge(self.fourier[0:len(self.omega_range)]))
            plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Fourier Spectrum [m/s^2]')
        plt.show()
        return fig_fourier_phase

    def plot_psd(self, herz=True):
        import matplotlib.pyplot as plt
        fig_psd = plt.figure()
        if herz is True:
            plt.plot(self.f_range, self.psd[0:len(self.f_range)])
            plt.xlabel('Frequency [Hz]')
        if herz is False:
            plt.plot(self.omega_range, self.psd[0:len(self.omega_range)])
            plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Power Spectral Density [m^2/s^4]')
        plt.show()
        return fig_psd

    def response_spectrum(self, period_max=4, d_period=0.02, damping_zeta=0.05):
        self.period_range = np.arange(d_period, period_max, d_period)
        self.omega_resp_spec = 1/self.period_range * 2 * np.pi

        a = np.zeros([self.omega_resp_spec.size, self.length])
        v = np.zeros([self.omega_resp_spec.size, self.length])
        u = np.zeros([self.omega_resp_spec.size, self.length])

        # Sweep of Frequency
        for W in range(len(self.omega_resp_spec)):

            # Time Integration
            for i in range(self.length - 1):
                a[W, i + 1] = ((u[W, i] * self.omega_resp_spec[W] ** 2 + v[W, i]
                                * (2 * damping_zeta * self.omega_resp_spec[W] + self.omega_resp_spec[W] ** 2 * self.dt)
                                + a[W, i] * (damping_zeta * self.omega_resp_spec[W] * self.dt
                                + (self.omega_resp_spec[W] ** 2) / 4 * self.dt ** 2) - self.signal[i + 1])
                               / (- 1 - damping_zeta * self.omega_resp_spec[W] * self.dt
                                  - ((self.omega_resp_spec[W] ** 2) * self.dt ** 2) / 4))

                v[W, i + 1] = v[W, i] + (a[W, i] + a[W, i + 1]) / 2 * self.dt

                u[W, i + 1] = u[W, i] + v[W, i] * self.dt + (a[W, i] + a[W, i + 1]) / 4 * self.dt ** 2

        # Writing results
        self.Sd = np.max(np.absolute(u), axis=1)
        self.Sv = self.Sd * self.omega_resp_spec
        self.Sa = self.Sd * self.omega_resp_spec ** 2

    def plot_response_spectrum(self, type='acceleration'):
        import matplotlib.pyplot as plt
        fig_resp_spec = plt.figure()
        x = self.period_range
        if type == 'acceleration':
            y = self.Sa
            plt.ylabel('Pseudo Acceleration [m/s^2]')
        elif type == 'velocity':
            y = self.Sv
            plt.ylabel('Pseudo Velocity [m/s]')
        elif type == 'displacement':
            y = self.Sd
            plt.ylabel('Displacement [m]')
        plt.plot(x, y)
        plt.xlabel('Period [s]')
        plt.show()
        return fig_resp_spec
