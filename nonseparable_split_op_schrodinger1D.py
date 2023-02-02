import numpy as np
from scipy.fftpack import fft, ifft # Tools for fourier transform
from scipy import linalg

class NonseparableSplitOpSchrdoniger1D(object):
    """
    Implementation of the method ``Exponential Unitary Integrators for Nonseparable Quantum Hamiltonians’’ [arXiv:2211.08155]
    to propagate the 1d Schrodinger equation with the Hamiltonian

        H = k(p) + g(x) + [T, [T, V]],

    where T = T(p) and V = V(x).
    """
    def __init__(self, *, x_grid_dim, x_amplitude, k, g, T, V, dt, t=0, **kwargs):
        """
        :param x_grid_dim: the grid size
        :param x_amplitude: the maximum value of the coordinates
        :param g: the potential energy (as a function)
        :param f: the kinetic energy (as a function)
        :param T: the momentum function
        :param V: the coordinate function
        :param dt: time increment
        :param kwargs: ignored
        """
        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.k = k
        self.g = g
        self.V = V
        self.T = T
        self.t = t
        self.dt = dt

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        x = self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # generate momentum range as it corresponds to FFT frequencies
        p = self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros(self.x.size, dtype=complex)

        ################################################################################################################
        #
        # Pre-calculating the exponents
        #
        ################################################################################################################

        # Define the constants from the Chin's paper
        t2 = -np.cbrt(6)
        t1 = -t2
        v1 = 1 / t2 ** 2
        v2 = -0.5 * v1
        v0 = -2 * (v1 + v2)

        ε = -1j * np.cbrt(self.dt)

        self.minus_ones = (-1) ** np.arange(self.wavefunction.size)

        self.expV0 = np.exp(ε * v0 * V(x))
        self.expV1 = np.exp(ε * v1 * V(x))
        self.expV2 = np.exp(ε * v2 * V(x))
        self.expV2_minus = self.expV2 * self.minus_ones
        self.expg_minus_ones = np.exp(-1j * dt * g(x)) * self.minus_ones

        self.expT1 = np.exp(ε * t1 * T(p))
        self.expT2 = np.exp(ε * t2 * T(p))
        self.expk = np.exp(-1j * dt * k(p))

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in range(time_steps):

            # advance the wavefunction by dt
            self.single_step_propagation()

            # calculate the Ehrenfest theorems
            # self.get_ehrenfest()

        return self.wavefunction

    def single_step_propagation(self):
        """
        Perform a single step propagation of the wavefunction. The wavefunction is normalized.
        :return: self.wavefunction
        """
        self.wavefunction *= self.expV2_minus

        # going to the momentum representation
        self.wavefunction = fft(self.wavefunction, overwrite_x=True)
        self.wavefunction *= self.expT2
        # going back to the coordinate representation
        self.wavefunction = ifft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.expV1

        # going to the momentum representation
        self.wavefunction = fft(self.wavefunction, overwrite_x=True)
        self.wavefunction *= self.expT1
        # going back to the coordinate representation
        self.wavefunction = ifft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.expV0

        # going to the momentum representation
        self.wavefunction = fft(self.wavefunction, overwrite_x=True)
        self.wavefunction *= self.expT1
        # going back to the coordinate representation
        self.wavefunction = ifft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.expV1

        # going to the momentum representation
        self.wavefunction = fft(self.wavefunction, overwrite_x=True)
        self.wavefunction *= self.expT2
        # going back to the coordinate representation
        self.wavefunction = ifft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.expV2

        # going to the momentum representation
        self.wavefunction = fft(self.wavefunction, overwrite_x=True)
        self.wavefunction *= self.expk
        # going back to the coordinate representation
        self.wavefunction = ifft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.expg_minus_ones

        # make a time increment
        self.t += self.dt

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or function specifying the wave function
        :return: self
        """
        if isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            try:
                self.wavefunction[:] = wavefunc(self.x)
            except TypeError:
                raise ValueError("wavefunc must be either function or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self