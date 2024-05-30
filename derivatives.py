import numpy as np

def spectral_derivative(u, dx, order):
    N = len(u)
    k = np.fft.fftfreq(N, d=dx) * 2j * np.pi
    k = k ** order
    u_hat = np.fft.fft(u)
    du_hat = k * u_hat
    du = np.fft.ifft(du_hat)
    return np.real(du)
