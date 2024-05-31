import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel
from sklearn.gaussian_process.kernels import RBF
from scipy.linalg import solve
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt

def kernel_spatial_derivatives(x1, x2, u_noisy, kernel, order=1, **kwargs):
    x1 = x1[:, None]
    x2 = x2[:, None]
    if kernel == 'rbf':
        length_scale = kwargs.get('l')
        rbf = RBF(length_scale=length_scale)
        U = rbf(x1, x2)
        N = U.shape[0]
        rbf_derivatives = {}
        if order >= 1:
            Ux =  -(x1 - x2) / (length_scale ** 2) * U
            rbf_derivatives['ux'] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = ((x1 - x2) ** 2 - length_scale ** 2) / (length_scale ** 4) * U
            rbf_derivatives['uxx'] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = (-(x1 - x2) ** 3 / length_scale**3 + 3 * (x1 - x2) / length_scale**2) * U
            rbf_derivatives['uxxx'] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = ((x1 - x2) ** 4 / length_scale**4 - 6 * (x1 - x2) ** 2 / length_scale**3 + 3 / length_scale**2) * U
            rbf_derivatives['uxxxx'] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        return rbf_derivatives

    if kernel == 'poly':
        c = kwargs.get('coef0')
        d = kwargs.get('degree')
        g = kwargs.get('gamma')
        U = polynomial_kernel(X=x1, Y=x2, degree=d, gamma=g, coef0=c)
        N = U.shape[0]
        poly_derivatives = {}
        if order >= 1:
            Ux =  d * (g * np.dot(x1, x2.T) + c) ** (d-1) * x2
            poly_derivatives['ux'] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = d * (d-1) * (g * np.dot(x1, x2.T) + c) ** (d - 2) * x2**2
            poly_derivatives['uxx'] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = d * (d-1) * (d-2) * (g * np.dot(x1, x2.T) + c) ** (d - 3) * x2**3
            poly_derivatives['uxxx'] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = d * (d-1) * (d-2) * (d-3) * (g * np.dot(x1, x2.T) + c) ** (d - 4) * x2**4
            poly_derivatives['uxxxx'] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        return poly_derivatives

    if kernel == 'sigmoid':
        g = kwargs.get('gamma')
        c = kwargs.get('coef0')
        U = sigmoid_kernel(X=x1, Y=x2, gamma=g, coef0 = c)
        N = U.shape[0]
        sigmoid_derivatives = {}
        tanh_value = np.tanh(g * np.dot(x1, x2.T) + c)
        sech2_value = 1 - tanh_value**2
        if order >= 1:
            Ux = g * x2 * sech2_value
            sigmoid_derivatives['ux'] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = -2 * g**2 * x2**2 * tanh_value * sech2_value
            sigmoid_derivatives['uxx'] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = g**3 * x2**3 * (2 * sech2_value**2 - 6 * tanh_value**2 * sech2_value)
            sigmoid_derivatives['uxxx'] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = g**4 * x2**4 * (-8 * tanh_value * sech2_value**2 + 24 * tanh_value**3 * sech2_value - 6 * sech2_value**3)
            sigmoid_derivatives['uxxxx'] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)        
        return sigmoid_derivatives


def functional_form_PDE(s, s_test, ut, kernel: str, **kwargs):
    if kernel == 'rbf':
        length_scale = kwargs.get('l')
        K = RBF(length_scale=length_scale)
        P = K(s_test, s) @ solve((K(s, s) + 0.01 * np.eye(s.shape[0])), ut)
        return P
    if kernel == 'poly':
        c = kwargs.get('coef0')
        d = kwargs.get('degree')
        g = kwargs.get('gamma')
        K = polynomial_kernel(X=s, Y=s, degree=d, gamma=g, coef0=c)
        K_test = polynomial_kernel(X=s_test, Y=s, degree=d, gamma=g, coef0=c)
        P = K_test @ solve((K + 0.01 * np.eye(s.shape[0])), ut)
        return P
    if kernel == 'sigmoid':
        g = kwargs.get('gamma')
        c = kwargs.get('coef0')
        K = sigmoid_kernel(X=s, Y=s, gamma=g, coef0 = c)
        K_test = sigmoid_kernel(X=s_test, Y=s, gamma=g, coef0 = c)
        P = K_test @ solve((K + 0.01 * np.eye(s.shape[0])), ut)
        return P

def kernel_smoothing(u_noisy, x, kernel: str, reg=0.1, **kwargs):
    x = x[:, None]
    Nx = len(x)
    if kernel == 'rbf':
        length_scale = kwargs.get('l')
        rbf = RBF(length_scale=length_scale)
        U = rbf(x)
    elif kernel == 'poly':
        coef0 = kwargs.get('coef0')
        degree = kwargs.get('degree')
        gamma = kwargs.get('gamma')
        U = polynomial_kernel(X=x,degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == 'sigmoid':
        gamma = kwargs.get('gamma')
        coef0 = kwargs.get('coef0')
        U = sigmoid_kernel(X=x, gamma=gamma, coef0 = coef0)

    u_smooth = U @ solve((U + reg * np.eye(Nx)), u_noisy)
    return u_smooth

def plot_comparison(
    x, t, u, u_noisy, u_smooth
):
    y = np.linspace(x[0], x[-1], len(x) + 1)
    x = np.linspace(t[0], t[-1], len(t) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].pcolormesh(x, y, u)
    axs[0].set(title="True Data")
    axs[1].pcolormesh(x, y, u_noisy)
    axs[1].set(title="Noisy Data")
    axs[2].pcolormesh(x, y, u_smooth)
    axs[2].set(title="Smoothed Data")
    plt.show()
    return fig, axs