import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel
from sklearn.gaussian_process.kernels import RBF
from scipy.linalg import solve
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt


def add_noise(u, noise_percentage, seed=42):
    np.random.seed(seed)
    u_std = np.std(u)
    noise_std = (noise_percentage / 100) * u_std
    u_noisy = u + np.random.normal(scale=noise_std, size=u.shape)
    return u_noisy


def calculate_ut_and_s(system, x, u_smooth, derivatives):
    X, Y = np.meshgrid(x, x)
    s = np.column_stack([X, u_smooth, derivatives["ux"], derivatives["uxx"]])
    if system == "Diffusion":
        ut = -derivatives["uxx"]
    elif system == "Burgers":
        ut = -0.1 * derivatives["uxx"] + u_smooth * derivatives["ux"]
    elif system == "Kuramoto-Sivashinsky":
        s = np.column_stack(
            [
                X,
                u_smooth,
                derivatives["ux"],
                derivatives["uxx"],
                derivatives["uxxx"],
                derivatives["uxxxx"],
            ]
        )
        ut = derivatives["uxx"] + derivatives["uxxxx"] + derivatives["ux"] * u_smooth
    else:
        raise ValueError("Invalid system specified")
    return ut, s


def find_best_parameters(u, u_noisy, x, kernel_type, param_grid):
    mse_list = []
    for params in param_grid:
        u_smooth = kernel_smoothing(u_noisy, x, kernel=kernel_type, **params)
        mse = mean_squared_error(u, u_smooth)
        mse_list.append({**params, "mse": mse})
    best_params = min(mse_list, key=lambda x: x["mse"])
    return best_params, mse_list


def generate_param_grid(kernel_type):
    if kernel_type == "rbf":
        return [{"l": l} for l in [0.01, 0.1, 1, 5, 10]]
    elif kernel_type == "poly":
        return [
            {"coef0": c, "degree": d, "gamma": gamma}
            for gamma in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            for c in [0.25, 0.5, 1, 5]
            for d in [1, 2, 3, 4, 5]
        ]
    elif kernel_type == "sigmoid":
        return [
            {"coef0": c, "gamma": gamma}
            for gamma in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            for c in [0.25, 0.5, 1, 5]
        ]


def kernel_spatial_derivatives(x1, x2, u_noisy, kernel, order=1, **kwargs):
    x1 = x1[:, None]
    x2 = x2[:, None]
    if kernel == "rbf":
        length_scale = kwargs.get("l")
        rbf = RBF(length_scale=length_scale)
        U = rbf(x1, x2)
        N = U.shape[0]
        rbf_derivatives = {}
        if order >= 1:
            Ux = -(x1 - x2) / (length_scale**2) * U
            rbf_derivatives["ux"] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = ((x1 - x2) ** 2 - length_scale**2) / (length_scale**4) * U
            rbf_derivatives["uxx"] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = (
                -((x1 - x2) ** 3) / length_scale**3 + 3 * (x1 - x2) / length_scale**2
            ) * U
            rbf_derivatives["uxxx"] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = (
                (x1 - x2) ** 4 / length_scale**4
                - 6 * (x1 - x2) ** 2 / length_scale**3
                + 3 / length_scale**2
            ) * U
            rbf_derivatives["uxxxx"] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        return rbf_derivatives

    if kernel == "poly":
        c = kwargs.get("coef0")
        d = kwargs.get("degree")
        g = kwargs.get("gamma")
        U = polynomial_kernel(X=x1, Y=x2, degree=d, gamma=g, coef0=c)
        N = U.shape[0]
        poly_derivatives = {}
        if order >= 1:
            Ux = d * (g * np.dot(x1, x2.T) + c) ** (d - 1) * x2
            poly_derivatives["ux"] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = d * (d - 1) * (g * np.dot(x1, x2.T) + c) ** (d - 2) * x2**2
            poly_derivatives["uxx"] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = d * (d - 1) * (d - 2) * (g * np.dot(x1, x2.T) + c) ** (d - 3) * x2**3
            poly_derivatives["uxxx"] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = (
                d
                * (d - 1)
                * (d - 2)
                * (d - 3)
                * (g * np.dot(x1, x2.T) + c) ** (d - 4)
                * x2**4
            )
            poly_derivatives["uxxxx"] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        return poly_derivatives

    if kernel == "sigmoid":
        g = kwargs.get("gamma")
        c = kwargs.get("coef0")
        U = sigmoid_kernel(X=x1, Y=x2, gamma=g, coef0=c)
        N = U.shape[0]
        sigmoid_derivatives = {}
        tanh_value = np.tanh(g * np.dot(x1, x2.T) + c)
        sech2_value = 1 - tanh_value**2
        if order >= 1:
            Ux = g * x2 * sech2_value
            sigmoid_derivatives["ux"] = Ux @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 2:
            Uxx = -2 * g**2 * x2**2 * tanh_value * sech2_value
            sigmoid_derivatives["uxx"] = Uxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 3:
            Uxxx = g**3 * x2**3 * (2 * sech2_value**2 - 6 * tanh_value**2 * sech2_value)
            sigmoid_derivatives["uxxx"] = Uxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        if order >= 4:
            Uxxxx = (
                g**4
                * x2**4
                * (
                    -8 * tanh_value * sech2_value**2
                    + 24 * tanh_value**3 * sech2_value
                    - 6 * sech2_value**3
                )
            )
            sigmoid_derivatives["uxxxx"] = Uxxxx @ solve((U + 0.1 * np.eye(N)), u_noisy)
        return sigmoid_derivatives


def functional_form_PDE(s, s_test, ut, kernel: str, **kwargs):
    if kernel == "rbf":
        length_scale = kwargs.get("l")
        K = RBF(length_scale=length_scale)
        P = K(s_test, s) @ solve((K(s, s) + 0.01 * np.eye(s.shape[0])), ut)
        return P
    if kernel == "poly":
        c = kwargs.get("coef0")
        d = kwargs.get("degree")
        g = kwargs.get("gamma")
        K = polynomial_kernel(X=s, Y=s, degree=d, gamma=g, coef0=c)
        K_test = polynomial_kernel(X=s_test, Y=s, degree=d, gamma=g, coef0=c)
        P = K_test @ solve((K + 0.01 * np.eye(s.shape[0])), ut)
        return P
    if kernel == "sigmoid":
        g = kwargs.get("gamma")
        c = kwargs.get("coef0")
        K = sigmoid_kernel(X=s, Y=s, gamma=g, coef0=c)
        K_test = sigmoid_kernel(X=s_test, Y=s, gamma=g, coef0=c)
        P = K_test @ solve((K + 0.01 * np.eye(s.shape[0])), ut)
        return P


def kernel_smoothing(u_noisy, x, kernel: str, reg=0.1, **kwargs):
    x = x[:, None]
    Nx = len(x)
    if kernel == "rbf":
        length_scale = kwargs.get("l")
        rbf = RBF(length_scale=length_scale)
        U = rbf(x)
    elif kernel == "poly":
        coef0 = kwargs.get("coef0")
        degree = kwargs.get("degree")
        gamma = kwargs.get("gamma")
        U = polynomial_kernel(X=x, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "sigmoid":
        gamma = kwargs.get("gamma")
        coef0 = kwargs.get("coef0")
        U = sigmoid_kernel(X=x, gamma=gamma, coef0=coef0)

    u_smooth = U @ solve((U + reg * np.eye(Nx)), u_noisy)
    return u_smooth


def load_data(system):
    if system == "Burgers":
        u = np.load("data/Burgers.npy")
        x = np.linspace(-8, 8, 1000)
        t = np.linspace(0, 16, 1000)
    elif system == "Diffusion":
        u = np.load("data/Diffusion.npy")
        x = np.linspace(-8, 8, 1000)
        t = np.linspace(0, 16, 1000)
    elif system == "Kuramoto-Sivashinsky":
        u = np.load("data/Kuramoto-Sivashinsky.npy")
        x = np.linspace(0, 100, 1000)
        t = np.linspace(0, 100, 1000)
    else:
        raise ValueError("Invalid system specified")
    u = np.squeeze(u)
    return u, x, t
