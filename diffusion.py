import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import kernels
from scipy.linalg import solve
import jax.numpy as jnp
import jax

def calculate_mse(f, f_new):
    return mean_squared_error(f, f_new)

def rbf_kernel_derivative(x1, x2, length_scale, order=1):
    rbf = kernels.RBF(length_scale=length_scale)
    K = rbf(x1, x2)
    if order == 1:
        return -(x1[:, None, :] - x2[None, :, :]) / (length_scale ** 2) * K[:, :, None]
    elif order == 2:
        return ((x1[:, None, :] - x2[None, :, :]) ** 2 - length_scale ** 2) / (length_scale ** 4) * K[:, :, None]
    else:
        raise ValueError("Higher order derivatives not implemented")

def P(S, s, P_coef, rbf):
    K_sS = rbf(S, s)
    return K_sS @ P_coef

def smooth_and_evaluate_rbf(u_diff_noisy, x, l, Nx):
    rbf = kernels.RBF(length_scale=l)
    u_rbf = rbf(x[:, None], x[:, None])
    u_diff_smooth = u_rbf @ solve((u_rbf + 0.1 * np.eye(Nx)), u_diff_noisy)
    
    f = np.cos(2 * np.pi * x)
    
    u_kx = rbf_kernel_derivative(x[:, None], x[:, None], length_scale=l, order=1).squeeze() @ solve(
        (u_rbf + 0.1 * np.eye(Nx)), u_diff_smooth
    )
    u_kxx = rbf_kernel_derivative(x[:, None], x[:, None], length_scale=l, order=2).squeeze() @ solve(
        (u_rbf + 0.1 * np.eye(Nx)), u_diff_smooth
    )
    s = np.hstack([u_diff_smooth, u_kx, u_kxx])
    K = rbf(s, s)
    P_coef = solve((K + 0.01 * np.eye(Nx)), f)
    f_new = P(s, s, P_coef, rbf)
    
    return calculate_mse(f, f_new), u_diff_smooth

def smooth_and_evaluate_poly(u_diff_noisy, x, deg, coef, gam, Nx):
    u_poly = sklearn.metrics.pairwise.polynomial_kernel(x[:, np.newaxis], x[:, np.newaxis], degree=deg, coef0=coef, gamma=gam)
    u_diff_smooth = u_poly @ solve((u_poly + 0.1 * np.eye(Nx)), u_diff_noisy)

    f = np.cos(2 * np.pi * x)
    sk = sklearn.metrics.pairwise.polynomial_kernel(x[:, np.newaxis], x[:, np.newaxis], degree=2, coef0=1, gamma=1)

    u_kx = jnp.gradient(sk, axis=0) @ solve((sk + 0.1 * np.eye(Nx)), u_diff_smooth)
    u_kxx = jnp.gradient(jnp.gradient(sk, axis=0), axis=0) @ solve((sk + 0.1 * np.eye(Nx)), u_diff_smooth)

    s = np.hstack([u_diff_smooth, u_kx, u_kxx])

    K = sklearn.metrics.pairwise.polynomial_kernel(s, s, degree=deg, coef0=coef, gamma=gam)
    P_coef = solve((K + 0.01 * np.eye(Nx)), f)
    K_sS = sklearn.metrics.pairwise.polynomial_kernel(X=s, Y=s, degree=deg, coef0=coef, gamma=gam)
    f_new = K_sS @ P_coef

    return calculate_mse(f, f_new), u_diff_smooth

def smooth_and_evaluate_sigmoid(u_diff_noisy, x, l, Nx):
    rbf = kernels.RBF(length_scale=l)
    u_rbf = rbf(x[:, None], x[:, None])
    u_diff_smooth = u_rbf @ solve((u_rbf + 0.1 * np.eye(Nx)), u_diff_noisy)
    
    f = np.cos(2 * np.pi * x)
    
    u_kx = rbf_kernel_derivative(x[:, None], x[:, None], length_scale=l, order=1).squeeze() @ solve(
        (u_rbf + 0.1 * np.eye(Nx)), u_diff_smooth
    )
    u_kxx = rbf_kernel_derivative(x[:, None], x[:, None], length_scale=l, order=2).squeeze() @ solve(
        (u_rbf + 0.1 * np.eye(Nx)), u_diff_smooth
    )
    s = np.hstack([u_diff_smooth, u_kx, u_kxx])
    K = rbf(s, s)
    P_coef = solve((K + 0.01 * np.eye(Nx)), f)
    f_new = P(s, s, P_coef, rbf)
    
    return calculate_mse(f, f_new), u_diff_smooth