# %%
from derivatives import (
    first_derivative,
    second_derivative
    )
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt 
import numpy as np
import sklearn

l = 0.1
# %% ODE Lotka Volterra
# def lv(t ,z, alpha, beta, gamma, delta):
#     x, y = z
#     xt = alpha*x-beta*x*y
#     yt = delta*x*y-gamma*y
#     return [xt, yt]

# x0 = 10
# y0 = 5
# z0 = [x0, y0]
# alpha = 1.1
# beta = 0.4
# gamma = 0.4
# delta = 1.1
# t_eval = np.linspace(0, 50, 500)
# t_span = (t_eval[0], t_eval[-1])
# zf = solve_ivp(lv, t_span, z0, t_eval=t_eval, args=(alpha, beta, gamma, delta))
# tf = zf.t
# xf = zf.y[0]
# yf = zf.y[1]

# plt.figure()
# plt.plot(tf, xf)
# plt.plot(tf, yf)
# plt.show()
# %% Nonlinear PDE
# def nonlinear_pde(t, u, a, dx):
#     u[0] = 0
#     u[1] = 0
#     ux = first_derivative(u, dx)
#     ut = - ux + a * u**3
#     return np.array(ut)

# z_nlpde = []
# z_nlpde.append(solve_ivp(nonlinear_pde, t_span=t_span, y0 = u0, args=(a, dx), t_eval=t_eval).y.T)
# z_nlpde = np.squeeze(z_nlpde)
# plt.imshow(z_nlpde.T)

# %% Heat Diffusion
# %% PDE Heat Diffusion Equation
a = 0.01
l = 0.1
L = 1
Nx = 200
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
u0 = np.exp((-np.sin(np.pi * x)**2)/2)
t_span = (0, 10)
t_eval = np.linspace(0, 10, 200)

def diffusion(t, u, a, dx):
    u[0] = 0
    u[-1] = 0
    uxx = second_derivative(u, dx)
    ut = a * uxx
    return np.array(ut)

u_diff = []
u_diff.append(solve_ivp(diffusion, t_span=t_span, y0 = u0, args=(a, dx), t_eval=t_eval).y.T)
u_diff = np.squeeze(u_diff)

m, n = u_diff.shape
u_diff_noisy_normal = u_diff + np.random.normal(scale=0.1, size=(m, n))
rbf = kernels.RBF(length_scale=l)
x = x[:, None]
# u_rbf = sklearn.metrics.pairwise.polynomial_kernel(x)
u_rbf = rbf(x, x)
u_diff_smooth_normal = u_rbf @ solve((u_rbf + 0.1*np.eye(Nx)), u_diff_noisy_normal)
plt.figure()
fig, axis = plt.subplots(1, 3, figsize=(15, 5))
axis[0].imshow(u_diff.T)
axis[0].set_title('True Data')
axis[1].imshow(u_diff_noisy_normal.T)
axis[1].set_title('Gaussian Noisy Data')
axis[2].imshow(u_diff_smooth_normal.T)
axis[2].set_title('Smoothed Data')
plt.show()

def rbf_kernel_derivative(x1, x2, length_scale=l, order=1):
    K = rbf(x1, x2)
    if order == 1:
        return -(x1[:, None, :] - x2[None, :, :]) / (length_scale ** 2) * K[:, :, None]
    elif order == 2:
        return ((x1[:, None, :] - x2[None, :, :]) ** 2 - length_scale ** 2) / (length_scale ** 4) * K[:, :, None]
    else:
        raise ValueError("Higher order derivatives not implemented")
    
def P(S, s, P_coef):
    K_sS = rbf(S, s)
    return K_sS @ P_coef

f = np.cos(2 * np.pi * x)
# f = np.zeros_like(x)
SS = np.random.random((200, 600))

u_kx_normal = rbf_kernel_derivative(x, x, order=1).squeeze() @ solve(
    (u_rbf + 0.1*np.eye(Nx)), u_diff_smooth_normal
    )
u_kxx_normal = rbf_kernel_derivative(x, x, order=2).squeeze() @ solve(
    (u_rbf + 0.1*np.eye(Nx)), u_diff_smooth_normal
    )
s_normal = np.hstack([u_diff_smooth_normal, u_kx_normal, u_kxx_normal])
K_normal = rbf(s_normal,s_normal)
P_coef_normal = solve((K_normal + 0.01 * np.eye(Nx)), f)
f_new_normal = P(s_normal, s_normal, P_coef_normal)
plt.figure()
plt.title("normal")
plt.plot(f)
plt.plot(f_new_normal)

np.array_equal(f, f_new_normal)