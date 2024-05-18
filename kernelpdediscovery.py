# %%
from jax import grad
from scipy.integrate import solve_ivp
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import scipy.ndimage
import sklearn
import pysindy as ps
# %% ODE Lotka Volterra
def lv(t ,z, alpha, beta, gamma, delta):
    x, y = z
    xt = alpha*x-beta*x*y
    yt = delta*x*y-gamma*y
    return [xt, yt]

x0 = 10
y0 = 5
z0 = [x0, y0]
alpha = 1.1
beta = 0.4
gamma = 0.4
delta = 1.1
t_eval = np.linspace(0, 50, 500)
t_span = (t_eval[0], t_eval[-1])
zf = solve_ivp(lv, t_span, z0, t_eval=t_eval, args=(alpha, beta, gamma, delta))
tf = zf.t
xf = zf.y[0]
yf = zf.y[1]

plt.figure()
plt.plot(tf, xf)
plt.plot(tf, yf)
plt.show()
# %% PDE Heat Diffusion Equation
a = 0.01
L = 1
Nx = 200
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
u0 = jnp.exp((-jnp.sin(jnp.pi * x)**2)/2)
t_span = (0, 10)
t_eval = np.linspace(0, 10, 200)

def diffusion(t, u, a, dx):
    u[0] = 0
    u[-1] = 0
    uxx = ps.differentiation.SpectralDerivative(d=2)._differentiate(u, dx)
    # ux = grad(lambda u: u)
    # uxx = grad(ux)
    ut = a * uxx
    return np.array(ut)
z_diff = []
z_diff.append(solve_ivp(diffusion, t_span=t_span, y0 = u0, args=(a, dx), t_eval=t_eval).y.T)
z_diff = np.squeeze(z_diff)
# plt.figure()
# plt.imshow(z_diff.T)
m, n = z_diff.shape
z_diff_noisy = z_diff + np.random.uniform(size=(m, n))
# plt.figur/e()
# plt.imshow(z_diff_noisy.T)
# %% Nonlinear PDE
def nonlinear_pde(t, u, a, dx):
    u[0] = 0
    u[1] = 0
    ux = ps.differentiation.SpectralDerivative(d=1)._differentiate(u, dx)
    ut = - ux + a * u**3
    return np.array(ut)

z_nlpde = []
z_nlpde.append(solve_ivp(nonlinear_pde, t_span=t_span, y0 = u0, args=(a, dx), t_eval=t_eval).y.T)
z_nlpde = np.squeeze(z_nlpde)
plt.imshow(z_nlpde.T)
# %% Smoothing
z_diff_smooth = scipy.ndimage.gaussian_filter(z_diff_noisy, sigma=10)
# plt.imshow(z_diff_smooth.T)
figure, axis = plt.subplots(3, 1)
figure.add_subplot(3, 1, 1)
plt.imshow(z_diff.T)
figure.add_subplot(3, 1, 2)
plt.imshow(z_diff_noisy.T)
figure.add_subplot(3, 1, 3)
plt.imshow(z_diff_smooth.T)

# %% Kernels and Functional Form of PDE
