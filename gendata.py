import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from GPy.kern import RBF
from jax import grad

### Nonlinear Elliptic PDE ###
m = 100
x = jnp.linspace(1, 10, m)
def u(x1):
    return 4*jnp.sin(2*x1)
def f(x1):
    return -1*grad(u)(x1)+0.01*u(x1)**3
field = jnp.array([u(x_val) for x_val in x])
result = jnp.array([f(x_val) for x_val in x])
plt.plot(field,x)
plt.plot(result,x)