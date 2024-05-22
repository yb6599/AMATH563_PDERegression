import numpy as np

def first_derivative(u, dx):
    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    du[0] = (u[1] - u[0]) / dx
    du[-1] = (u[-1] - u[-2]) / dx
    return du

def second_derivative(u, dx):
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    d2u[0] = (u[1] - 2 * u[0] + u[0]) / (dx**2)
    d2u[-1] = (u[-1] - 2 * u[-1] + u[-2]) / (dx**2)
    return d2u

