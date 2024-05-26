import numpy as np
import numdifftools as nd

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


def third_derivative(u, dx):
    d3u = np.zeros_like(u)
    d3u[2:-2] = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx**3)
    d3u[0] = (u[2] - 2 * u[1] + 2 * u[0] - u[0]) / (2 * dx**3)
    d3u[1] = (u[3] - 2 * u[2] + 2 * u[0] - u[0]) / (2 * dx**3)
    d3u[-2] = (u[-1] - 2 * u[-1] + 2 * u[-3] - u[-4]) / (2 * dx**3)
    d3u[-1] = (u[-1] - 2 * u[-1] + 2 * u[-2] - u[-3]) / (2 * dx**3)
    return d3u

def fourth_derivative(u, dx):
    d4u = np.zeros_like(u)
    d4u[2:-2] = (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4]) / (dx**4)
    d4u[0] = (u[2] - 4 * u[1] + 6 * u[0] - 4 * u[0] + u[0]) / (dx**4)
    d4u[1] = (u[3] - 4 * u[2] + 6 * u[1] - 4 * u[0] + u[0]) / (dx**4)
    d4u[-2] = (u[-1] - 4 * u[-2] + 6 * u[-3] - 4 * u[-4] + u[-5]) / (dx**4)
    d4u[-1] = (u[-1] - 4 * u[-2] + 6 * u[-3] - 4 * u[-4] + u[-5]) / (dx**4)
    return d4u

# def first_derivative(u, dx):
#     return nd.Derivative(lambda x: np.interp(x, np.arange(len(u)), u), n=1)(np.arange(len(u)) * dx)

# def second_derivative(u, dx):
#     return nd.Derivative(lambda x: np.interp(x, np.arange(len(u)), u), n=2)(np.arange(len(u)) * dx)

# def third_derivative(u, dx):
#     return nd.Derivative(lambda x: np.interp(x, np.arange(len(u)), u), n=3)(np.arange(len(u)) * dx)

# def fourth_derivative(u, dx):
#     return nd.Derivative(lambda x: np.interp(x, np.arange(len(u)), u), n=4)(np.arange(len(u)) * dx)

