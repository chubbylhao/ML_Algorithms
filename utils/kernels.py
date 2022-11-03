import numpy as np


def linear_kernel(**kwargs):
    """ 线性核 """
    def f(x1, x2):
        return np.inner(x1, x2)
    return f


def polynomial_kernel(power, coeff, **kwargs):
    """ 多项式核 """
    def f(x1, x2):
        return (np.inner(x1, x2) + coeff) ** power
    return f


def rbf_kernel(gamma, **kwargs):
    """ 高斯核/rbf核 """
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(- gamma * distance)
    return f
