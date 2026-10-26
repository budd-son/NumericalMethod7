# -*- coding: utf-8 -*-
"""
Данные варианта №8:

    f(x) = 3.7*cos(1.5x)*exp(-4x/3) + 2.4*sin(4.5x)*exp(2x/3) + 4
    a = 1.8, b = 2.3
    alpha = 0, beta = 3/5

    p(x) = (x-a)^(-alpha) * (b-x)^(-beta)   -- весовая функция
    F(x) = p(x) * f(x)
"""
import numpy as np

a = 1.1
b = 2.3
alpha = 4.0/5.0
beta = 0.0


def f(x):
    """Гладкая часть подынтегральной функции f(x)."""
    x = np.asarray(x, dtype=float)
    return (3.5 * np.cos(0.7 * x) * np.exp(-5.0 * x / 3.0)
            + 2.4 * np.sin(5.5 * x) * np.exp(-3.0 * x / 4.0)
            + 5.0)

def p(x):
    """Весовая функция p(x) = (x-a)^(-alpha) * (b-x)^(-beta)."""
    x = np.asarray(x, dtype=float)
    res = np.ones_like(x, dtype=float)
    if alpha != 0.0:
        res = res * (x - a) ** (-alpha)
    if beta != 0.0:
        res = res * (b - x) ** (-beta)
    return res


def F(x):
    """Полная подынтегральная функция F(x) = p(x)*f(x)."""
    return p(x) * f(x)
