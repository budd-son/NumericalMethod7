# -*- coding: utf-8 -*-
"""
Эталонное ("точное") значение интегралов -- вычисляется высокоточным
численным интегрированием (используется ТОЛЬКО для построения графиков
погрешности, а не как часть исследуемых квадратурных методов).
"""
from scipy import integrate
from .variant import a, b, alpha, beta, f


def exact_unweighted():
    """J = int_a^b f(x) dx  (p(x) = 1, часть 1.1)."""
    val, err = integrate.quad(f, a, b, epsabs=1e-14, epsrel=1e-14)
    return val


def exact_weighted():
    """J = int_a^b f(x) (x-a)^-alpha (b-x)^-beta dx  (часть 1.2 / 2)."""
    val, err = integrate.quad(f, a, b, weight='alg', wvar=(-alpha, -beta),
                               epsabs=1e-14, epsrel=1e-14)
    return val
