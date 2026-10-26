# -*- coding: utf-8 -*-
"""
Часть 1.2 / 2.1: составная трёхточечная (интерполяционная) квадратурная
формула типа Ньютона-Котеса с весом p(x) = (b-x)^(-beta).

На каждом частичном отрезке [z_{i-1}, z_i] берём 3 равноотстоящих узла
x1=z_{i-1}, x2=(z_{i-1}+z_i)/2, x3=z_i и строим интерполяционную формулу

    int_{z_{i-1}}^{z_i} p(x) f(x) dx ~= A1 f(x1) + A2 f(x2) + A3 f(x3)

Коэффициенты Aj находятся из системы (13):
    sum_j Aj xj^s = mu_s,   s = 0, 1, 2
(mu_s -- аналитические моменты веса на подотрезке, см. moments.py).

АСТ малой формулы: n-1 = 2 (три узла => точна для многочленов степени <=2).
"""
import numpy as np
from .moments import moment_about

ACT = 2  # алгебраическая степень точности малой 3-точечной ИКФ


def _small_formula_nodes_weights(z_lo, z_hi):
    # локальные безразмерные координаты tau=(x-center)/(h/2) in [-1,1]
    # для хорошей обусловленности матрицы Вандермонда при малых h
    center = 0.5 * (z_lo + z_hi)
    half_h = 0.5 * (z_hi - z_lo)
    tau1, tau2, tau3 = -1.0, 0.0, 1.0
    mus = [moment_about(s, center, z_lo, z_hi) for s in range(3)]
    nu = [mus[s] / half_h ** s for s in range(3)]
    V = np.array([[1, 1, 1],
                  [tau1, tau2, tau3],
                  [tau1 ** 2, tau2 ** 2, tau3 ** 2]], dtype=float)
    A = np.linalg.solve(V, np.array(nu, dtype=float))
    nodes = center + half_h * np.array([tau1, tau2, tau3])
    return nodes, A


def composite_newton_cotes3(f, a, b, n):
    """Составная трёхточечная КФ Ньютона-Котеса с весом p(x)=(b-x)^-beta.
    n -- число частичных отрезков на [a,b]."""
    z = np.linspace(a, b, n + 1)
    total = 0.0
    for i in range(n):
        nodes, A = _small_formula_nodes_weights(z[i], z[i + 1])
        total += np.sum(A * f(nodes))
    return total
