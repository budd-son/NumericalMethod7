# -*- coding: utf-8 -*-
"""
Часть 1.2 / 2.2: составная трёхточечная квадратурная формула
наивысшей алгебраической степени точности (типа Гаусса) с весом
p(x) = (b-x)^(-beta).

На каждом частичном отрезке [z_{i-1}, z_i] (n=3 узла):

1) считаем моменты mu_0..mu_5 веса на подотрезке (аналитически, moments.py);
2) находим коэффициенты a0,a1,a2 узлового многочлена
       omega(x) = x^3 + a2 x^2 + a1 x + a0
   из СЛАУ (19):  sum_{j=0}^{2} a_j * mu_{j+s} = -mu_{3+s},  s=0,1,2;
3) находим корни omega(x)=0 -- узлы x1,x2,x3 -- АНАЛИТИЧЕСКИ методом
   Кардано (тригонометрическая форма, cardano.py);
4) находим веса A1,A2,A3 из СЛАУ (21): sum_j Aj xj^s = mu_s, s=0,1,2.

АСТ малой формулы: 2n-1 = 5.
"""
import numpy as np
from .moments import moment_about
from .cardano import solve_cubic_cardano

ACT = 5  # алгебраическая степень точности малой 3-точечной КФ Гаусса


def _small_formula_nodes_weights(z_lo, z_hi):
    # Локальные безразмерные координаты tau = (x-center)/(h/2) in [-1,1]
    # (устраняют как большой сдвиг x~2, так и разбегание моментов по
    # порядку величины ~h^s при малых h -- источники плохой
    # обусловленности систем (19),(21)).
    center = 0.5 * (z_lo + z_hi)
    half_h = 0.5 * (z_hi - z_lo)
    mus = [moment_about(s, center, z_lo, z_hi) for s in range(6)]  # mu'_0..mu'_5
    nu = [mus[s] / half_h ** s for s in range(6)]  # моменты в переменной tau

    # система (19) для узлового многочлена omega(tau)=tau^3+a2 tau^2+a1 tau+a0
    M = np.array([
        [nu[0], nu[1], nu[2]],
        [nu[1], nu[2], nu[3]],
        [nu[2], nu[3], nu[4]],
    ], dtype=float)
    rhs = -np.array([nu[3], nu[4], nu[5]], dtype=float)
    a0, a1, a2 = np.linalg.solve(M, rhs)

    # аналитическое решение кубического уравнения методом Кардано (в tau)
    tau_nodes = solve_cubic_cardano(1.0, a2, a1, a0)
    tau_nodes = np.array(sorted(tau_nodes))
    nodes = center + half_h * tau_nodes

    # проверка: узлы должны лежать внутри отрезка
    if not (np.all(nodes > z_lo - 1e-9) and np.all(nodes < z_hi + 1e-9)):
        raise RuntimeError(f"Узлы Гаусса {nodes} вне отрезка [{z_lo},{z_hi}]")

    # веса из системы (21) при s=0,1,2 (в локальных координатах tau);
    # т.к. sum_k A_k tau_k^s = nu_s = mu_s/half_h^s <=> sum_k A_k t_k^s = mu_s,
    # найденные A совпадают с искомыми весами в исходных координатах x.
    tau1, tau2, tau3 = tau_nodes
    V = np.array([[1, 1, 1],
                  [tau1, tau2, tau3],
                  [tau1 ** 2, tau2 ** 2, tau3 ** 2]], dtype=float)
    A = np.linalg.solve(V, np.array(nu[:3], dtype=float))
    return nodes, A


def composite_gauss3(f, a, b, n):
    """Составная трёхточечная КФ типа Гаусса с весом p(x)=(b-x)^-beta.
    n -- число частичных отрезков на [a,b]."""
    z = np.linspace(a, b, n + 1)
    total = 0.0
    for i in range(n):
        nodes, A = _small_formula_nodes_weights(z[i], z[i + 1])
        total += np.sum(A * f(nodes))
    return total
