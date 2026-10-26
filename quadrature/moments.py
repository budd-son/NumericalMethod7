# -*- coding: utf-8 -*-
"""
Аналитическое вычисление моментов весовой функции
    p(x) = (b - x)^(-beta)          (alpha = 0 в нашем варианте)
на произвольном подотрезке [z_lo, z_hi] через замену переменных t = b - x
(см. "Вычисление моментов весовой функции через замену переменных").

    mu_s(z_lo, z_hi) = int_{z_lo}^{z_hi} x^s (b-x)^{-beta} dx

Замена t = b - x, x = b - t, dx = -dt:
    mu_s = int_{t=u_lo}^{u_hi} (b-t)^s * t^{-beta} dt,
где u_lo = b - z_hi,  u_hi = b - z_lo   (u_lo < u_hi, т.к. z_lo < z_hi)

Раскладываем (b-t)^s по биному Ньютона:
    (b-t)^s = sum_{k=0}^{s} C(s,k) b^{s-k} (-1)^k t^k

и интегрируем почленно (t^{k-beta} имеет первообразную t^{k-beta+1}/(k-beta+1),
т.к. k - beta + 1 != 0 при beta не целом):

    mu_s = sum_{k=0}^{s} C(s,k) b^{s-k} (-1)^k *
           (u_hi^{k-beta+1} - u_lo^{k-beta+1}) / (k-beta+1)
"""
import numpy as np
from math import comb
from .variant import b as B, beta as BETA


def _pow_diff(u_hi, u_lo, power):
    """
    Численно устойчиво вычисляет u_hi^power - u_lo^power для 0 <= u_lo < u_hi.
    При малых (u_hi-u_lo) (т.е. u_hi/u_lo ~ 1) прямая разность страдает от
    катастрофического сокращения разрядов -- используем
        u_hi^power - u_lo^power = u_lo^power * expm1(power*log(u_hi/u_lo)),
    что устойчиво, т.к. expm1(x) точен при x->0.
    """
    if u_lo <= 0.0:
        return u_hi ** power
    ratio_log = np.log(u_hi / u_lo)
    return (u_lo ** power) * np.expm1(power * ratio_log)


def moment(s, z_lo, z_hi, b=B, beta=BETA):
    """mu_s = int_{z_lo}^{z_hi} x^s (b-x)^{-beta} dx, аналитически."""
    return moment_about(s, 0.0, z_lo, z_hi, b, beta)


def moment_about(s, center, z_lo, z_hi, b=B, beta=BETA):
    """
    mu'_s = int_{z_lo}^{z_hi} (x-center)^s (b-x)^{-beta} dx, аналитически.

    Та же замена u=b-x, но раскладываем по биному (x-center)^s = (c-u)^s,
    где c = b - center (вместо b при center=0). Используется, чтобы
    работать в локальных, хорошо обусловленных координатах на каждом
    малом отрезке разбиения (при малых h числа x~2 дают плохо
    обусловленную матрицу Вандермонда, а x-center -- нет).
    """
    assert z_lo < z_hi <= b
    c = b - center
    u_lo = b - z_hi
    u_hi = b - z_lo
    total = 0.0
    for k in range(s + 1):
        coef = comb(s, k) * (c ** (s - k)) * ((-1) ** k)
        power = k - beta + 1
        total += coef * _pow_diff(u_hi, u_lo, power) / power
    return total


def moments_vector(max_s, z_lo, z_hi, b=B, beta=BETA):
    """Вектор моментов mu_0, ..., mu_{max_s} на [z_lo, z_hi]."""
    return [moment(s, z_lo, z_hi, b, beta) for s in range(max_s + 1)]
