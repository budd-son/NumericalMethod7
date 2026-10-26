# -*- coding: utf-8 -*-
"""
Часть 1.1: составные квадратурные формулы для p(x)=1 (интеграл от f(x)):
левых прямоугольников, средних прямоугольников, трапеций, Симпсона.

Каждая функция возвращает приближённое значение интеграла J(f) на [a,b]
при разбиении на n равных частей. Алгебраическая степень точности (АСТ)
малой (элементарной) формулы указана в ACT-словаре ниже и используется
при построении графиков и печати в консоль.
"""
import numpy as np

ACT = {
    "left_rectangles": 0,
    "mid_rectangles": 1,
    "trapezoid": 1,
    "simpson": 3,
}


def left_rectangles(f, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n)  # x_0 .. x_{n-1}
    return h * np.sum(f(x))


def mid_rectangles(f, a, b, n):
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(x))


def trapezoid(f, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return h * (y[0] / 2.0 + y[-1] / 2.0 + np.sum(y[1:-1]))


def simpson(f, a, b, n):
    """Составная формула Симпсона. n -- число элементарных отрезков
    длины h=(b-a)/n (n может быть любым натуральным; каждая пара соседних
    отрезков образует малую формулу Симпсона по трём узлам -- поэтому n
    внутри приводим к чётному числу шагов)."""
    if n % 2 == 1:
        n += 1  # для формулы Симпсона нужно чётное число шагов
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return h / 3.0 * s


METHODS = {
    "left_rectangles": left_rectangles,
    "mid_rectangles": mid_rectangles,
    "trapezoid": trapezoid,
    "simpson": simpson,
}
