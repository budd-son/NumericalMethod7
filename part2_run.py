# -*- coding: utf-8 -*-
"""
Часть №2 задания, вариант №8.

2.1  Вычислить J2 = int_a^b f(x)(b-x)^-beta dx с точностью eps=1e-6,
     используя составную 3-точечную КФ Ньютона-Котеса. Погрешность --
     методом Ричардсона (r>1), скорость сходимости m -- правилом Эйткена.
     Вывести m, АСТ и шаг h, при котором достигнута точность.
2.2  То же самое для составной 3-точечной КФ Гаусса.
"""
import warnings
warnings.filterwarnings("ignore")

from quadrature.variant import a, b, f
from quadrature.reference import exact_weighted
from quadrature.newton_cotes3 import composite_newton_cotes3, ACT as ACT_NC3
from quadrature.gauss3 import composite_gauss3, ACT as ACT_G3
from quadrature.richardson import refine_to_tolerance

EPS = 1e-6


def run(name, quad_func, act):
    print("\n" + "=" * 70)
    print(f"{name}  (АСТ малой формулы = {act})")
    print("=" * 70)
    res = refine_to_tolerance(quad_func, a, b, EPS, N0=1, L=2, verbose=True)
    J2 = exact_weighted()
    print("-" * 70)
    if res["converged"]:
        print(f"Требуемая точность eps={EPS:g} ДОСТИГНУТА.")
    else:
        print(f"Точность eps={EPS:g} НЕ достигнута за отведённое число шагов.")
    print(f"  Число разбиений N          = {res['N']}")
    print(f"  Шаг разбиения h            = {res['h']:.6e}")
    print(f"  Скорость сходимости m      = {res['m']:.4f}   (оценка Эйткена)")
    print(f"  АСТ малой формулы          = {act}")
    print(f"  Значение S_N               = {res['S']:.10f}")
    print(f"  Уточнение по Ричардсону    = {res['S_richardson']:.10f}")
    print(f"  Оценка погрешности |R|     = {abs(res['error_est']):.3e}")
    print(f"  Факт. погрешность |S-J2|   = {abs(res['S']-J2):.3e}   (J2 = {J2:.10f}, для контроля)")
    return res


def main():
    res_nc = run("2.1  Составная 3-точечная КФ Ньютона-Котеса", 
                 lambda aa, bb, N: composite_newton_cotes3(f, aa, bb, N), ACT_NC3)
    res_g = run("2.2  Составная 3-точечная КФ Гаусса",
                lambda aa, bb, N: composite_gauss3(f, aa, bb, N), ACT_G3)

    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА (часть 2)")
    print("=" * 70)
    print(f"{'Формула':30s}{'АСТ':>6s}{'m':>10s}{'h':>14s}{'N':>8s}")
    print(f"{'Ньютона-Котеса (3 узла)':30s}{ACT_NC3:>6d}{res_nc['m']:>10.4f}"
          f"{res_nc['h']:>14.3e}{res_nc['N']:>8d}")
    print(f"{'Гаусса (3 узла)':30s}{ACT_G3:>6d}{res_g['m']:>10.4f}"
          f"{res_g['h']:>14.3e}{res_g['N']:>8d}")


if __name__ == "__main__":
    main()
