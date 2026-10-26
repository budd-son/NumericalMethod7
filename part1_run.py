# -*- coding: utf-8 -*-
"""
Часть №1 задания (п.1.1 - 1.3), вариант №8.

Запускает все составные квадратурные формулы:
  1.1  левые прямоугольники, средние прямоугольники, трапеции, Симпсон
       (p(x) = 1, интеграл от f(x))
  1.2  составная 3-точечная ИКФ Ньютона-Котеса и составная 3-точечная
       КФ типа Гаусса (с весом p(x) = (x-a)^-alpha (b-x)^-beta,
       узлы Гаусса находятся аналитически методом Кардано)
  1.3  графики зависимости |погрешности| от числа разбиений n
       (с указанием АСТ каждой формулы), плюс вывод АСТ в консоль
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadrature.variant import a, b, alpha, beta, f
from quadrature.reference import exact_unweighted, exact_weighted
from quadrature.elementary import (left_rectangles, mid_rectangles,
                                    trapezoid, simpson, ACT as ACT_ELEM)
from quadrature.newton_cotes3 import composite_newton_cotes3, ACT as ACT_NC3
from quadrature.gauss3 import composite_gauss3, ACT as ACT_G3

OUT = "outputs"


def main():
    print("=" * 70)
    print("ВАРИАНТ №8")
    print(f"f(x) = 3.7 cos(1.5x) exp(-4x/3) + 2.4 sin(4.5x) exp(2x/3) + 4")
    print(f"a={a}, b={b}, alpha={alpha}, beta={beta}")
    print("=" * 70)

    J1 = exact_unweighted()
    J2 = exact_weighted()
    print(f"\n'Эталонное' значение J1 = int_a^b f(x)dx           = {J1:.12f}")
    print(f"'Эталонное' значение J2 = int_a^b f(x)/(b-x)^beta dx = {J2:.12f}")

    # ---------------------------------------------------------------
    # 1.1: составные формулы для p(x)=1
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Пункт 1.1. Составные КФ для p(x)=1 (алгебраическая степень точности "
          "малой формулы, АСТ):")
    for name, act in ACT_ELEM.items():
        print(f"  {name:18s}  АСТ = {act}")

    methods_11 = {
        "left_rectangles": (left_rectangles, "Левые прямоугольники"),
        "mid_rectangles": (mid_rectangles, "Средние прямоугольники"),
        "trapezoid": (trapezoid, "Трапеции"),
        "simpson": (simpson, "Симпсон"),
    }

    ns_11 = np.unique(np.round(np.geomspace(2, 400, 30)).astype(int))
    errors_11 = {key: [] for key in methods_11}
    for n in ns_11:
        for key, (func, _) in methods_11.items():
            val = func(f, a, b, n)
            errors_11[key].append(abs(val - J1))

    plt.figure(figsize=(8, 6))
    for key, (func, label) in methods_11.items():
        act = ACT_ELEM[key]
        plt.loglog(ns_11, errors_11[key], marker="o", markersize=3,
                    label=f"{label} (АСТ={act})")
    plt.xlabel("число разбиений n")
    plt.ylabel("абсолютная погрешность |J - S_n|")
    plt.title("П.1.1: погрешность составных КФ (p(x)=1), вариант №8")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/part1_1_errors.png", dpi=150)
    plt.close()
    print(f"\nГрафик сохранён: {OUT}/part1_1_errors.png")

    # ---------------------------------------------------------------
    # 1.2: составные 3-точечные КФ Ньютона-Котеса и Гаусса (с весом)
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Пункт 1.2. Составные 3-точечные КФ c весом p(x)=(x-a)^-alpha(b-x)^-beta:")
    print(f"  {'3-точечная Ньютона-Котеса':30s} АСТ = {ACT_NC3}")
    print(f"  {'3-точечная Гаусса':30s} АСТ = {ACT_G3}")

    ns_12 = np.unique(np.round(np.geomspace(1, 150, 24)).astype(int))
    err_nc, err_g = [], []
    ns_nc_ok, ns_g_ok = [], []
    for n in ns_12:
        try:
            val = composite_newton_cotes3(f, a, b, n)
            err_nc.append(abs(val - J2))
            ns_nc_ok.append(n)
        except Exception:
            pass
        try:
            val = composite_gauss3(f, a, b, n)
            err_g.append(abs(val - J2))
            ns_g_ok.append(n)
        except Exception:
            pass

    plt.figure(figsize=(8, 6))
    plt.loglog(ns_nc_ok, err_nc, marker="o", markersize=3,
                label=f"3-точ. Ньютона-Котеса (АСТ={ACT_NC3})")
    plt.loglog(ns_g_ok, err_g, marker="s", markersize=3,
                label=f"3-точ. Гаусса (АСТ={ACT_G3})")
    plt.xlabel("число разбиений n")
    plt.ylabel("абсолютная погрешность |J - S_n|")
    plt.title("П.1.2: погрешность составных 3-точечных КФ (с весом), вариант №8")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/part1_2_errors.png", dpi=150)
    plt.close()
    print(f"График сохранён: {OUT}/part1_2_errors.png")

    print("\n" + "=" * 70)
    print("Итоговая сводка АСТ малых формул:")
    for key, act in ACT_ELEM.items():
        print(f"  {key:18s}: АСТ = {act}")
    print(f"  {'newton_cotes3':18s}: АСТ = {ACT_NC3}")
    print(f"  {'gauss3':18s}: АСТ = {ACT_G3}")


if __name__ == "__main__":
    main()
