# -*- coding: utf-8 -*-
"""
Часть 2: практическая оценка погрешности составных квадратурных формул
методом Ричардсона (r>1) со скоростью сходимости m, оцениваемой правилом
Эйткена (формула (36)), см. Тема 7, разделы "Метод Ричардсона" и
"Процесс Эйткена".

Алгоритм refine_to_tolerance:
  1. Строим последовательность сеток с шагами h, h/L, h/L^2, ... (N, NL, NL^2,...)
  2. По каждой новой тройке (S1,S2,S3) вычисляем оценку скорости сходимости m
     по формуле Эйткена (36):
         m ~= -ln(|S3-S2|/|S2-S1|) / ln(L)
  3. По формуле Ричардсона (32) оцениваем погрешность наиболее точного
     (последнего) значения S3 на шаге h3=h/L^2:
         R3 ~= (S3-S2) / (L^m - 1)
  4. Если |R3| < eps -- останов, иначе сетку измельчаем дальше (N *= L)
     и повторяем п.2-3 для новой тройки последних значений.
"""
import numpy as np


def refine_to_tolerance(quad_func, a, b, eps, N0=1, L=2, max_steps=60, verbose=False):
    """
    quad_func(a, b, N) -- составная квадратурная формула на N частичных
    отрезках.
    Возвращает словарь с полями: converged, N, h, m (оценка Эйткена),
    S (значение на самой мелкой из использованных сеток),
    S_richardson (уточнение по Ричардсону), error_est, history.
    """
    Ns = [N0, N0 * L, N0 * L * L]
    Ss = [quad_func(a, b, N) for N in Ns]
    history = list(zip(Ns, Ss))

    for _ in range(max_steps):
        S1, S2, S3 = Ss[-3], Ss[-2], Ss[-1]
        d_num = S3 - S2
        d_den = S2 - S1

        m = np.nan
        R3 = None
        if d_den != 0:
            ratio = abs(d_num / d_den)
            if ratio > 0:
                m = -np.log(ratio) / np.log(L)
        if not np.isnan(m) and m > 0:
            R3 = d_num / (L ** m - 1.0)

        h3 = (b - a) / Ns[-1]
        if verbose:
            print(f"  N={Ns[-1]:>8d}  h={h3:.3e}  S={S3:.12f}  "
                  f"m~={m:.4f}  |R|~={abs(R3) if R3 is not None else float('nan'):.3e}")

        if R3 is not None and abs(R3) < eps:
            return {
                "converged": True, "N": Ns[-1], "h": h3, "m": m,
                "S": S3, "S_richardson": S3 + R3, "error_est": R3,
                "history": history,
            }

        newN = Ns[-1] * L
        Ns.append(newN)
        Ss.append(quad_func(a, b, newN))
        history.append((newN, Ss[-1]))

    return {"converged": False, "N": Ns[-1], "h": (b - a) / Ns[-1],
            "m": m, "S": Ss[-1], "S_richardson": None,
            "error_est": R3, "history": history}
