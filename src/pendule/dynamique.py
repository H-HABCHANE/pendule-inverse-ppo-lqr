"""Dynamique du pendule inversé sur chariot : équations, intégrateurs, linéarisation.

Modèle : chariot de masse M sur rail sans frottement, masse ponctuelle m au bout
d'une tige sans masse de longueur l, force horizontale u appliquée au chariot.

État  s = [x, ẋ, θ, θ̇]  (m, m/s, rad, rad/s)
θ = 0 : pendule vertical vers le haut ; θ > 0 : pendule penché vers +x.
Une force positive tend donc à faire diminuer θ.

Équations du mouvement (Lagrange, diapo 5) :
    (M + m) ẍ + m l θ̈ cos θ − m l θ̇² sin θ = u
    l θ̈ + ẍ cos θ = g sin θ
"""

from __future__ import annotations

import numpy as np

from pendule.config import Physique


def derivees(s: np.ndarray, u: float, p: Physique) -> np.ndarray:
    """ṡ = f(s, u), obtenu en résolvant les deux équations en (ẍ, θ̈)."""
    _, x_point, theta, theta_point = s
    sin, cos = np.sin(theta), np.cos(theta)
    D = p.l * (p.M + p.m * sin**2)
    terme = u + p.m * p.l * theta_point**2 * sin
    x_pp = (p.l * terme - p.m * p.l * cos * p.g * sin) / D
    theta_pp = ((p.M + p.m) * p.g * sin - cos * terme) / D
    return np.array([x_point, x_pp, theta_point, theta_pp])


def pas_euler(s: np.ndarray, u: float, dt: float, p: Physique) -> np.ndarray:
    """Euler explicite, identique au notebook : s(k+1) = s(k) + dt · f(s(k), u)."""
    return s + dt * derivees(s, u, p)


def pas_rk4(s: np.ndarray, u: float, dt: float, p: Physique) -> np.ndarray:
    """Runge-Kutta d'ordre 4, force constante sur le pas (bloqueur d'ordre zéro)."""
    k1 = derivees(s, u, p)
    k2 = derivees(s + 0.5 * dt * k1, u, p)
    k3 = derivees(s + 0.5 * dt * k2, u, p)
    k4 = derivees(s + dt * k3, u, p)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


INTEGRATEURS = {"euler": pas_euler, "rk4": pas_rk4}


def linearisation(p: Physique) -> tuple[np.ndarray, np.ndarray]:
    """Matrices (A, B) du modèle linéarisé autour de s = 0, u = 0 (diapo 5)."""
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -p.m * p.g / p.M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (p.M + p.m) * p.g / (p.M * p.l), 0.0],
        ]
    )
    B = np.array([[0.0], [1.0 / p.M], [0.0], [-1.0 / (p.M * p.l)]])
    return A, B


def angle_limite_rattrapage(p: Physique, u_max: float) -> float:
    """Angle θ* (rad) au-delà duquel un pendule lâché sans vitesse ne peut plus être rattrapé.

    θ̈ = ((M + m) g sin θ − u cos θ) / (l (M + m sin² θ)). Pour |u| <= u_max, θ̈ > 0 dès que
    tan θ > u_max / ((M + m) g) : l'ensemble {θ > θ*, θ̇ >= 0} est alors invariant, quel que
    soit le contrôleur. Avec les valeurs du projet (4 N), θ* ≈ 20,34°.
    """
    return float(np.arctan(u_max / ((p.M + p.m) * p.g)))
