"""Commande LQR calculée sur le modèle linéarisé, appliquée au modèle non linéaire.

K = R⁻¹ Bᵀ P, où P est la solution de l'équation de Riccati continue
AᵀP + PA − P B R⁻¹ Bᵀ P + Q = 0 (diapo 10). La loi u = −K s est échantillonnée
à la période dt et saturée à ±u_max par la simulation, comme pour l'agent PPO.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from pendule.config import Config
from pendule.dynamique import linearisation


def gain_lqr(cfg: Config, R: float | None = None) -> np.ndarray:
    """Gain K (forme (4,)) tel que u = −K s. ``R`` permet de remplacer la valeur de config.yaml."""
    A, B = linearisation(cfg.physique)
    Q = np.diag(cfg.lqr.Q)
    R_mat = np.array([[cfg.lqr.R if R is None else R]])
    P = solve_continuous_are(A, B, Q, R_mat)
    return np.linalg.solve(R_mat, B.T @ P).ravel()


class ControleurLQR:
    nom = "LQR"

    def __init__(self, cfg: Config, R: float | None = None):
        self.K = gain_lqr(cfg, R)

    def __call__(self, s: np.ndarray) -> float:
        """Force demandée en N, avant saturation."""
        return float(-self.K @ np.asarray(s, dtype=np.float64))
