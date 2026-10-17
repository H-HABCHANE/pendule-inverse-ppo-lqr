"""Métriques calculées sur une trajectoire."""

from __future__ import annotations

import numpy as np

from pendule.config import Config
from pendule.simulation import Trajectoire


def temps_premier_passage(traj: Trajectoire, seuil_deg: float) -> float:
    """Premier instant où |θ| < seuil (méthode d'origine du notebook, cellule « CODE 2 »).

    Attention : un pendule qui traverse simplement la verticale est compté comme
    « stabilisé ». Conservé pour comparer aux chiffres des diapos.
    """
    idx = np.flatnonzero(np.abs(np.rad2deg(traj.etats[:, 2])) < seuil_deg)
    return float(traj.t[idx[0]]) if len(idx) else float("nan")


def metriques_origine(traj: Trajectoire, cfg: Config) -> dict:
    """Les grandeurs affichées dans les tableaux du notebook et des diapos 12 et 24."""
    return {
        "t_stab_premier_passage_s": temps_premier_passage(traj, cfg.evaluation.seuil_stabilisation_deg),
        "theta_max_deg": float(np.max(np.abs(np.rad2deg(traj.etats[:, 2])))),
        "u_demande_initial_N": float(abs(traj.u_demande[0])),
        "u_max_applique_N": float(np.max(np.abs(traj.u))),
        "duree_simulee_s": float(traj.t[-1]),
    }
