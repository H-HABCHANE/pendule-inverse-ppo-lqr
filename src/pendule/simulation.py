"""Boucle fermée commune à tous les contrôleurs (PPO, LQR).

Un contrôleur est un appelable état -> force demandée (N). La simulation applique
la même saturation, le même pas de temps, le même intégrateur et la même durée
quel que soit le contrôleur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendule.config import Config
from pendule.dynamique import INTEGRATEURS


@dataclass
class Trajectoire:
    t: np.ndarray  # (N+1,) instants [s]
    etats: np.ndarray  # (N+1, 4) [x, ẋ, θ, θ̇]
    u_demande: np.ndarray  # (N,) force demandée par le contrôleur [N]
    u: np.ndarray  # (N,) force appliquée après saturation [N]
    arret: bool  # True si la simulation s'est arrêtée sur un critère d'arrêt


def etat_initial(theta0_deg: float, theta_point0: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, np.deg2rad(theta0_deg), theta_point0])


def simuler(controleur, s0, cfg: Config, duree_s: float | None = None) -> Trajectoire:
    sim, env = cfg.simulation, cfg.environnement
    pas = INTEGRATEURS[sim.integrateur]
    n = int(round((duree_s or cfg.evaluation.duree_s) / sim.dt))
    s = np.asarray(s0, dtype=np.float64)
    etats, u_demande, u_applique = [s], [], []
    arret = False
    for _ in range(n):
        demande = controleur(s.astype(np.float32))
        u = float(np.clip(demande, -sim.u_max, sim.u_max))
        s = pas(s, u, sim.dt, cfg.physique)
        etats.append(s)
        u_demande.append(demande)
        u_applique.append(u)
        if cfg.evaluation.arret_aux_limites and (
            abs(s[2]) > np.deg2rad(env.theta_arret_deg) or abs(s[0]) > env.x_arret
        ):
            arret = True
            break
    etats = np.array(etats)
    return Trajectoire(np.arange(len(etats)) * sim.dt, etats, np.array(u_demande), np.array(u_applique), arret)
