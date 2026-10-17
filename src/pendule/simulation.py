"""Boucle fermée commune à tous les contrôleurs (PPO, LQR).

Un contrôleur est un appelable état -> force demandée (N). La simulation applique
la même saturation, le même pas de temps, le même intégrateur, la même durée et les
mêmes critères d'arrêt (section « evaluation » de config.yaml) quel que soit le contrôleur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendule.config import Config
from pendule.dynamique import INTEGRATEURS

DUREE_ATTEINTE = "duree_atteinte"
LIMITES_DEPASSEES = "limites_depassees"
ERREUR_NUMERIQUE = "erreur_numerique"


@dataclass
class Trajectoire:
    t: np.ndarray  # (N+1,) instants [s]
    etats: np.ndarray  # (N+1, 4) [x, ẋ, θ, θ̇]
    u_demande: np.ndarray  # (N,) force demandée par le contrôleur [N]
    u: np.ndarray  # (N,) force appliquée après saturation [N]
    fin: str  # DUREE_ATTEINTE, LIMITES_DEPASSEES ou ERREUR_NUMERIQUE


def etat_initial(theta0_deg: float, theta_point0: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, np.deg2rad(theta0_deg), theta_point0])


def simuler(controleur, s0, cfg: Config, duree_s: float | None = None) -> Trajectoire:
    sim, ev = cfg.simulation, cfg.evaluation
    pas = INTEGRATEURS[sim.integrateur]
    n = int(round((duree_s or ev.duree_s) / sim.dt))
    s = np.asarray(s0, dtype=np.float64)
    etats, u_demande, u_applique = [s], [], []
    fin = DUREE_ATTEINTE
    for _ in range(n):
        demande = float(controleur(s.astype(np.float32)))
        u = float(np.clip(demande, -sim.u_max, sim.u_max))
        with np.errstate(all="ignore"):
            s = pas(s, u, sim.dt, cfg.physique)
        etats.append(s)
        u_demande.append(demande)
        u_applique.append(u)
        if not np.all(np.isfinite(s)):
            fin = ERREUR_NUMERIQUE
            break
        if ev.arret_aux_limites and (abs(s[2]) > np.deg2rad(ev.theta_arret_deg) or abs(s[0]) > ev.x_arret):
            fin = LIMITES_DEPASSEES
            break
    etats = np.array(etats)
    return Trajectoire(np.arange(len(etats)) * sim.dt, etats, np.array(u_demande), np.array(u_applique), fin)


def reussites_lot(controleur, S0: np.ndarray, cfg: Config) -> np.ndarray:
    """Version vectorisée de « simuler + critère de réussite » pour N états initiaux (N, 4).

    Utilisée pour la carte du bassin d'attraction. Mêmes règles que ``simuler`` et
    ``metriques.metriques`` : arrêt aux limites, bande ±seuil tenue jusqu'à la fin
    pendant au moins ``maintien_min_s``. Les tests vérifient l'accord avec la version scalaire.
    """
    sim, ev = cfg.simulation, cfg.evaluation
    pas = INTEGRATEURS[sim.integrateur]
    n = int(round(ev.duree_s / sim.dt))
    S = np.array(S0, dtype=np.float64)
    seuil = np.deg2rad(ev.seuil_stabilisation_deg)
    actif = np.ones(len(S), dtype=bool)
    dernier_hors_bande = np.where(np.abs(S[:, 2]) <= seuil, -1, 0)  # indice du dernier pas hors bande
    for k in range(1, n + 1):
        u = np.clip(controleur(S.astype(np.float32)), -sim.u_max, sim.u_max)
        with np.errstate(all="ignore"):
            S_suiv = pas(S.T, u, sim.dt, cfg.physique).T
        S = np.where(actif[:, None], S_suiv, S)
        echec = ~np.all(np.isfinite(S), axis=1)
        if ev.arret_aux_limites:
            echec |= (np.abs(S[:, 2]) > np.deg2rad(ev.theta_arret_deg)) | (np.abs(S[:, 0]) > ev.x_arret)
        actif &= ~echec
        dernier_hors_bande = np.where(actif & (np.abs(S[:, 2]) > seuil), k, dernier_hors_bande)
    debut = dernier_hors_bande + 1
    maintien = (n - debut) * sim.dt
    return actif & (maintien >= ev.maintien_min_s - 1e-9)
