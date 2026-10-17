"""Métriques calculées sur une trajectoire."""

from __future__ import annotations

import numpy as np

from pendule.config import Config
from pendule.simulation import DUREE_ATTEINTE, Trajectoire

# Issue d'une simulation
STABILISE = "stabilise"  # temps d'établissement défini (critère de réussite)
NON_STABILISE = "non_stabilise"  # resté dans les limites, mais pas stabilisé pendant la simulation
# LIMITES_DEPASSEES et ERREUR_NUMERIQUE : voir simulation.py


def temps_premier_passage(traj: Trajectoire, seuil_deg: float) -> float:
    """Premier instant où |θ| < seuil (méthode d'origine du notebook, cellule « CODE 2 »).

    Attention : un pendule qui traverse simplement la verticale est compté comme
    « stabilisé ». Conservé uniquement pour comparer aux chiffres des diapos.
    """
    idx = np.flatnonzero(np.abs(np.rad2deg(traj.etats[:, 2])) < seuil_deg)
    return float(traj.t[idx[0]]) if len(idx) else float("nan")


def temps_etablissement(t: np.ndarray, dans_bande: np.ndarray, maintien_min_s: float) -> float:
    """Premier instant à partir duquel ``dans_bande`` reste vrai jusqu'à la fin.

    NaN si le signal est hors bande au dernier instant, ou si la durée de maintien
    finale est inférieure à ``maintien_min_s``.
    """
    if len(dans_bande) == 0 or not dans_bande[-1]:
        return float("nan")
    hors = np.flatnonzero(~dans_bande)
    debut = 0 if len(hors) == 0 else hors[-1] + 1
    if t[-1] - t[debut] < maintien_min_s - 1e-9:
        return float("nan")
    return float(t[debut])


def nombre_traversees(theta: np.ndarray, seuil: float) -> int:
    """Nombre de passages d'un côté à l'autre de la verticale.

    Seuls les échantillons hors de la bande ±seuil comptent, pour ne pas compter
    le bruit autour de zéro.
    """
    signes = np.sign(theta[np.abs(theta) > seuil])
    return int(np.count_nonzero(np.diff(signes)))


def metriques_origine(traj: Trajectoire, cfg: Config) -> dict:
    """Les grandeurs affichées dans les tableaux du notebook et des diapos 12 et 24."""
    return {
        "t_stab_premier_passage_s": temps_premier_passage(traj, cfg.evaluation.seuil_stabilisation_deg),
        "theta_max_deg": float(np.max(np.abs(np.rad2deg(traj.etats[:, 2])))),
        "u_demande_initial_N": float(abs(traj.u_demande[0])),
        "u_max_applique_N": float(np.max(np.abs(traj.u))),
        "duree_simulee_s": float(traj.t[-1]),
    }


def metriques(traj: Trajectoire, cfg: Config) -> dict:
    ev = cfg.evaluation
    x, x_point, theta, theta_point = traj.etats.T
    t, dt = traj.t, cfg.simulation.dt
    seuil = np.deg2rad(ev.seuil_stabilisation_deg)

    angle_ok = np.abs(theta) <= seuil
    complet_ok = (
        angle_ok
        & (np.abs(theta_point) <= np.deg2rad(ev.theta_point_tol_deg))
        & (np.abs(x) <= ev.x_tol)
        & (np.abs(x_point) <= ev.x_point_tol)
    )
    fin_normale = traj.fin == DUREE_ATTEINTE
    t_etab = temps_etablissement(t, angle_ok, ev.maintien_min_s) if fin_normale else float("nan")
    t_complet = temps_etablissement(t, complet_ok, ev.maintien_min_s) if fin_normale else float("nan")

    if not fin_normale:
        issue = traj.fin
    elif np.isnan(t_etab):
        issue = NON_STABILISE
    else:
        issue = STABILISE

    fini = np.all(np.isfinite(traj.etats), axis=1)
    if not fini.any():  # état initial déjà invalide : rien à mesurer
        nan = float("nan")
        return {"issue": issue, "reussite": False, "t_etablissement_s": nan, "t_stabilisation_complete_s": nan,
                "t_stab_premier_passage_s": nan, "depassement_deg": nan, "nb_traversees": 0, "theta_max_deg": nan,
                "u_max_N": nan, "u_demande_max_N": nan, "energie_u2_N2s": nan, "x_final_m": nan, "x_max_m": nan,
                "duree_simulee_s": float(t[-1])}  # fmt: skip
    th = theta[fini]
    signe0 = np.sign(theta[0]) if theta[0] != 0 else 1.0
    return {
        "issue": issue,
        "reussite": issue == STABILISE,
        "t_etablissement_s": t_etab,
        "t_stabilisation_complete_s": t_complet,
        "t_stab_premier_passage_s": temps_premier_passage(traj, ev.seuil_stabilisation_deg),
        "depassement_deg": float(max(0.0, np.rad2deg(np.max(-signe0 * th)))),
        "nb_traversees": nombre_traversees(th, seuil),
        "theta_max_deg": float(np.rad2deg(np.max(np.abs(th)))),
        "u_max_N": float(np.max(np.abs(traj.u))),
        "u_demande_max_N": float(np.max(np.abs(traj.u_demande))),
        "energie_u2_N2s": float(np.sum(traj.u**2) * dt),
        "x_final_m": float(abs(x[fini][-1])),
        "x_max_m": float(np.max(np.abs(x[fini]))),
        "duree_simulee_s": float(t[-1]),
    }
