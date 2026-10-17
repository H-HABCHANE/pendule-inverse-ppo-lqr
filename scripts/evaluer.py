"""Évalue tous les contrôleurs avec le même protocole (section « evaluation » de config.yaml).

Contrôleurs évalués :
  - LQR de référence (Q, R de config.yaml), saturé à u_max   -> comparaison principale
  - le même LQR sans saturation                              -> borne idéale (force illimitée)
  - LQR saturé pour chaque valeur de R_balayage
  - agents PPO : chaque dossier passé à --modeles contient seed_*/model.zip

Exemples :
    python scripts/evaluer.py                                       # LQR seuls
    python scripts/evaluer.py --modeles runs/zone20 runs/zone10     # LQR + PPO
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger, verifier_compatibles
from pendule.dynamique import angle_limite_rattrapage
from pendule.lqr import ControleurLQR
from pendule.metriques import metriques
from pendule.simulation import etat_initial, simuler


def evaluer(controleur, cfg, famille: str, graine="-", zone=None) -> list[dict]:
    limite = np.rad2deg(angle_limite_rattrapage(cfg.physique, cfg.simulation.u_max))
    lignes = []
    for angle in cfg.evaluation.angles_deg:
        traj = simuler(controleur, etat_initial(angle), cfg)
        lignes.append(
            {
                "controleur": controleur.nom,
                "famille": famille,
                "graine": graine,
                "theta0_deg": angle,
                "dans_zone_entrainement": None if zone is None else abs(angle) <= zone,
                "rattrapable_physiquement": abs(angle) <= limite,
                **metriques(traj, cfg),
            }
        )
    return lignes


def evaluer_lqr(cfg) -> list[dict]:
    R = cfg.lqr.R
    lignes = evaluer(ControleurLQR(cfg, nom=f"LQR saturé (R={R:g})"), cfg, "LQR référence")
    sans_saturation = cfg.modifier("simulation", u_max=float("inf"))
    lignes += evaluer(ControleurLQR(cfg, nom=f"LQR non saturé (R={R:g})"), sans_saturation, "LQR non saturé")
    for r in cfg.lqr.R_balayage:
        if r != R:
            lignes += evaluer(ControleurLQR(cfg, R=r, nom=f"LQR saturé (R={r:g})"), cfg, "LQR balayage R")
    return lignes


def evaluer_ppo(dossier_runs: Path, protocole) -> list[dict]:
    from pendule.ppo import charger_agent

    dossiers = sorted(p.parent for p in dossier_runs.glob("seed_*/model.zip"))
    if not dossiers:
        raise SystemExit(f"aucun modèle trouvé dans {dossier_runs}/seed_*/model.zip")
    lignes, reference = [], None
    for dossier in dossiers:
        agent, cfg = charger_agent(dossier, protocole)
        if reference is None:
            reference = cfg
        verifier_compatibles(cfg, reference)  # toutes les graines : même environnement
        zone = cfg.environnement.zone_entrainement_deg
        agent.nom = f"PPO ±{zone:g}°"
        lignes += evaluer(agent, cfg, "PPO", graine=cfg.ppo.seed, zone=zone)
    return lignes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--modeles", nargs="*", default=[], help="dossiers contenant seed_*/model.zip")
    parser.add_argument("--sortie", default="resultats", help="dossier des CSV")
    args = parser.parse_args()

    cfg = charger(args.config)
    lignes = evaluer_lqr(cfg)
    for dossier in args.modeles:
        lignes += evaluer_ppo(Path(dossier), cfg)

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(lignes)
    df.to_csv(sortie / "evaluation.csv", index=False, float_format="%.4f")
    cfg.sauvegarder(sortie / "config.yaml")
    colonnes = ["controleur", "graine", "theta0_deg", "issue", "t_etablissement_s", "t_stab_premier_passage_s"]
    print(df[colonnes].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {sortie / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
