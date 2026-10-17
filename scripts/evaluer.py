"""Évalue le LQR de référence et, si fournis, des agents PPO entraînés.

Tous les contrôleurs sont simulés avec la même configuration ; pour PPO, c'est la
configuration sauvegardée avec le modèle (les modèles doivent être compatibles).

Exemples :
    python scripts/evaluer.py                                  # LQR seul
    python scripts/evaluer.py --modeles runs/zone20            # LQR + toutes les graines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger, verifier_compatibles
from pendule.figures import figure_trajectoires
from pendule.lqr import ControleurLQR
from pendule.metriques import metriques_origine
from pendule.simulation import etat_initial, simuler


def evaluer(controleur, cfg, graine="-") -> tuple[list[dict], dict]:
    lignes, trajs = [], {}
    for angle in cfg.evaluation.angles_deg:
        traj = simuler(controleur, etat_initial(angle), cfg)
        trajs[angle] = traj
        lignes.append(
            {"controleur": controleur.nom, "graine": graine, "theta0_deg": angle, **metriques_origine(traj, cfg)}
        )
    return lignes, trajs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--modeles", default=None, help="dossier contenant seed_*/model.zip")
    parser.add_argument("--sortie", default="resultats", help="dossier des CSV")
    parser.add_argument("--figures", default=None, help="dossier où enregistrer les trajectoires (optionnel)")
    args = parser.parse_args()

    cfg = charger(args.config)
    agents = []
    if args.modeles:
        from pendule.ppo import charger_agent

        dossiers = sorted(p.parent for p in Path(args.modeles).glob("seed_*/model.zip"))
        if not dossiers:
            raise SystemExit(f"aucun modèle trouvé dans {args.modeles}/seed_*/model.zip")
        for dossier in dossiers:
            agent, cfg_agent = charger_agent(dossier)
            if agents:  # toutes les graines doivent partager le même environnement
                verifier_compatibles(cfg_agent, agents[0][1])
            agents.append((agent, cfg_agent))
        cfg = agents[0][1]

    lignes, trajs = evaluer(ControleurLQR(cfg), cfg)
    if args.figures:
        figure_trajectoires(
            trajs, cfg, "LQR de référence — modèle non linéaire", Path(args.figures) / "trajectoires_lqr.png"
        )
    for agent, cfg_agent in agents:
        lignes_ppo, trajs = evaluer(agent, cfg_agent, graine=cfg_agent.ppo.seed)
        lignes += lignes_ppo
        if args.figures:
            figure_trajectoires(
                trajs,
                cfg_agent,
                f"PPO (graine {cfg_agent.ppo.seed}, zone ±{cfg_agent.environnement.zone_entrainement_deg:g}°)",
                Path(args.figures) / f"trajectoires_ppo_seed{cfg_agent.ppo.seed}.png",
            )

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(lignes)
    df.to_csv(sortie / "evaluation.csv", index=False, float_format="%.4f")
    cfg.sauvegarder(sortie / "config.yaml")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {sortie / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
