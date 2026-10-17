"""Carte du bassin d'attraction : réussite sur une grille d'états initiaux (θ0, θ̇0), x0 = ẋ0 = 0.

Même critère de réussite que evaluer.py. Pour PPO, la valeur d'une case est la fraction
des graines qui réussissent.

Exemple :
    python scripts/bassin.py --modeles runs/zone20 runs/zone10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger
from pendule.dynamique import angle_limite_rattrapage
from pendule.figures import figure_bassin
from pendule.lqr import ControleurLQR
from pendule.simulation import etat_initial, reussites_lot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--modeles", nargs="*", default=[])
    parser.add_argument("--sortie", default="resultats")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    cfg = charger(args.config)
    thetas = np.array(cfg.bassin.theta0_deg.valeurs())
    omegas = np.array(cfg.bassin.theta_point0.valeurs())
    grille = np.array([etat_initial(t, w) for t in thetas for w in omegas])
    forme = (len(thetas), len(omegas))

    cartes = {f"LQR saturé (R={cfg.lqr.R:g})": reussites_lot(ControleurLQR(cfg), grille, cfg).reshape(forme)}
    zones = {}
    if args.modeles:
        from pendule.ppo import charger_agent

        for d in args.modeles:
            runs = sorted(p.parent for p in Path(d).glob("seed_*/model.zip"))
            succes = []
            for run in runs:
                agent, cfg_agent = charger_agent(run, cfg)
                succes.append(reussites_lot(agent, grille, cfg_agent).reshape(forme))
                print(f"{run} : {succes[-1].mean():.1%} de la grille stabilisée")
            nom = f"PPO ±{cfg_agent.environnement.zone_entrainement_deg:g}° ({len(runs)} graines)"
            cartes[nom] = np.mean(succes, axis=0)
            zones[nom] = cfg_agent.environnement.zone_entrainement_deg

    T, W = np.meshgrid(thetas, omegas, indexing="ij")
    df = pd.DataFrame({"theta0_deg": T.ravel(), "theta_point0_rad_s": W.ravel()})
    for nom, carte in cartes.items():
        df[nom] = carte.ravel()
    Path(args.sortie).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.sortie) / "bassin.csv", index=False, float_format="%.4f")

    limite = np.rad2deg(angle_limite_rattrapage(cfg.physique, cfg.simulation.u_max))
    print(figure_bassin(cartes, thetas, omegas, limite, zones, Path(args.figures) / "bassin_attraction.png"))
    for nom, carte in cartes.items():
        print(f"{nom} : {carte.mean():.1%} de la grille (moyenne sur les graines)")


if __name__ == "__main__":
    main()
