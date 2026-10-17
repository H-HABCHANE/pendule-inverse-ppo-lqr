"""Compare Euler explicite (défaut) et RK4 avec le même pas dt.

1. Dérive d'énergie sans commande (le système est conservatif : l'énergie doit rester constante).
2. Mêmes contrôleurs et mêmes états initiaux simulés avec les deux intégrateurs.
   Les agents PPO ont été entraînés avec Euler : RK4 est pour eux un léger changement de modèle.

Exemple :
    python scripts/integrateurs.py --modeles runs/zone20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger
from pendule.dynamique import INTEGRATEURS
from pendule.figures import figure_integrateurs
from pendule.lqr import ControleurLQR
from pendule.metriques import metriques
from pendule.simulation import etat_initial, simuler


def energie(s: np.ndarray, p) -> float:
    _, xp, th, thp = s
    cinetique = 0.5 * (p.M + p.m) * xp**2 + p.m * p.l * xp * thp * np.cos(th) + 0.5 * p.m * p.l**2 * thp**2
    return cinetique + p.m * p.g * p.l * np.cos(th)


def derive_energie(cfg, integrateur: str, duree_s: float = 10.0):
    s = etat_initial(60.0)
    e0, ecarts = energie(s, cfg.physique), [0.0]
    for _ in range(int(round(duree_s / cfg.simulation.dt))):
        s = INTEGRATEURS[integrateur](s, 0.0, cfg.simulation.dt, cfg.physique)
        ecarts.append(energie(s, cfg.physique) / e0 - 1)
    return np.arange(len(ecarts)) * cfg.simulation.dt, np.array(ecarts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--modeles", default=None, help="dossier seed_*/ d'agents PPO (optionnel)")
    parser.add_argument("--sortie", default="resultats")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    cfg = charger(args.config)
    variantes = {i: cfg.modifier("simulation", integrateur=i) for i in ("euler", "rk4")}
    controleurs = [(f"LQR saturé (R={cfg.lqr.R:g})", ControleurLQR(cfg), None)]
    if args.modeles:
        from pendule.ppo import charger_agent

        for run in sorted(p.parent for p in Path(args.modeles).glob("seed_*/model.zip")):
            agent, cfg_agent = charger_agent(run, cfg)
            nom = f"PPO ±{cfg_agent.environnement.zone_entrainement_deg:g}°"
            controleurs.append((nom, agent, cfg_agent))

    lignes = []
    for nom, ctrl, cfg_agent in controleurs:
        base = cfg_agent or cfg
        graine = "-" if cfg_agent is None else cfg_agent.ppo.seed
        for angle in cfg.evaluation.angles_deg:
            for integ in variantes:
                c = base.modifier("simulation", integrateur=integ)
                m = metriques(simuler(ctrl, etat_initial(angle), c), c)
                lignes.append({"controleur": nom, "graine": graine, "theta0_deg": angle, "integrateur": integ, **m})
    df = pd.DataFrame(lignes)
    Path(args.sortie).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.sortie) / "euler_vs_rk4.csv", index=False, float_format="%.4f")

    derive = {i: derive_energie(cfg, i) for i in variantes}
    exemples = {}
    for nom, ctrl, cfg_agent in controleurs[:2]:  # LQR et la première graine PPO
        base = cfg_agent or cfg
        for angle in (15.0, 18.0):
            exemples[(nom, angle)] = {
                i: simuler(ctrl, etat_initial(angle), base.modifier("simulation", integrateur=i)) for i in variantes
            }
    print(figure_integrateurs(derive, exemples, Path(args.figures) / "euler_vs_rk4.png"))

    pivot = df.pivot_table(index=["controleur", "graine", "theta0_deg"], columns="integrateur",
                           values="t_etablissement_s", aggfunc="first")  # fmt: skip
    print(pivot.to_string(float_format=lambda v: f"{v:.2f}"))
    for i, (_, e) in derive.items():
        print(f"dérive d'énergie après 10 s sans commande ({i}) : {100 * e[-1]:+.4f} %")


if __name__ == "__main__":
    main()
