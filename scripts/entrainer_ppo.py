"""Entraîne un ou plusieurs agents PPO.

Exemples :
    python scripts/entrainer_ppo.py --seed 0 1 2
    python scripts/entrainer_ppo.py --seed 0 --zone 10 --sortie runs/zone10
    python scripts/entrainer_ppo.py --seed 0 --timesteps 20000 --sortie runs/essai   # vérification rapide
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pendule.config import CONFIG_DEFAUT, charger
from pendule.ppo import entrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT, help="fichier de paramètres (défaut : config.yaml)")
    parser.add_argument("--seed", type=int, nargs="+", default=None, help="une ou plusieurs graines")
    parser.add_argument("--zone", type=float, default=None, help="zone d'entraînement ±zone (degrés)")
    parser.add_argument("--timesteps", type=int, default=None, help="nombre de pas d'entraînement par graine")
    parser.add_argument("--sortie", default=None, help="dossier de sortie (défaut : runs/zone<zone>)")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    cfg = charger(args.config)
    if args.zone is not None:
        cfg = cfg.modifier("environnement", zone_entrainement_deg=args.zone)
    if args.timesteps is not None:
        cfg = cfg.modifier("ppo", total_timesteps=args.timesteps)
    graines = args.seed if args.seed is not None else [cfg.ppo.seed]
    sortie = Path(args.sortie or f"runs/zone{cfg.environnement.zone_entrainement_deg:g}")

    for graine in graines:
        dossier = sortie / f"seed_{graine}"
        zone = cfg.environnement.zone_entrainement_deg
        print(f"graine {graine} : {cfg.ppo.total_timesteps} pas, zone ±{zone:g}° -> {dossier}")
        entrainer(cfg.modifier("ppo", seed=graine), dossier, verbose=args.verbose)


if __name__ == "__main__":
    main()
