"""Mesure du débit d'entraînement PPO (pas/s) en fonction du nombre d'environnements parallèles.

Chaque mise à jour PPO utilise toujours 2048 transitions (n_steps = 2048 / n_envs par
environnement), pour que seul le parallélisme change. À lancer sur une machine au repos.

Exemple :
    python scripts/debit_vecenv.py --n-envs 1 2 4 8 --timesteps 32768
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger


def mesurer(cfg, n_envs: int, vec: str, timesteps: int) -> float:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from pendule.env import PenduleInverseEnv

    classe = SubprocVecEnv if vec == "subproc" else DummyVecEnv
    env = make_vec_env(PenduleInverseEnv, n_envs=n_envs, seed=0, vec_env_cls=classe, env_kwargs={"cfg": cfg})
    p = cfg.ppo
    modele = PPO("MlpPolicy", env, n_steps=2048 // n_envs, batch_size=p.batch_size, n_epochs=p.n_epochs,
                 learning_rate=p.learning_rate, seed=0, device="cpu")  # fmt: skip
    debut = time.perf_counter()
    modele.learn(total_timesteps=timesteps)
    duree = time.perf_counter() - debut
    env.close()
    return timesteps / duree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--sortie", default="resultats")
    args = parser.parse_args()

    cfg = charger(args.config)
    lignes = []
    for vec in ("dummy", "subproc"):
        for n in args.n_envs:
            debit = mesurer(cfg, n, vec, args.timesteps)
            lignes.append({"vec_env": vec, "n_envs": n, "pas_par_seconde": round(debit, 1)})
            print(f"{vec:8s} n_envs={n:2d} : {debit:7.1f} pas/s")
    df = pd.DataFrame(lignes)
    df["coeurs"] = os.cpu_count()
    Path(args.sortie).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.sortie) / "debit_vecenv.csv", index=False)


if __name__ == "__main__":
    main()
