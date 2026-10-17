"""Entraînement d'un agent PPO (Stable-Baselines3) et utilisation comme contrôleur.

Chaque entraînement produit un dossier contenant model.zip, config.yaml (copie
exacte des paramètres utilisés) et monitor.csv (récompense et longueur des épisodes).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pendule.config import Config, charger


def entrainer(cfg: Config, dossier: str | Path, verbose: int = 0) -> Path:
    # Imports lourds gardés locaux : importer le package ne charge pas PyTorch.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from pendule.env import PenduleInverseEnv

    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    cfg.sauvegarder(dossier / "config.yaml")
    p = cfg.ppo
    env = Monitor(PenduleInverseEnv(cfg), filename=str(dossier / "monitor.csv"))
    modele = PPO(
        "MlpPolicy",
        env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        seed=p.seed,
        device="cpu",
        verbose=verbose,
    )
    debut = time.perf_counter()
    modele.learn(total_timesteps=p.total_timesteps)
    duree = time.perf_counter() - debut
    modele.save(dossier / "model.zip")
    env.close()
    infos = {"duree_s": round(duree, 1), "pas_par_seconde": round(p.total_timesteps / duree, 1)}
    (dossier / "infos.json").write_text(json.dumps(infos, indent=2) + "\n")
    return dossier


class ControleurPPO:
    nom = "PPO"

    def __init__(self, modele):
        self.modele = modele

    def __call__(self, s: np.ndarray) -> float:
        action, _ = self.modele.predict(s, deterministic=True)
        return float(action[0])


def charger_agent(dossier: str | Path) -> tuple[ControleurPPO, Config]:
    """Charge un modèle avec la configuration exacte utilisée pour l'entraîner."""
    from stable_baselines3 import PPO

    dossier = Path(dossier)
    cfg = charger(dossier / "config.yaml")
    return ControleurPPO(PPO.load(dossier / "model.zip", device="cpu")), cfg
