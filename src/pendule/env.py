"""Environnement Gymnasium du pendule inversé (reprise de « PenduleInverseEnv » du notebook).

Action      : u ∈ [−u_max, u_max], force en N (saturée dans step)
Observation : [x, ẋ, θ, θ̇] en float32
Fin d'épisode :
  terminated -> |θ| > theta_arret ou |x| > x_arret
  truncated  -> durée de l'épisode atteinte
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pendule.config import Config
from pendule.dynamique import INTEGRATEURS


class PenduleInverseEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._pas = INTEGRATEURS[cfg.simulation.integrateur]
        u_max = cfg.simulation.u_max
        self.action_space = spaces.Box(-u_max, u_max, shape=(1,), dtype=np.float32)
        # Bornes reprises du notebook.
        borne = np.array([5.0, 10.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-borne, borne, dtype=np.float32)
        self.etat = np.zeros(4)
        self.n_pas = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """État initial aléatoire (distribution d'entraînement) ou ``options["etat_initial"]``."""
        super().reset(seed=seed)
        options = options or {}
        if "etat_initial" in options:
            self.etat = np.array(options["etat_initial"], dtype=np.float64)
        else:
            e = self.cfg.environnement
            tirage = self.np_random.uniform
            self.etat = np.array(
                [
                    tirage(-e.x0, e.x0),
                    tirage(-e.x_point0, e.x_point0),
                    tirage(-1.0, 1.0) * np.deg2rad(e.zone_entrainement_deg),
                    tirage(-e.theta_point0, e.theta_point0),
                ]
            )
        self.n_pas = 0
        return self._observation(), {}

    def step(self, action):
        sim, env = self.cfg.simulation, self.cfg.environnement
        u = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -sim.u_max, sim.u_max))
        self.etat = self._pas(self.etat, u, sim.dt, self.cfg.physique)
        self.n_pas += 1

        x, _, theta, _ = self.etat
        hors_limites = abs(theta) > np.deg2rad(env.theta_arret_deg) or abs(x) > env.x_arret
        terminated = bool(hors_limites)
        truncated = bool(self.n_pas >= self.cfg.pas_par_episode)
        recompense = env.recompense.penalite_arret if terminated else self._recompense(u)
        return self._observation(), recompense, terminated, truncated, {"u": u}

    def _recompense(self, u: float) -> float:
        """+survie − 10 θ² − 0,1 x² − 0,01 u² − 0,5 θ̇² (poids dans config.yaml)."""
        r = self.cfg.environnement.recompense
        x, _, theta, theta_point = self.etat
        return float(
            r.survie
            - r.poids_theta * theta**2
            - r.poids_x * x**2
            - r.poids_u * u**2
            - r.poids_theta_point * theta_point**2
        )

    def _observation(self) -> np.ndarray:
        return self.etat.astype(np.float32)
