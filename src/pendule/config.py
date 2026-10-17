"""Chargement et validation de config.yaml.

Les sections du fichier sont converties en dataclasses figées : une clé inconnue
ou manquante provoque une erreur au lieu d'être ignorée silencieusement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
CONFIG_DEFAUT = RACINE / "config.yaml"


@dataclass(frozen=True)
class Physique:
    M: float
    m: float
    l: float
    g: float


@dataclass(frozen=True)
class Simulation:
    dt: float
    integrateur: str
    u_max: float

    def __post_init__(self):
        if self.integrateur not in ("euler", "rk4"):
            raise ValueError(f"integrateur inconnu : {self.integrateur!r}")


@dataclass(frozen=True)
class LQR:
    Q: tuple[float, ...]
    R: float
    R_balayage: tuple[float, ...]


@dataclass(frozen=True)
class Recompense:
    survie: float
    poids_theta: float
    poids_x: float
    poids_u: float
    poids_theta_point: float
    penalite_arret: float


@dataclass(frozen=True)
class Environnement:
    zone_entrainement_deg: float
    theta_arret_deg: float
    x_arret: float
    duree_episode_s: float
    x0: float
    x_point0: float
    theta_point0: float
    recompense: Recompense


@dataclass(frozen=True)
class PPO:
    total_timesteps: int
    seed: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float


@dataclass(frozen=True)
class Evaluation:
    angles_deg: tuple[float, ...]
    duree_s: float
    arret_aux_limites: bool
    theta_arret_deg: float
    x_arret: float
    seuil_stabilisation_deg: float
    maintien_min_s: float
    x_tol: float
    x_point_tol: float
    theta_point_tol_deg: float


@dataclass(frozen=True)
class Plage:
    min: float
    max: float
    pas: float

    def valeurs(self):
        n = int(round((self.max - self.min) / self.pas)) + 1
        return [self.min + k * self.pas for k in range(n)]


@dataclass(frozen=True)
class Bassin:
    theta0_deg: Plage
    theta_point0: Plage


@dataclass(frozen=True)
class Config:
    physique: Physique
    simulation: Simulation
    lqr: LQR
    environnement: Environnement
    ppo: PPO
    evaluation: Evaluation
    bassin: Bassin

    @property
    def pas_par_episode(self) -> int:
        return int(round(self.environnement.duree_episode_s / self.simulation.dt))

    @property
    def pas_evaluation(self) -> int:
        return int(round(self.evaluation.duree_s / self.simulation.dt))

    def modifier(self, section: str, **valeurs) -> Config:
        """Copie de la configuration avec quelques valeurs d'une section remplacées."""
        return replace(self, **{section: replace(getattr(self, section), **valeurs)})

    def vers_dict(self) -> dict:
        return _listes(asdict(self))

    def sauvegarder(self, chemin: str | Path) -> None:
        Path(chemin).write_text(yaml.safe_dump(self.vers_dict(), sort_keys=False, allow_unicode=True))


def _construire(cls, donnees: dict, nom: str):
    attendues = {f.name: f for f in fields(cls)}
    inconnues = set(donnees) - set(attendues)
    manquantes = set(attendues) - set(donnees)
    if inconnues or manquantes:
        raise ValueError(f"[{nom}] clés inconnues {sorted(inconnues)}, manquantes {sorted(manquantes)}")
    valeurs = {}
    for cle, f in attendues.items():
        v = donnees[cle]
        if isinstance(v, dict):  # sous-section : son type est donné par l'annotation
            v = _construire(globals()[f.type], v, f"{nom}.{cle}")
        elif isinstance(v, list):
            v = tuple(float(x) for x in v)
        valeurs[cle] = v
    return cls(**valeurs)


def _listes(d):
    if isinstance(d, dict):
        return {k: _listes(v) for k, v in d.items()}
    if isinstance(d, tuple):
        return list(d)
    return d


SECTIONS = {
    "physique": Physique,
    "simulation": Simulation,
    "lqr": LQR,
    "environnement": Environnement,
    "ppo": PPO,
    "evaluation": Evaluation,
    "bassin": Bassin,
}

# Sections décrivant l'agent et son entraînement : pour un modèle, elles sont lues dans
# la copie sauvegardée avec lui. Les autres décrivent le protocole d'évaluation.
SECTIONS_ENTRAINEMENT = ("physique", "simulation", "environnement", "ppo")


def depuis_dict(donnees: dict) -> Config:
    if set(donnees) != set(SECTIONS):
        raise ValueError(f"sections attendues {sorted(SECTIONS)}, reçues {sorted(donnees)}")
    return Config(**{nom: _construire(cls, donnees[nom], nom) for nom, cls in SECTIONS.items()})


def charger(chemin: str | Path = CONFIG_DEFAUT) -> Config:
    return depuis_dict(yaml.safe_load(Path(chemin).read_text()))


def charger_modele(chemin: str | Path, protocole: Config) -> Config:
    """Configuration d'un modèle : sections d'entraînement du modèle, protocole d'évaluation courant.

    Ainsi un modèle est toujours simulé avec la physique et l'environnement sur lesquels
    il a été entraîné, et évalué exactement comme les autres contrôleurs.
    """
    sauvegarde = yaml.safe_load(Path(chemin).read_text())
    donnees = protocole.vers_dict()
    for section in SECTIONS_ENTRAINEMENT:
        donnees[section] = sauvegarde[section]
    return depuis_dict(donnees)


def verifier_compatibles(a: Config, b: Config) -> None:
    """Lève une erreur si deux configurations diffèrent ailleurs que dans la section ppo.

    Des modèles entraînés avec des graines ou des budgets différents sont comparables ;
    des modèles entraînés sur une autre physique ou un autre environnement ne le sont pas.
    """
    for section in ("physique", "simulation", "environnement"):
        if getattr(a, section) != getattr(b, section):
            raise ValueError(
                f"configurations incompatibles dans [{section}] :\n  {getattr(a, section)}\n  {getattr(b, section)}"
            )
