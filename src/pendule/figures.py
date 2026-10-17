"""Figures. Les courbes ne sont jamais écrêtées : les axes s'adaptent aux données."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pendule.config import Config  # noqa: E402
from pendule.simulation import Trajectoire  # noqa: E402

GRIS = "#52514e"
# Palette catégorielle (ordre fixe) : une couleur par angle initial.
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def figure_trajectoires(trajs: dict[float, Trajectoire], cfg: Config, titre: str, chemin: str | Path) -> Path:
    """θ, θ̇, u, x, ẋ en fonction du temps, une courbe par angle initial."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    axes = axes.ravel()
    grandeurs = [
        ("θ (°)", lambda tr: np.rad2deg(tr.etats[:, 2])),
        ("θ̇ (°/s)", lambda tr: np.rad2deg(tr.etats[:, 3])),
        ("u appliquée (N)", None),
        ("x (m)", lambda tr: tr.etats[:, 0]),
        ("ẋ (m/s)", lambda tr: tr.etats[:, 1]),
    ]
    for k, (angle, tr) in enumerate(trajs.items()):
        couleur = COULEURS[k % len(COULEURS)]
        for ax, (_, fn) in zip(axes[:5], grandeurs, strict=True):
            if fn is None:
                ax.step(tr.t[:-1], tr.u, where="post", color=couleur, lw=1.3, label=f"θ0 = {angle:g}°")
            else:
                ax.plot(tr.t, fn(tr), color=couleur, lw=1.3, label=f"θ0 = {angle:g}°")
    for ax, (label, _) in zip(axes[:5], grandeurs, strict=True):
        ax.set(xlabel="temps (s)", ylabel=label)
        ax.grid(alpha=0.3)
    u_max = cfg.simulation.u_max
    for signe in (1, -1):
        axes[2].axhline(signe * u_max, color=GRIS, ls="--", lw=0.8)
    axes[0].legend(fontsize=8)
    axes[5].axis("off")
    s = cfg.simulation
    fig.suptitle(f"{titre}\n|u| ≤ {u_max:g} N, dt = {s.dt:g} s, intégrateur {s.integrateur}", fontsize=11)
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return chemin
