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


# Une couleur fixe par famille de contrôleurs (jamais réattribuée selon le rang).
COULEURS_FAMILLES = {
    "LQR saturé": "#2a78d6",
    "PPO ±20°": "#eb6834",
    "PPO ±10°": "#1baf7a",
    "LQR non saturé": "#6250d6",
}


def _enregistrer(fig, chemin) -> Path:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return chemin


def figure_comparaison(series: dict[str, dict[float, list[Trajectoire]]], cfg: Config, chemin) -> Path:
    """θ(t) et u(t) par angle initial ; une couleur par contrôleur, une courbe par graine.

    Une croix marque l'arrêt de la simulation quand une limite est dépassée.
    """
    angles = list(next(iter(series.values())).keys())
    fig, axes = plt.subplots(2, len(angles), figsize=(3.4 * len(angles), 6.2), sharex=True, layout="constrained")
    for nom, par_angle in series.items():
        couleur = COULEURS_FAMILLES[nom]
        for j, angle in enumerate(angles):
            trajs = par_angle[angle]
            alpha = 1.0 if len(trajs) == 1 else 0.6
            for tr in trajs:
                theta = np.rad2deg(tr.etats[:, 2])
                axes[0, j].plot(tr.t, theta, color=couleur, lw=1.3, alpha=alpha)
                axes[1, j].step(tr.t[:-1], tr.u, where="post", color=couleur, lw=1.1, alpha=alpha)
                if tr.fin != "duree_atteinte":
                    axes[0, j].plot(tr.t[-1], theta[-1], "x", color=couleur, ms=8, mew=2)
    seuil, u_max = cfg.evaluation.seuil_stabilisation_deg, cfg.simulation.u_max
    for j, angle in enumerate(angles):
        axes[0, j].set_title(f"θ0 = {angle:g}°")
        axes[0, j].axhspan(-seuil, seuil, color=GRIS, alpha=0.15, lw=0)
        for signe in (1, -1):
            axes[1, j].axhline(signe * u_max, color=GRIS, ls="--", lw=0.8)
        axes[1, j].set_xlabel("temps (s)")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("θ (°)")
    axes[1, 0].set_ylabel("u appliquée (N)")
    poignees = [plt.Line2D([], [], color=COULEURS_FAMILLES[n], lw=2, label=n) for n in series]
    poignees += [
        plt.Line2D([], [], color=GRIS, ls="", marker="x", label="arrêt : |θ| > 90° ou |x| > 4 m"),
        plt.Rectangle((0, 0), 1, 1, color=GRIS, alpha=0.15, label=f"bande ±{seuil:g}°"),
    ]
    fig.legend(handles=poignees, loc="outside lower center", ncol=len(poignees))
    s = cfg.simulation
    fig.suptitle(
        f"Trajectoires sur le modèle non linéaire (courbes non écrêtées)\n"
        f"|u| ≤ {u_max:g} N, dt = {s.dt:g} s, Euler, PPO : une courbe par graine",
        fontsize=11,
    )
    return _enregistrer(fig, chemin)


def figure_reussite(resume, limite_deg: float, chemin) -> Path:
    """Taux de réussite par |θ0| (angles ± regroupés), une barre par contrôleur."""
    familles = [f for f in COULEURS_FAMILLES if f in set(resume["groupe"])]
    angles = sorted(resume["abs_theta0_deg"].unique())
    x = np.arange(len(angles))
    largeur = 0.8 / len(familles)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for k, nom in enumerate(familles):
        sous = resume[resume["groupe"] == nom].set_index("abs_theta0_deg").reindex(angles)
        pos = x - 0.4 + largeur * (k + 0.5)
        ax.bar(pos, sous["taux_reussite"].fillna(0), largeur * 0.92, color=COULEURS_FAMILLES[nom], label=nom)
        for p, ok, n in zip(pos, sous["n_reussites"], sous["n_essais"], strict=True):
            if not np.isnan(n):
                ax.annotate(f"{int(ok)}/{int(n)}", (p, 0), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7)  # fmt: skip
    irrattrapables = [i for i, a in enumerate(angles) if a > limite_deg]
    if irrattrapables:
        ax.axvspan(irrattrapables[0] - 0.5, len(angles) - 0.5, color=GRIS, alpha=0.12, lw=0)
        texte = f"|θ0| > {limite_deg:.2f}° :\nirrattrapable avec la\nforce disponible"
        ax.annotate(texte, (irrattrapables[0] - 0.45, 0.62), fontsize=8, color=GRIS)
    ax.set_xticks(x, [f"±{a:g}°" for a in angles])
    ax.set(xlabel="angle initial (θ̇0 = 0)", ylabel="taux de réussite", ylim=(0, 1.08))
    ax.set_title("Réussite = |θ| ≤ 0,5° tenu jusqu'à la fin (≥ 1 s), sans dépasser |θ| = 90° ni |x| = 4 m, en 10 s")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _enregistrer(fig, chemin)


def figure_bassin(cartes: dict[str, np.ndarray], thetas, omegas, limite_deg: float, zones: dict, chemin) -> Path:
    """Cartes de réussite sur la grille (θ0, θ̇0). Valeur = fraction des graines qui réussissent.

    Zone hachurée : états dont on a prouvé qu'ils sont irrattrapables avec |u| ≤ u_max
    (|θ0| > θ* et θ̇0 du même signe que θ0). Ailleurs, la limite exacte n'est pas connue.
    """
    fig, axes = plt.subplots(1, len(cartes), figsize=(5.2 * len(cartes), 4.4), sharey=True, layout="constrained",
                             squeeze=False)  # fmt: skip
    pas_t, pas_w = thetas[1] - thetas[0], omegas[1] - omegas[0]
    etendue = (thetas[0] - pas_t / 2, thetas[-1] + pas_t / 2, omegas[0] - pas_w / 2, omegas[-1] + pas_w / 2)
    for ax, (nom, carte) in zip(axes[0], cartes.items(), strict=True):
        im = ax.imshow(carte.T, origin="lower", extent=etendue, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        for signe in (1, -1):
            ax.fill_between([signe * limite_deg, signe * etendue[1]], 0, signe * etendue[3], facecolor="none",
                            hatch="///", edgecolor=GRIS, lw=0)  # fmt: skip
            ax.axvline(signe * limite_deg, color=GRIS, ls="--", lw=1)
            if nom in zones:
                ax.axvline(signe * zones[nom], color="#eb6834", ls=":", lw=1.5)
        ax.set(title=nom, xlabel="θ0 (°)")
    axes[0, 0].set_ylabel("θ̇0 (rad/s)")
    fig.colorbar(im, ax=axes[0, -1], label="fraction de réussite")
    poignees = [
        plt.Line2D([], [], color=GRIS, ls="--", label=f"θ* = ±{limite_deg:.2f}°"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", hatch="///", edgecolor=GRIS, label="irrattrapable (prouvé)"),
        plt.Line2D([], [], color="#eb6834", ls=":", lw=1.5, label="zone d'entraînement PPO"),
    ]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3)
    fig.suptitle("Bassin d'attraction (x0 = ẋ0 = 0), critère de réussite des tableaux", fontsize=11)
    return _enregistrer(fig, chemin)


def figure_apprentissage(dossiers: dict[str, list[Path]], chemin, fenetre: int = 100) -> Path:
    """Récompense et longueur des épisodes pendant l'entraînement (moyenne glissante)."""
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for nom, runs in dossiers.items():
        for run in runs:
            df = pd.read_csv(run / "monitor.csv", skiprows=1)
            pas = df["l"].cumsum()
            for ax, col in zip(axes, ["r", "l"], strict=True):
                ax.plot(pas, df[col].rolling(fenetre, min_periods=1).mean(), color=COULEURS_FAMILLES[nom], lw=1.2,
                        alpha=0.8, label=nom if run == runs[0] else None)  # fmt: skip
    axes[0].set(title=f"Récompense par épisode (moyenne sur {fenetre} épisodes)", xlabel="pas d'entraînement")
    axes[1].set(title="Longueur des épisodes (max 250 pas = 5 s)", xlabel="pas d'entraînement", ylabel="pas")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Courbes d'apprentissage PPO — une courbe par graine", fontsize=11)
    return _enregistrer(fig, chemin)


def figure_integrateurs(derive: dict, trajs: dict, chemin) -> Path:
    """Euler explicite vs RK4 : dérive d'énergie sans commande, et trajectoires en boucle fermée.

    ``derive`` : {"euler": (t, E/E0 − 1), "rk4": (...)} ;
    ``trajs`` : {(controleur, angle): {"euler": Trajectoire, "rk4": Trajectoire}}.
    """
    fig, axes = plt.subplots(1, 1 + len(trajs), figsize=(3.6 * (1 + len(trajs)), 3.6), layout="constrained")
    styles = {"euler": dict(color="#2a78d6", lw=1.4), "rk4": dict(color="#eb6834", lw=1.4, ls="--")}
    for nom, (t, e) in derive.items():
        axes[0].plot(t, 100 * e, label=nom.upper() if nom == "rk4" else "Euler", **styles[nom])
    axes[0].set(title="Énergie sans commande (θ0 = 60°)", xlabel="temps (s)", ylabel="écart relatif d'énergie (%)")
    axes[0].legend(fontsize=8)
    for ax, ((nom, angle), paire) in zip(axes[1:], trajs.items(), strict=True):
        for integ, tr in paire.items():
            ax.plot(tr.t, np.rad2deg(tr.etats[:, 2]), **styles[integ])
            if tr.fin != "duree_atteinte":
                ax.plot(tr.t[-1], np.rad2deg(tr.etats[-1, 2]), "x", color=styles[integ]["color"], ms=8, mew=2)
        ax.set(title=f"{nom}, θ0 = {angle:g}°", xlabel="temps (s)", ylabel="θ (°)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Euler explicite (défaut, comme le notebook) vs RK4 — dt = 0,02 s", fontsize=11)
    return _enregistrer(fig, chemin)
