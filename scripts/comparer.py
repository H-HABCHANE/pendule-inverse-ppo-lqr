"""Agrège evaluation.csv, produit les tableaux de synthèse et les figures de comparaison.

Exemple :
    python scripts/comparer.py --evaluation resultats/evaluation.csv --modeles runs/zone20 runs/zone10

Sorties : resultats/resume.csv, resultats/balayage_R.csv, resultats/diapos_vs_recalcule.md,
figures/reussite.png, figures/comparaison_trajectoires.png, figures/apprentissage.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pendule.config import CONFIG_DEFAUT, charger
from pendule.dynamique import angle_limite_rattrapage
from pendule.figures import figure_apprentissage, figure_comparaison, figure_reussite
from pendule.lqr import ControleurLQR
from pendule.simulation import etat_initial, simuler

# Valeurs lues dans les diapos de soutenance (docs/), citées pour comparaison uniquement.
DIAPOS = {
    ("LQR", 5.0): {"diapo 12": "0,31", "diapo 24": "0,31"},
    ("LQR", 15.0): {"diapo 12": "0,44", "diapo 24": "0,37"},
    ("LQR", 18.0): {"diapo 12": "diverge", "diapo 24": "diverge"},
    ("LQR", 20.0): {"diapo 12": "—", "diapo 24": "diverge"},
    ("PPO", 5.0): {"diapo 12": "—", "diapo 24": "0,56"},
    ("PPO", 15.0): {"diapo 12": "—", "diapo 24": "0,64"},
    ("PPO", 18.0): {"diapo 12": "—", "diapo 24": "0,74"},
    ("PPO", 20.0): {"diapo 12": "—", "diapo 24": "1,08 (θ max 1628°)"},
}

METRIQUES = [
    "t_etablissement_s",
    "t_stab_premier_passage_s",
    "depassement_deg",
    "nb_traversees",
    "u_max_N",
    "energie_u2_N2s",
    "x_final_m",
]


def groupe(controleur: str, r_ref: float) -> str | None:
    if controleur == f"LQR saturé (R={r_ref:g})":
        return "LQR saturé"
    if controleur.startswith("LQR non saturé"):
        return "LQR non saturé"
    if controleur.startswith("PPO"):
        return controleur
    return None


def resumer(df: pd.DataFrame, cles: list[str]) -> pd.DataFrame:
    """Taux de réussite, et moyenne ± écart-type des métriques sur les essais réussis."""
    g = df.groupby(cles, sort=False)
    resume = g.agg(n_essais=("reussite", "size"), n_reussites=("reussite", "sum"))
    resume["taux_reussite"] = resume["n_reussites"] / resume["n_essais"]
    reussis = df[df["reussite"]].groupby(cles, sort=False)
    for m in METRIQUES:
        resume[f"{m}_moy"] = reussis[m].mean()
        resume[f"{m}_std"] = reussis[m].std()
    resume["issues"] = g["issue"].agg(lambda s: ", ".join(f"{k}={v}" for k, v in s.value_counts().items()))
    return resume.reset_index()


def fmt(moy, std, n) -> str:
    if n == 0 or np.isnan(moy):
        return "—"
    return f"{moy:.2f}" if n == 1 or np.isnan(std) else f"{moy:.2f} ± {std:.2f}"


def tableau_diapos(df: pd.DataFrame, r_ref: float) -> str:
    lignes = [
        "| Contrôleur | θ0 | Diapo 12 | Diapo 24 | 1er passage < 0,5° (méthode d'origine) | "
        "Temps d'établissement (bande tenue) | Issue |",
        "|---|---|---|---|---|---|---|",
    ]
    for (famille, angle), cite in DIAPOS.items():
        nom = f"LQR saturé (R={r_ref:g})" if famille == "LQR" else "PPO ±20°"
        sous = df[(df["controleur"] == nom) & (df["theta0_deg"] == angle)]
        if sous.empty:
            continue
        pp, te = sous["t_stab_premier_passage_s"], sous["t_etablissement_s"]
        n_ok = int(sous["reussite"].sum())
        passage = fmt(pp.mean(), pp.std(), pp.notna().sum()) if pp.notna().any() else "jamais"
        etab = fmt(te.mean(), te.std(), n_ok) if n_ok else "non stabilisé"
        issues = ", ".join(f"{k}={v}" for k, v in sous["issue"].value_counts().items())
        lignes.append(
            f"| {nom} | {angle:g}° | {cite['diapo 12']} | {cite['diapo 24']} | {passage} | {etab} | {issues} |"
        )
    return "\n".join(lignes) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DEFAUT)
    parser.add_argument("--evaluation", default="resultats/evaluation.csv")
    parser.add_argument("--modeles", nargs="*", default=[], help="dossiers seed_*/ pour les trajectoires")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    cfg = charger(args.config)
    r_ref = cfg.lqr.R
    chemin = Path(args.evaluation)
    df = pd.read_csv(chemin)
    df["groupe"] = [groupe(c, r_ref) for c in df["controleur"]]
    df["abs_theta0_deg"] = df["theta0_deg"].abs()

    resume = resumer(df, ["controleur", "theta0_deg"])
    resume.to_csv(chemin.with_name("resume.csv"), index=False, float_format="%.4f")
    balayage = resumer(df[df["controleur"].str.startswith("LQR saturé")], ["controleur", "abs_theta0_deg"])
    balayage.to_csv(chemin.with_name("balayage_R.csv"), index=False, float_format="%.4f")
    chemin.with_name("diapos_vs_recalcule.md").write_text(tableau_diapos(df, r_ref))

    limite = np.rad2deg(angle_limite_rattrapage(cfg.physique, cfg.simulation.u_max))
    par_groupe = resumer(df[df["groupe"].notna()], ["groupe", "abs_theta0_deg"])
    figure_reussite(par_groupe, limite, Path(args.figures) / "reussite.png")

    if args.modeles:
        from pendule.ppo import charger_agent

        angles = [5.0, 15.0, 18.0, 20.0]
        series = {"LQR saturé": {a: [simuler(ControleurLQR(cfg), etat_initial(a), cfg)] for a in angles}}
        dossiers = {}
        for d in args.modeles:
            runs = sorted(p.parent for p in Path(d).glob("seed_*/model.zip"))
            agents = [charger_agent(r, cfg) for r in runs]
            nom = f"PPO ±{agents[0][1].environnement.zone_entrainement_deg:g}°"
            dossiers[nom] = runs
            if nom == "PPO ±20°":
                series[nom] = {a: [simuler(ag, etat_initial(a), c) for ag, c in agents] for a in angles}
        figure_comparaison(series, cfg, Path(args.figures) / "comparaison_trajectoires.png")
        figure_apprentissage(dossiers, Path(args.figures) / "apprentissage.png")

    colonnes = ["controleur", "theta0_deg", "n_reussites", "n_essais", "t_etablissement_s_moy", "t_etablissement_s_std"]
    with pd.option_context("display.width", 200):
        print(resume[colonnes].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {chemin.with_name('resume.csv')}, {chemin.with_name('diapos_vs_recalcule.md')}, {args.figures}/")


if __name__ == "__main__":
    main()
