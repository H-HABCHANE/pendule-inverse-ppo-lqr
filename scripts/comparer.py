"""Agrège un fichier evaluation.csv : moyenne ± écart-type sur les graines, par contrôleur et par angle.

Exemple :
    python scripts/comparer.py resultats/evaluation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def agreger(df: pd.DataFrame) -> pd.DataFrame:
    colonnes = [
        c
        for c in df.columns
        if c not in ("controleur", "graine", "theta0_deg") and pd.api.types.is_numeric_dtype(df[c])
    ]
    groupes = df.groupby(["controleur", "theta0_deg"], sort=False)
    resume = groupes[colonnes].agg(["mean", "std"])
    resume.columns = [f"{c}_{stat}" for c, stat in resume.columns]
    resume.insert(0, "n_graines", groupes.size())
    return resume.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evaluation", help="fichier evaluation.csv produit par evaluer.py")
    args = parser.parse_args()
    chemin = Path(args.evaluation)
    resume = agreger(pd.read_csv(chemin))
    resume.to_csv(chemin.with_name("resume.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(resume.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {chemin.with_name('resume.csv')}")


if __name__ == "__main__":
    main()
