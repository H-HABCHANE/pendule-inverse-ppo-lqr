| Contrôleur | θ0 | Diapo 12 | Diapo 24 | 1er passage < 0,5° (méthode d'origine) | Temps d'établissement (bande tenue) | Issue |
|---|---|---|---|---|---|---|
| LQR saturé (R=0.1) | 5° | 0,31 | 0,31 | 0.30 | 1.38 | stabilise=1 |
| LQR saturé (R=0.1) | 15° | 0,44 | 0,37 | 0.46 | 3.96 | stabilise=1 |
| LQR saturé (R=0.1) | 18° | diverge | diverge | jamais | non stabilisé | limites_depassees=1 |
| LQR saturé (R=0.1) | 20° | — | diverge | jamais | non stabilisé | limites_depassees=1 |
| PPO ±20° | 5° | — | 0,56 | 0.38 ± 0.05 | 2.03 ± 0.34 | stabilise=3 |
| PPO ±20° | 15° | — | 0,64 | 0.53 ± 0.03 | 5.33 ± 3.07 | stabilise=3 |
| PPO ±20° | 18° | — | 0,74 | 0.69 ± 0.03 | 5.94 ± 0.08 | stabilise=2, limites_depassees=1 |
| PPO ±20° | 20° | — | 1,08 (θ max 1628°) | 1.11 ± 0.06 | non stabilisé | limites_depassees=3 |
