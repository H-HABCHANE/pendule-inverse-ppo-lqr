import numpy as np
import pytest

from pendule.config import charger, charger_modele
from pendule.dynamique import angle_limite_rattrapage, derivees
from pendule.lqr import ControleurLQR
from pendule.metriques import NON_STABILISE, STABILISE, metriques, nombre_traversees, temps_etablissement
from pendule.simulation import ERREUR_NUMERIQUE, LIMITES_DEPASSEES, etat_initial, reussites_lot, simuler

CFG = charger()
T = np.arange(0, 10.0001, 0.02)


def test_etablissement_simple():
    assert temps_etablissement(T, T >= 1.0, 1.0) == pytest.approx(1.0)


def test_premier_passage_ne_compte_pas():
    # Entre dans la bande à 0,5 s, en sort à 0,8 s, y revient définitivement à 1,2 s.
    dans = ((T >= 0.5) & (T < 0.8)) | (T >= 1.2)
    assert temps_etablissement(T, dans, 1.0) == pytest.approx(1.2)


def test_hors_bande_a_la_fin():
    assert np.isnan(temps_etablissement(T, T < 9.0, 1.0))


def test_maintien_trop_court():
    dans = T >= 9.5
    assert np.isnan(temps_etablissement(T, dans, 1.0))
    assert temps_etablissement(T, dans, 0.5) == pytest.approx(9.5)


def test_nombre_traversees():
    theta = np.deg2rad(np.array([5, 2, 0.1, -0.1, -3, -1, 0.2, 4, 0.3]))
    assert nombre_traversees(theta, np.deg2rad(0.5)) == 2


def test_issues():
    lqr = ControleurLQR(CFG)
    assert metriques(simuler(lqr, etat_initial(5.0), CFG), CFG)["issue"] == STABILISE
    assert metriques(simuler(lqr, etat_initial(18.0), CFG), CFG)["issue"] == LIMITES_DEPASSEES
    # Durée trop courte pour se stabiliser, sans sortir des limites
    courte = CFG.modifier("evaluation", duree_s=0.5)
    assert metriques(simuler(lqr, etat_initial(15.0), courte), courte)["issue"] == NON_STABILISE
    nan = simuler(lqr, np.array([0.0, np.nan, 0.1, 0.0]), CFG)
    assert nan.fin == ERREUR_NUMERIQUE and metriques(nan, CFG)["issue"] == ERREUR_NUMERIQUE


def test_saturation_est_la_cause_de_l_echec_du_lqr():
    illimite = CFG.modifier("simulation", u_max=float("inf"))
    for angle in (18.0, 20.0):
        assert metriques(simuler(ControleurLQR(CFG), etat_initial(angle), illimite), illimite)["reussite"]


def test_limite_physique():
    p, u_max = CFG.physique, CFG.simulation.u_max
    limite = angle_limite_rattrapage(p, u_max)
    assert np.rad2deg(limite) == pytest.approx(20.34, abs=0.01)
    # Juste au-delà : θ̈ > 0 même avec la force maximale dans le bon sens
    s = np.array([0.0, 0.0, limite + np.deg2rad(0.1), 0.0])
    assert derivees(s, u_max, p)[3] > 0
    # Juste en deçà : la force maximale fait redescendre le pendule
    s[2] = limite - np.deg2rad(0.1)
    assert derivees(s, u_max, p)[3] < 0


def test_version_vectorisee_identique_a_la_scalaire():
    lqr = ControleurLQR(CFG)
    S0 = np.array([etat_initial(a, w) for a in (-19.0, -5.0, 0.0, 12.0, 17.0, 22.0) for w in (-1.0, 0.0, 0.7)])
    attendu = [metriques(simuler(lqr, s0, CFG), CFG)["reussite"] for s0 in S0]
    np.testing.assert_array_equal(reussites_lot(lqr, S0, CFG), attendu)


def test_modele_garde_son_environnement_mais_prend_le_protocole_courant(tmp_path):
    ancien = CFG.modifier("environnement", zone_entrainement_deg=10.0).modifier("evaluation", duree_s=6.0)
    ancien.sauvegarder(tmp_path / "config.yaml")
    cfg = charger_modele(tmp_path / "config.yaml", CFG)
    assert cfg.environnement.zone_entrainement_deg == 10.0
    assert cfg.evaluation == CFG.evaluation
