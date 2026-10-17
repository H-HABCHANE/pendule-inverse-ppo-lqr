import numpy as np
import pytest

from pendule.config import CONFIG_DEFAUT, charger, verifier_compatibles
from pendule.env import PenduleInverseEnv
from pendule.lqr import ControleurLQR
from pendule.metriques import temps_premier_passage
from pendule.simulation import Trajectoire, etat_initial, simuler

CFG = charger()


def test_check_env_gymnasium():
    from gymnasium.utils.env_checker import check_env

    check_env(PenduleInverseEnv(CFG), skip_render_check=True)


def test_reset_reproductible_avec_graine():
    a, b = PenduleInverseEnv(CFG), PenduleInverseEnv(CFG)
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    for _ in range(20):
        oa, ra, *_ = a.step(np.array([1.0]))
        ob, rb, *_ = b.step(np.array([1.0]))
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb
    assert not np.array_equal(a.reset(seed=1)[0], a.reset(seed=2)[0])


def test_reset_dans_la_zone_entrainement():
    env = PenduleInverseEnv(CFG)
    zone = np.deg2rad(CFG.environnement.zone_entrainement_deg)
    assert max(abs(env.reset(seed=k)[0][2]) for k in range(300)) <= zone + 1e-6


@pytest.mark.parametrize("action, attendu", [(10.0, 4.0), (-10.0, -4.0), (1.5, 1.5)])
def test_saturation_env(action, attendu):
    env = PenduleInverseEnv(CFG)
    env.reset(options={"etat_initial": np.zeros(4)})
    assert env.step(np.array([action]))[4]["u"] == pytest.approx(attendu)


def test_saturation_simulation_lqr():
    traj = simuler(ControleurLQR(CFG), etat_initial(15.0), CFG)
    assert traj.u_demande[0] == pytest.approx(10.14, abs=0.01)  # diapo 12 : 10,1 N demandés
    assert np.max(np.abs(traj.u)) == pytest.approx(CFG.simulation.u_max)


def test_terminated_et_truncated():
    env = PenduleInverseEnv(CFG)
    env.reset(options={"etat_initial": etat_initial(24.0)})
    for _ in range(CFG.pas_par_episode):
        _, r, terminated, truncated, _ = env.step(np.array([0.0]))
        if terminated or truncated:
            break
    assert terminated and not truncated and r == CFG.environnement.recompense.penalite_arret

    env.reset(options={"etat_initial": np.zeros(4)})
    for k in range(CFG.pas_par_episode):
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert not terminated and truncated == (k == CFG.pas_par_episode - 1)


def test_temps_premier_passage():
    t = np.arange(0, 2.0001, 0.02)
    theta = np.deg2rad(np.where(t < 0.5, 5.0, 0.1))
    etats = np.zeros((len(t), 4))
    etats[:, 2] = theta
    traj = Trajectoire(t, etats, np.zeros(len(t) - 1), np.zeros(len(t) - 1), False)
    assert temps_premier_passage(traj, 0.5) == pytest.approx(0.5)


def test_config_sauvegarde_et_compatibilite(tmp_path):
    cfg = CFG.modifier("ppo", seed=7)
    cfg.sauvegarder(tmp_path / "config.yaml")
    relue = charger(tmp_path / "config.yaml")
    assert relue == cfg
    verifier_compatibles(CFG, relue)
    with pytest.raises(ValueError):
        verifier_compatibles(CFG, CFG.modifier("simulation", u_max=15.0))
    assert charger(CONFIG_DEFAUT) == CFG
