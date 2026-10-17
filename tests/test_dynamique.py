import numpy as np
import pytest

from pendule.config import charger
from pendule.dynamique import derivees, linearisation, pas_euler, pas_rk4
from pendule.lqr import gain_lqr

CFG = charger()
P = CFG.physique
DT = CFG.simulation.dt


def dynamique_notebook(state, u, M=1.0, m=0.1, l=0.5, g=9.81, dt=0.02):
    """Copie de PenduleInverseEnv._dynamique (archive/PIR-2.ipynb), en float64."""
    x, dx, theta, dtheta = state
    D = l * (M + m * np.sin(theta) ** 2)
    rhs1 = u + m * l * dtheta**2 * np.sin(theta)
    ddx = (l * rhs1 - m * l * np.cos(theta) * g * np.sin(theta)) / D
    ddtheta = ((M + m) * g * np.sin(theta) - np.cos(theta) * rhs1) / D
    return np.array([x + dx * dt, dx + ddx * dt, theta + dtheta * dt, dtheta + ddtheta * dt])


def test_euler_identique_au_notebook():
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = rng.uniform([-1, -2, -1.5, -5], [1, 2, 1.5, 5])
        u = rng.uniform(-4, 4)
        np.testing.assert_allclose(pas_euler(s, u, DT, P), dynamique_notebook(s, u), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pas", [pas_euler, pas_rk4])
@pytest.mark.parametrize("theta", [0.0, np.pi])
def test_equilibre_sans_commande(pas, theta):
    s = np.array([0.0, 0.0, theta, 0.0])
    for _ in range(500):
        s = pas(s, 0.0, DT, P)
    np.testing.assert_allclose(s, [0.0, 0.0, theta, 0.0], atol=1e-9)


def test_lineaire_et_non_lineaire_coincident_aux_petits_angles():
    A, B = linearisation(P)
    for amplitude in (1e-2, 1e-3):
        s = amplitude * np.array([0.3, -0.5, 1.0, 0.7])
        u = amplitude * 2.0
        ecart = np.linalg.norm(derivees(s, u, P) - (A @ s + B.ravel() * u))
        assert ecart < 10 * amplitude**2  # l'écart est d'ordre 2


def test_linearisation_egale_jacobien_numerique():
    A, B = linearisation(P)
    eps = 1e-6
    A_num = np.column_stack([(derivees(eps * e, 0.0, P) - derivees(-eps * e, 0.0, P)) / (2 * eps) for e in np.eye(4)])
    B_num = (derivees(np.zeros(4), eps, P) - derivees(np.zeros(4), -eps, P)) / (2 * eps)
    np.testing.assert_allclose(A, A_num, atol=1e-6)
    np.testing.assert_allclose(B.ravel(), B_num, atol=1e-6)


def test_valeurs_propres_boucle_ouverte_diapo6():
    A, _ = linearisation(P)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(A).real), [-4.646, 0, 0, 4.646], atol=1e-3)


def test_controlabilite_diapo8():
    A, B = linearisation(P)
    kalman = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(4)])
    assert np.linalg.matrix_rank(kalman) == 4


def test_gain_et_valeurs_propres_lqr_diapos9_10():
    A, B = linearisation(P)
    K = gain_lqr(CFG)
    np.testing.assert_allclose(K, [-3.16, -4.25, -38.75, -8.22], atol=0.005)
    vp = np.linalg.eigvals(A - B @ K[None, :])
    vp = vp[np.lexsort((vp.imag, vp.real))]
    np.testing.assert_allclose(vp, [-5.04 - 1.99j, -5.04 + 1.99j, -1.06 - 1.00j, -1.06 + 1.00j], atol=0.005)
