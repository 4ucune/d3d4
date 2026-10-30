"""`d3d.transforms` contra implementaciones independientes con bucles
(traducción elemento a elemento de uv2cen.m, cur2ca.m y la máscara seca de
d3d_trimfil.m) y propiedades geométricas de la rotación."""
import numpy as np
import pytest

from d3d import transforms as tr

rng = np.random.default_rng(42)
N, M, K, T = 5, 7, 3, 2


def _campo(shape, nan_frac=0.2):
    a = rng.normal(size=shape).astype(np.float32)
    a[rng.random(shape) < nan_frac] = np.nan
    return a


def _mean_loop(a, b):
    vals = [q for q in (a, b) if not np.isnan(q)]
    return sum(vals) / max(len(vals), 1)


def _uv2cen_loop(u, v):
    """u, v: (N, M). Bucles explícitos según uv2cen.m."""
    U = np.full(u.shape, np.nan)
    V = np.full(v.shape, np.nan)
    for n in range(u.shape[0]):
        for m in range(u.shape[1]):
            if m > 0:
                U[n, m] = _mean_loop(u[n, m], u[n, m - 1])
            if n > 0:
                V[n, m] = _mean_loop(v[n, m], v[n - 1, m])
    return U, V


def test_uv2cen_2d_igual_a_bucles():
    u, v = _campo((N, M)), _campo((N, M))
    U, V = tr.uv2cen(u, v, n_axis=0, m_axis=1)
    Ul, Vl = _uv2cen_loop(u.astype(float), v.astype(float))
    np.testing.assert_allclose(U, Ul, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(V, Vl, rtol=1e-6, equal_nan=True)
    assert U.dtype == np.float32


def test_uv2cen_4d_tnmk_igual_a_2d_por_rebanada():
    u, v = _campo((T, N, M, K)), _campo((T, N, M, K))
    U, V = tr.uv2cen(u, v, n_axis=1, m_axis=2)
    for t in range(T):
        for k in range(K):
            Ul, Vl = _uv2cen_loop(u[t, :, :, k].astype(float), v[t, :, :, k].astype(float))
            np.testing.assert_allclose(U[t, :, :, k], Ul, rtol=1e-6, equal_nan=True)
            np.testing.assert_allclose(V[t, :, :, k], Vl, rtol=1e-6, equal_nan=True)


def test_uv2cen_ambas_nan_da_cero_como_matlab():
    u = np.array([[np.nan, np.nan]], np.float32)
    U, _ = tr.uv2cen(u, u, n_axis=0, m_axis=1)
    assert np.isnan(U[0, 0]) and U[0, 1] == 0


def test_cur2ca_igual_a_bucles_y_ejes():
    u, v = _campo((T, N, M, K), 0), _campo((T, N, M, K), 0)
    alf = rng.uniform(-180, 180, size=(N, M))
    x, y = tr.cur2ca(u, v, alf, n_axis=1, m_axis=2)
    for t, n, m, k in [(0, 0, 0, 0), (1, 4, 6, 2), (0, 2, 3, 1)]:
        a = np.deg2rad(alf[n, m])
        assert x[t, n, m, k] == pytest.approx(u[t, n, m, k] * np.cos(a) - v[t, n, m, k] * np.sin(a), rel=1e-5)
        assert y[t, n, m, k] == pytest.approx(u[t, n, m, k] * np.sin(a) + v[t, n, m, k] * np.cos(a), rel=1e-5)


def test_cur2ca_conserva_modulo_y_orientacion():
    u, v = _campo((N, M), 0), _campo((N, M), 0)
    alf = rng.uniform(-180, 180, size=(N, M))
    x, y = tr.cur2ca(u, v, alf, n_axis=0, m_axis=1)
    np.testing.assert_allclose(np.hypot(x, y), np.hypot(u, v), rtol=1e-5)
    # ángulo cartesiano = ángulo en la malla + ALFAS
    d = np.degrees(np.arctan2(y, x) - np.arctan2(v, u)) - alf
    np.testing.assert_allclose((d + 180) % 360 - 180, 0, atol=1e-3)


def test_cur2ca_casos_exactos():
    one = np.ones((1, 1))
    x, y = tr.cur2ca(one, 0 * one, np.full((1, 1), 90.0), n_axis=0, m_axis=1)
    assert x[0, 0] == pytest.approx(0, abs=1e-12) and y[0, 0] == pytest.approx(1)
    x, y = tr.cur2ca(0 * one, one, np.full((1, 1), 90.0), n_axis=0, m_axis=1)
    assert x[0, 0] == pytest.approx(-1) and y[0, 0] == pytest.approx(0, abs=1e-12)


def test_cur2ca_ejes_n_m_en_otro_orden():
    """(M, N) con n_axis=1, m_axis=0 da lo mismo que (N, M) transpuesto."""
    u, v = _campo((N, M), 0), _campo((N, M), 0)
    alf = rng.uniform(-90, 90, size=(N, M))
    x, y = tr.cur2ca(u, v, alf, n_axis=0, m_axis=1)
    xt, yt = tr.cur2ca(u.T, v.T, alf, n_axis=1, m_axis=0)
    np.testing.assert_allclose(xt, x.T)
    np.testing.assert_allclose(yt, y.T)


def test_dry_mask_igual_a_bucles():
    kfu = rng.integers(0, 2, size=(T, N, M))
    kfv = rng.integers(0, 2, size=(T, N, M))
    got = tr.dry_mask(kfu, kfv, n_axis=1, m_axis=2)
    for t in range(T):
        for n in range(N):
            for m in range(M):
                esperado = (kfu[t, n, m] == 0 and kfu[t, n, max(m - 1, 0)] == 0
                            and kfv[t, n, m] == 0 and kfv[t, max(n - 1, 0), m] == 0)
                assert got[t, n, m] == esperado, (t, n, m)


def test_apply_mask_y_active_mask():
    kcs = np.array([[0, 1, 2], [-1, 1, 1]])
    act = tr.active_mask(kcs)
    np.testing.assert_array_equal(act, [[False, True, False], [True, True, True]])
    a = np.ones((T, 2, 3, K), np.float32)
    out = tr.apply_mask(a, act, n_axis=1, m_axis=2)
    assert np.isnan(out[:, 0, 0, :]).all() and np.isnan(out[:, 0, 2, :]).all()
    assert (out[:, 1] == 1).all()
    assert (a == 1).all()                      # no modifica la entrada
    with pytest.raises(ValueError):
        tr.apply_mask(a, act.T, n_axis=1, m_axis=2)


def test_invalid_to_nan():
    a = np.array([1, -999, 999], np.int32)
    out = tr.invalid_to_nan(a, sentinels=(-999, 999))
    assert out.dtype == np.float64 and np.isnan(out[1:]).all() and out[0] == 1
    f = np.array([-999.0, 2.0], np.float32)
    out = tr.invalid_to_nan(f)
    assert out.dtype == np.float32 and np.isnan(out[0]) and f[0] == -999
