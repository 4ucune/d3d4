"""Integración con un trim 3D real grande (Tumaco: 424×300, KMAX=10 sigma,
malla girada, 961 registros, ~80 GB). Opcional: se ejecuta solo si existen

    D3D_TUMACO_DEF = ruta del .def
    D3D_TUMACO_DAT = ruta del .dat   (en el caso original: '...Tumaco.dat.exe')

Verifica las convenciones de la malla con la propia geometría del archivo
(sin depender de MATLAB) y los campos derivados contra traducciones literales.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import d3d

DEF = os.environ.get("D3D_TUMACO_DEF")
DAT = os.environ.get("D3D_TUMACO_DAT")
pytestmark = pytest.mark.skipif(not (DEF and DAT and Path(DEF).exists() and Path(DAT).exists()),
                                reason="D3D_TUMACO_DEF / D3D_TUMACO_DAT no definidos")


@pytest.fixture(scope="module")
def tumaco():
    data = d3d.read(DEF, dat_path=DAT)
    yield data
    data.close()


def _c(data, name):
    return data[name].raw[0]


def test_alfas_es_la_direccion_m(tumaco):
    """ALFAS = ángulo de la dirección M (eje 1 de (N, M)) respecto a +X."""
    xz, yz, alf = _c(tumaco, "XZ"), _c(tumaco, "YZ"), _c(tumaco, "ALFAS").astype(float)
    act = np.abs(_c(tumaco, "KCS")) == 1
    ok = act.copy()
    ok[:, 1:-1] &= act[:, 2:] & act[:, :-2]
    ok[:, [0, -1]] = False
    ang = np.degrees(np.arctan2(yz[:, 2:] - yz[:, :-2], xz[:, 2:] - xz[:, :-2]))
    d = (ang - alf[:, 1:-1] + 180) % 360 - 180
    assert np.median(np.abs(d[ok[:, 1:-1]])) < 0.01
    assert np.abs(d[ok[:, 1:-1]]).max() < 0.5


def test_malla_escalonada(tumaco):
    """U1[n,m] entre los centros m y m+1; V1[n,m] entre n y n+1 (DPU0/DPV0 = min de DPS0)."""
    dps = _c(tumaco, "DPS0")
    for nombre, kc, vecino in (("DPU0", "KCU", dps[:, 1:]), ("DPV0", "KCV", dps[1:, :])):
        dp = _c(tumaco, nombre)
        sl = (slice(None), slice(0, -1)) if nombre == "DPU0" else (slice(0, -1), slice(None))
        m = (_c(tumaco, kc)[sl] != 0) & (np.abs(dp[sl]) < 900)
        np.testing.assert_array_equal(dp[sl][m], np.minimum(dps[sl], vecino)[m])


def test_formas_y_tiempo(tumaco):
    assert tumaco["U1"].shape == (961, 424, 300, 10)
    assert tumaco["KCS"].shape == (1, 424, 300)
    t = d3d.get_time(tumaco)
    assert len(t) == 961
    assert t[0] == np.datetime64("2020-06-10T00:00")
    assert (np.diff(t) == np.timedelta64(1, "h")).all()


def test_velocidad_contra_literal(tumaco):
    i = 480
    u = tumaco["U1"].raw[i].astype(float)
    v = tumaco["V1"].raw[i].astype(float)
    u[u == -999] = np.nan
    v[v == -999] = np.nan
    alf = np.deg2rad(_c(tumaco, "ALFAS").astype(float))
    kcs = _c(tumaco, "KCS")
    x, y = d3d.velocity_xy(tumaco, index=i)

    def media(a, b):
        vals = [q for q in (a, b) if not np.isnan(q)]
        return sum(vals) / max(len(vals), 1)

    rng = np.random.default_rng(0)
    for n, m, k in zip(rng.integers(1, 424, 500), rng.integers(1, 300, 500), rng.integers(0, 10, 500)):
        if abs(kcs[n, m]) != 1:
            assert np.isnan(x[n, m, k])
            continue
        U = media(u[n, m, k], u[n, m - 1, k])
        V = media(v[n, m, k], v[n - 1, m, k])
        assert x[n, m, k] == pytest.approx(U * np.cos(alf[n, m]) - V * np.sin(alf[n, m]), abs=1e-6)
        assert y[n, m, k] == pytest.approx(U * np.sin(alf[n, m]) + V * np.cos(alf[n, m]), abs=1e-6)


def test_water_level_mascara_seca_literal(tumaco):
    i = 480
    s1 = tumaco["S1"].raw[i].astype(np.float32)
    kfu, kfv = tumaco["KFU"][i], tumaco["KFV"][i]
    seca = ((kfu == 0) & (np.hstack([kfu[:, :1], kfu[:, :-1]]) == 0)
            & (kfv == 0) & (np.vstack([kfv[:1], kfv[:-1]]) == 0))
    esperado = np.isin(s1, [-999, 999]) | (np.abs(_c(tumaco, "KCS")) != 1) | seca
    np.testing.assert_array_equal(np.isnan(d3d.water_level(tumaco, index=i)), esperado)
    np.testing.assert_array_equal(tumaco["S1"][i], d3d.water_level(tumaco, index=i))


def test_serie_puntual_y_capa(tumaco):
    serie = tumaco["U1"][:, 200, 150, 0]
    assert serie.shape == (961,)
    np.testing.assert_array_equal(serie[[10, 500]],
                                  [tumaco["U1"][10][200, 150, 0], tumaco["U1"][500][200, 150, 0]])
    x, _ = d3d.velocity_xy(tumaco, index=slice(0, 961, 480), layer=0)
    assert x.shape == (3, 424, 300)
