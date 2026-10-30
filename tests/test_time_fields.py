"""Eje de tiempo (`d3d.get_time`) y campos (`d3d.water_level`, `d3d.velocity_xy`)."""
import numpy as np
import pytest

import d3d
from _nefis_writer import NefisWriter, pack


# ---------------- tiempo ----------------

def _segundos_ref(ref, info_group, counter, const):
    n = ref.grpdat[info_group]["size"][0]
    c = np.array([ref.read(info_group, counter, i)[0] for i in range(1, n + 1)], float)
    dt = float(str(np.float32(ref.read(const, "DT", 1)[0])))
    tunit = float(str(np.float32(ref.read(const, "TUNIT", 1)[0])))
    return c * dt * tunit


def test_get_time_trim(trim_sed, ref_trim_sed):
    seg = _segundos_ref(ref_trim_sed, "map-info-series", "ITMAPC", "map-const")
    np.testing.assert_allclose(d3d.get_time(trim_sed, as_seconds=True), seg)
    t = d3d.get_time(trim_sed)
    assert t.dtype == np.dtype("datetime64[ms]")
    assert t[0] == np.datetime64("2001-06-01") + np.timedelta64(int(round(seg[0] * 1000)), "ms")
    assert len(t) == trim_sed["S1"].shape[0]


def test_get_time_grupos_sedimento_y_promedio(trim_sed, ref_trim_sed):
    seg = _segundos_ref(ref_trim_sed, "map-infsed-serie", "ITMAPS", "map-const")
    np.testing.assert_allclose(d3d.get_time(trim_sed, "map-sed-series", as_seconds=True), seg)
    seg_avg = _segundos_ref(ref_trim_sed, "map-infavg-serie", "ITAVGS", "map-const")
    np.testing.assert_allclose(d3d.get_time(trim_sed, "map-avg-series", as_seconds=True), seg_avg)
    assert d3d.get_time(trim_sed, "map-const")[0] == np.datetime64("2001-06-01")


def test_get_time_trih(data_dir):
    from _ref_nefis import RefNefis
    ref = RefNefis(data_dir / "trih-sed.def", data_dir / "trih-sed.dat")
    with d3d.read(str(data_dir / "trih-sed.dat")) as data:
        seg = _segundos_ref(ref, "his-info-series", "ITHISC", "his-const")
        np.testing.assert_allclose(d3d.get_time(data, "his-series", as_seconds=True), seg)


def test_get_time_morfologico(trim_sed, ref_trim_sed):
    n = ref_trim_sed.grpdat["map-infsed-serie"]["size"][0]
    dias = np.array([ref_trim_sed.read("map-infsed-serie", "MORFT", i)[0] for i in range(1, n + 1)])
    np.testing.assert_allclose(d3d.get_time(trim_sed, kind="morpho", as_seconds=True), dias * 86400)


def test_get_time_errores(trim_sed):
    with pytest.raises(KeyError):
        d3d.get_time(trim_sed, "no-existe")
    with pytest.raises(ValueError):
        d3d.get_time(trim_sed, kind="otro")


def test_get_time_sintetico_hora_y_dt_float32(tmp_path):
    """ITDATE con HHMMSS ≠ 0 y DT = 0.2 almacenado como REAL*4."""
    w = NefisWriter()
    w.element("ITDATE", "INTEGER", 4, [2])
    w.element("DT", "REAL", 4, [1])
    w.element("TUNIT", "REAL", 4, [1])
    w.element("ITMAPC", "INTEGER", 4, [1])
    w.cell("map-const", ["ITDATE", "DT", "TUNIT"])
    w.cell("map-info-series", ["ITMAPC"])
    w.group("map-const", "map-const", [0])
    w.group("map-info-series", "map-info-series", [0])
    w.data_var("map-const", "map-const",
               {1: pack([20200131, 123045], "<i4") + pack([0.2], "<f4") + pack([60.0], "<f4")})
    counters = [0, 300, 1_000_000_000]
    w.data_var("map-info-series", "map-info-series",
               {i + 1: pack([c], "<i4") for i, c in enumerate(counters)})
    w.write(tmp_path / "t.def", tmp_path / "t.dat")
    from d3d.nefis_dll import NefisFile
    with NefisFile(tmp_path / "t.def", tmp_path / "t.dat") as nf:
        assert d3d.reference_date(nf) == np.datetime64("2020-01-31T12:30:45")
        t = d3d.get_time(nf, "map-info-series")
        # 300 * 0.2 min = 60 min exactos; 1e9 * 12 s = 1.2e10 s exactos (sin error de float32)
        assert t[1] == np.datetime64("2020-01-31T13:30:45")
        assert t[2] == np.datetime64("2020-01-31T12:30:45") + np.timedelta64(12_000_000_000, "s")


# ---------------- campos ----------------

def test_water_level_igual_a_literal(trim_sed, ref_trim_sed):
    r = ref_trim_sed
    kcs = r.read("map-const", "KCS", 1)
    for i in (0, 144, 288):
        s1 = r.read("map-series", "S1", i + 1).astype(np.float32)
        kfu = r.read("map-series", "KFU", i + 1)
        kfv = r.read("map-series", "KFV", i + 1)
        esperado = s1.copy()
        esperado[(s1 == -999) | (s1 == 999)] = np.nan
        esperado[np.abs(kcs) != 1] = np.nan
        kfvs = np.vstack([kfv[:1], kfv[:-1]])                 # kfv(n-1)
        kfus = np.hstack([kfu[:, :1], kfu[:, :-1]])           # kfu(m-1)
        esperado[(kfu == 0) & (kfus == 0) & (kfv == 0) & (kfvs == 0)] = np.nan
        np.testing.assert_array_equal(d3d.water_level(trim_sed, index=i), esperado)


def test_water_level_varios_registros(trim_sed):
    todos = d3d.water_level(trim_sed)
    assert todos.shape == trim_sed["S1"].shape
    np.testing.assert_array_equal(d3d.water_level(trim_sed, index=[3, 100]), todos[[3, 100]])


def test_velocity_xy_seleccion_de_tiempo_y_capa(trim_sed):
    x, y = d3d.velocity_xy(trim_sed, index=[5, 6, 7])
    assert x.shape == (3,) + trim_sed["U1"].shape[1:]
    xi, yi = d3d.velocity_xy(trim_sed, index=6)
    np.testing.assert_array_equal(xi, x[1])
    xk, yk = d3d.velocity_xy(trim_sed, index=[5, 6, 7], layer=2)
    np.testing.assert_array_equal(xk, x[..., 2])
    np.testing.assert_array_equal(yk, y[..., 2])


def test_vector_xy_transporte_2d(trim_sed, ref_trim_sed):
    """SBUUA/SBVVA: -999 → NaN, uv2cen, cur2ca(ALFAS) y máscara KCS (literal)."""
    r = ref_trim_sed
    x, y = d3d.vector_xy(trim_sed, "SBUUA", "SBVVA")
    assert x.shape == trim_sed["SBUUA"].shape            # (T, N, M, 1)

    def media(a, b):
        n = (~np.isnan(a)).astype(float) + (~np.isnan(b))
        return (np.nan_to_num(a) + np.nan_to_num(b)) / np.maximum(n, 1)

    u = r.read("map-avg-series", "SBUUA", 1)[..., 0].astype(float)   # (N, M)
    v = r.read("map-avg-series", "SBVVA", 1)[..., 0].astype(float)
    u[u == -999] = np.nan
    v[v == -999] = np.nan
    U = np.full(u.shape, np.nan)
    V = np.full(v.shape, np.nan)
    U[:, 1:] = media(u[:, 1:], u[:, :-1])                # U media en M
    V[1:, :] = media(v[1:, :], v[:-1, :])                # V media en N
    alf = np.deg2rad(r.read("map-const", "ALFAS", 1).astype(float))
    X = U * np.cos(alf) - V * np.sin(alf)
    Y = U * np.sin(alf) + V * np.cos(alf)
    inact = np.abs(r.read("map-const", "KCS", 1)) != 1
    X[inact] = np.nan
    Y[inact] = np.nan
    np.testing.assert_allclose(x[0, ..., 0], X, rtol=0, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(y[0, ..., 0], Y, rtol=0, atol=1e-6, equal_nan=True)


def test_vector_xy_formas_distintas(trim_sed):
    with pytest.raises(ValueError):
        d3d.vector_xy(trim_sed, "U1", "SBVVA")
