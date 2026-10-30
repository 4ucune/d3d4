"""Máscara por variable al leer y variables derivadas por nombre (trim-sed).

Las referencias son traducciones literales de d3d_trimfil.m sobre los datos
crudos del lector de referencia (no usan d3d.fields).
"""
import numpy as np
import pytest

import d3d
from d3d import transforms as tr


def _act(ref):
    return np.abs(ref.read("map-const", "KCS", 1)) == 1


def _conv2_2x2(act):
    """conv2(double(act), [1 1; 1 1], 'valid') > 0 sobre act extendida (idx1)."""
    n, m = act.shape
    a = np.vstack([act, act[-1:]])
    a = np.hstack([a, a[:, -1:]])
    out = np.zeros((n, m), dtype=bool)
    for i in range(n):
        for j in range(m):
            out[i, j] = a[i:i + 2, j:j + 2].sum() > 0
    return out


def _nan(a, mask2d, sentinels=(-999.0,)):
    a = np.array(a, dtype=float)
    for s in sentinels:
        a[a == s] = np.nan
    full = np.broadcast_to(mask2d.reshape(mask2d.shape + (1,) * (a.ndim - 2)), a.shape)
    a[~full] = np.nan
    return a


# ---------- posición ----------

@pytest.mark.parametrize("name,loc", [
    ("S1", "z"), ("R1", "z"), ("DPS0", "z"), ("XZ", "z"), ("WPHY", "z"),
    ("U1", "u"), ("V1", "v"), ("SBUU", "u"), ("SBVV", "v"), ("TAUKSI", "u"),
    ("XCOR", "d"), ("YCOR", "d"), ("DP0", "d"),
    ("KCS", None), ("KFU", None), ("ITMAPC", None), ("NAMCON", None), ("THICK", None),
])
def test_posicion(trim_sed, name, loc):
    assert trim_sed[name].location == loc


def test_trih_sin_mascara(data_dir):
    with d3d.read(str(data_dir / "trih-sed.dat")) as h:
        assert h.grid_shape is None
        assert all(v.location is None for v in h.variables.values())


def test_corner_mask_igual_a_conv2(ref_trim_sed):
    act = _act(ref_trim_sed)
    np.testing.assert_array_equal(tr.corner_mask(act), _conv2_2x2(act))
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random((7, 9)) > 0.6
        np.testing.assert_array_equal(tr.corner_mask(a), _conv2_2x2(a))


# ---------- máscara por defecto ----------

def test_s1_es_water_level(trim_sed):
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "S1"), d3d.water_level(trim_sed))
    np.testing.assert_array_equal(trim_sed["S1"][7], d3d.water_level(trim_sed, index=7))


def test_s1_literal(trim_sed, ref_trim_sed):
    """'water level': ±999 → NaN, |KCS| != 1 → NaN, celdas secas → NaN."""
    r = ref_trim_sed
    act = _act(r)
    for t in (0, 144, 288):
        s1 = _nan(r.read("map-series", "S1", t + 1), act, (-999.0, 999.0))
        kfu = r.read("map-series", "KFU", t + 1)
        kfv = r.read("map-series", "KFV", t + 1)
        seca = ((kfv[[0] + list(range(kfv.shape[0] - 1)), :] == 0) & (kfv == 0)
                & (kfu[:, [0] + list(range(kfu.shape[1] - 1))] == 0) & (kfu == 0))
        s1[seca] = np.nan
        np.testing.assert_array_equal(trim_sed["S1"][t], s1)


@pytest.mark.parametrize("name,group,esquina", [
    ("U1", "map-series", True), ("V1", "map-series", True),
    ("SBUU", "map-sed-series", True), ("R1", "map-series", False),
    ("WPHY", "map-series", False), ("DPS0", "map-const", False),
    ("XCOR", "map-const", True), ("DP0", "map-const", True), ("XZ", "map-const", False),
])
def test_mascara_literal(trim_sed, ref_trim_sed, name, group, esquina):
    r = ref_trim_sed
    act = _act(r)
    mask = _conv2_2x2(act) if esquina else act
    nt = r.grpdat[group]["size"][0]
    for t in sorted({0, nt // 2, nt - 1}):
        esperado = _nan(r.read(group, name, t + 1), mask)
        np.testing.assert_array_equal(trim_sed[name][t], esperado)
        np.testing.assert_array_equal(trim_sed[name].raw[t], r.read(group, name, t + 1))


@pytest.mark.parametrize("name", ["KCS", "KFU", "ITMAPC", "NAMCON"])
def test_sin_mascara_igual_a_crudo(trim_sed, name):
    v = trim_sed[name]
    a, b = v[:], v.raw[:]
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["S1", "U1", "XCOR", "R1"])
@pytest.mark.parametrize("sel", [
    (5,), (slice(10, 20),), ([0, 7, -1],), (slice(None), 1, 20),
    (slice(None), 0, 0), (3, slice(None), slice(5, 9)), (Ellipsis, 2), (-1, 1),
    (slice(None), np.array([0, 2]), np.array([3, 40])),
])
def test_seleccion_enmascarada_igual_a_completa(trim_sed, name, sel):
    """v[sel] == v[:][sel] también con máscara (incluida la seca de S1)."""
    v = trim_sed[name]
    completo = v[:]
    key = sel if len(sel) > 1 else sel[0]
    try:
        esperado = completo[sel]
    except IndexError:                      # selector inválido para esta variable
        with pytest.raises(IndexError):
            v[key]
        return
    np.testing.assert_array_equal(v[key], esperado)


def test_mask_false(trim_sed):
    a = d3d.get_var(trim_sed, "U1", index=3, mask=False)
    np.testing.assert_array_equal(a, trim_sed["U1"].raw[3])
    np.testing.assert_array_equal(trim_sed.read("U1", 3, mask=False), a)
    assert np.isnan(d3d.get_var(trim_sed, "U1", index=3)).any()
    assert not np.isnan(a).any()


# ---------- derivadas por nombre ----------

def test_derivadas_centro(trim_sed):
    for name, u, v in [("bed load transport", "SBUU", "SBVV"),
                       ("mean d.a. suspended transport", "SSUUA", "SSVVA"),
                       ("horizontal velocity", "U1", "V1")]:
        x, y, mag = d3d.get_var(trim_sed, name, index=[0, -1])
        ex, ey = d3d.vector_xy(trim_sed, u, v, index=[0, -1])
        if trim_sed[u].group != "map-series":
            ex, ey = ex[..., 0], ey[..., 0]          # una sola fracción
        np.testing.assert_array_equal(x, ex)
        np.testing.assert_array_equal(y, ey)
        np.testing.assert_array_equal(mag, np.hypot(ex, ey))


def test_total_transport_suma_en_caras(trim_sed, ref_trim_sed):
    r = ref_trim_sed
    t = 200
    u = sum(tr.invalid_to_nan(r.read("map-sed-series", n, t + 1).astype(float)) for n in ("SSUU", "SBUU"))
    v = sum(tr.invalid_to_nan(r.read("map-sed-series", n, t + 1).astype(float)) for n in ("SSVV", "SBVV"))
    uc, vc = tr.uv2cen(u, v, n_axis=0, m_axis=1)
    x, y = tr.cur2ca(uc, vc, r.read("map-const", "ALFAS", 1), n_axis=0, m_axis=1)
    act = _act(r)
    x, y = _nan(x, act)[..., 0], _nan(y, act)[..., 0]
    X, Y, _ = d3d.get_var(trim_sed, "total transport", index=t)
    np.testing.assert_allclose(X, x, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Y, y, rtol=0, atol=1e-6)


def test_staggered_es_crudo_enmascarado(trim_sed, ref_trim_sed):
    r = ref_trim_sed
    mask = _conv2_2x2(_act(r))
    x, y, mag = d3d.get_var(trim_sed, "staggered bedload transport", index=50)
    np.testing.assert_array_equal(x, _nan(r.read("map-sed-series", "SBUU", 51), mask)[..., 0])
    np.testing.assert_array_equal(y, _nan(r.read("map-sed-series", "SBVV", 51), mask)[..., 0])
    np.testing.assert_array_equal(mag, np.hypot(x, y))
    assert x.shape == trim_sed["S1"].shape[1:]


def test_derivada_componentes(trim_sed):
    x, y, mag = d3d.get_var(trim_sed, "bed load transport")
    assert x.shape == trim_sed["S1"].shape
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "bed load transport", component="X"), x)
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "bed load transport", component="Y"), y)
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "bed load transport", component="mag"), mag)
    with pytest.raises(ValueError):
        d3d.get_var(trim_sed, "bed load transport", component="Z")
    with pytest.raises(TypeError):
        d3d.get_var(trim_sed, "S1", component="X")
    with pytest.raises(KeyError):
        d3d.get_var(trim_sed, "no existe transport")


# ---------- escalares de QUICKPLOT por nombre ----------

def test_escalares_por_nombre(trim_sed):
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "water level", index=7), trim_sed["S1"][7])
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "vertical velocity", index=3),
                                  trim_sed["WPHY"][3])
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "water level", index=7, mask=False),
                                  trim_sed["S1"].raw[7])
    np.testing.assert_array_equal(d3d.get_var(trim_sed, "water level", index=(slice(None), 1, 20)),
                                  trim_sed["S1"][:, 1, 20])


def test_constituyentes_por_nombre(trim_sed):
    nombres = [s.decode().strip() for s in np.ravel(trim_sed["NAMCON"].raw[0])]
    lstci = int(np.ravel(trim_sed["LSTCI"].raw[0])[0])
    for i, nm in enumerate(nombres[:lstci]):
        np.testing.assert_array_equal(d3d.get_var(trim_sed, nm, index=10), trim_sed["R1"][10][..., i])
        np.testing.assert_array_equal(d3d.get_var(trim_sed, nm.upper(), index=10),
                                      trim_sed["R1"][10][..., i])
    assert set(nombres[:lstci]) <= set(d3d.fields.scalars(trim_sed))


def test_info_catalogo_completo(trim_sed, capsys):
    d3d.info(trim_sed)
    out = capsys.readouterr().out
    for n in trim_sed.keys():
        assert f"  {n} " in out, n
    for seccion in ("Variables NEFIS", "Campos escalares", "Campos vectoriales", "Funciones de d3d"):
        assert seccion in out
    for n in ("water level", "bed load transport", "d3d.water_level", "d3d.velocity_xy"):
        assert n in out
    for n in ("S1", "water level", "bed load transport", "nada"):
        trim_sed.info(n)
    out = capsys.readouterr().out
    assert "variable NEFIS" in out and "campo escalar" in out and "campo vectorial" in out
    assert "no encontrada" in out
