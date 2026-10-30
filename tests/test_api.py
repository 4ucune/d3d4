"""Interfaz pública (`d3d.read`, `d3d.get_var`, ...) sobre trim-sed."""
import numpy as np
import pytest

import d3d


# ---------- apertura ----------

def test_read_acepta_dat_y_def(data_dir):
    for ext in (".dat", ".def"):
        with d3d.read(str(data_dir / f"trim-sed{ext}")) as data:
            assert "S1" in data


def test_read_extension_invalida(tmp_path):
    with pytest.raises(ValueError):
        d3d.read(str(tmp_path / "trim-x.nc"))


def test_read_falta_def(tmp_path):
    (tmp_path / "trim-x.dat").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        d3d.read(str(tmp_path / "trim-x.dat"))


def test_read_falta_dat(tmp_path):
    (tmp_path / "trim-x.def").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        d3d.read(str(tmp_path / "trim-x.def"))


def test_variable_inexistente(trim_sed):
    with pytest.raises(KeyError):
        d3d.get_var(trim_sed, "NO_EXISTE")


# ---------- lectura cruda ----------

@pytest.mark.parametrize("name,group", [
    ("S1", "map-series"), ("U1", "map-series"), ("KCS", "map-const"),
    ("ALFAS", "map-const"), ("DP0", "map-const"), ("ITMAPC", "map-info-series"),
])
def test_get_var_crudo_igual_a_referencia(trim_sed, ref_trim_sed, name, group):
    """mask=False / .raw: sin transformación, (T, *dims NEFIS)."""
    nt = ref_trim_sed.grpdat[group]["size"][0]
    esperado = np.stack([ref_trim_sed.read(group, name, i) for i in range(1, nt + 1)])
    np.testing.assert_array_equal(d3d.get_var(trim_sed, name, mask=False), esperado)
    np.testing.assert_array_equal(d3d.get_var(trim_sed, name, index=nt - 1, mask=False), esperado[-1])
    np.testing.assert_array_equal(trim_sed[name].raw[-1], esperado[-1])


def test_variables_de_texto(trim_sed):
    assert d3d.get_var(trim_sed, "LAYER_MODEL", index=0)[0].strip() == b"SIGMA-MODEL"
    assert d3d.get_var(trim_sed, "COORDINATES", index=0)[0].strip() == b"CARTESIAN"


def test_accesos_repetidos(trim_sed):
    a = d3d.get_var(trim_sed, "S1", index=5)
    b = d3d.get_var(trim_sed, "S1", index=5)
    np.testing.assert_array_equal(a, b)


def test_unidades(trim_sed):
    assert trim_sed["S1"].attributes["units"].strip("[] ") == "M"
    assert trim_sed["U1"].attributes["units"].strip("[] ") == "M/S"


# ---------- velocidad en centros (velocity_xy) ----------

def _uv2cen(u, v):
    """Traducción literal de uv2cen.m para arreglos (T, N, M, K)."""
    U = np.full(u.shape, np.nan)
    V = np.full(v.shape, np.nan)
    a, b = v[:, 1:], v[:, :-1]
    n = (~np.isnan(a)).astype(float) + (~np.isnan(b))
    V[:, 1:] = (np.nan_to_num(a) + np.nan_to_num(b)) / np.maximum(n, 1)
    a, b = u[:, :, 1:], u[:, :, :-1]
    n = (~np.isnan(a)).astype(float) + (~np.isnan(b))
    U[:, :, 1:] = (np.nan_to_num(a) + np.nan_to_num(b)) / np.maximum(n, 1)
    return U, V


def test_velocity_xy_igual_a_uv2cen_cur2ca(trim_sed, ref_trim_sed):
    """velocity_xy = uv2cen + cur2ca(ALFAS) + máscara KCS (d3d_trimfil.m).

    LIMITACIÓN: en este caso ALFAS = 0 y solo hay una fila activa; la rotación
    y el promedio en N no quedan realmente ejercitados.
    """
    r = ref_trim_sed
    T = [0, 144, 288]
    u = np.stack([r.read("map-series", "U1", t + 1) for t in T]).astype(float)
    v = np.stack([r.read("map-series", "V1", t + 1) for t in T]).astype(float)
    u[u == -999] = np.nan
    v[v == -999] = np.nan
    U, V = _uv2cen(u, v)
    alf = np.deg2rad(r.read("map-const", "ALFAS", 1).astype(float))[None, :, :, None]
    X = U * np.cos(alf) - V * np.sin(alf)
    Y = U * np.sin(alf) + V * np.cos(alf)
    inact = np.abs(r.read("map-const", "KCS", 1)) != 1
    X[:, inact] = np.nan
    Y[:, inact] = np.nan
    nx, ny = d3d.velocity_xy(trim_sed, index=T)
    np.testing.assert_allclose(nx, X, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ny, Y, rtol=0, atol=1e-6)


# ---------- errores corregidos en la Fase 3 ----------

def test_tiempo_usa_tunit(trim_sed, ref_trim_sed):
    r = ref_trim_sed
    nt = r.grpdat["map-info-series"]["size"][0]
    it = np.array([r.read("map-info-series", "ITMAPC", i)[0] for i in range(1, nt + 1)])
    seg = it * r.read("map-const", "DT", 1)[0] * r.read("map-const", "TUNIT", 1)[0]
    np.testing.assert_allclose(d3d.extract_all(trim_sed, ["S1"])["time"], seg)


@pytest.mark.parametrize("name", ["S1", "U1", "KCS", "XCOR", "THICK", "ITDATE", "SBUUA", "NAMCON"])
def test_shape_coincide_con_dato(trim_sed, name):
    v = trim_sed[name]
    assert v.shape == v.full_shape == d3d.get_var(trim_sed, name).shape


def test_read_con_dat_path(tmp_path, data_dir):
    import shutil
    shutil.copy(data_dir / "trih-mud.def", tmp_path / "x.def")
    shutil.copy(data_dir / "trih-mud.dat", tmp_path / "x.dat.exe")
    with d3d.read(str(tmp_path / "x.def"), dat_path=str(tmp_path / "x.dat.exe")) as data:
        assert "ITHISC" in data and data["ITHISC"].shape == (3, 1)
    with pytest.raises(ValueError):
        d3d.read(str(tmp_path / "x.dat.exe"), dat_path=str(tmp_path / "x.dat.exe"))


def test_get_var_index_escribible_y_no_compartido(trim_sed):
    """Antes: index=i devolvía un arreglo de solo lectura compartido por la caché."""
    a = d3d.get_var(trim_sed, "S1", index=0)
    assert a.flags.writeable
    a[...] = -1
    assert (d3d.get_var(trim_sed, "S1", index=0) != -1).any()


# ---------- lectura selectiva ----------

@pytest.mark.parametrize("sel", [
    (5,), (slice(10, 20),), (slice(None, None, 50),), ([0, 7, -1],),
    (slice(None), 1, 20), (slice(None), 1, 20, 3), (100, slice(None), slice(5, 9), 0),
    (Ellipsis, 2), (-1, 1), (np.arange(289) % 7 == 0,),
])
def test_seleccion_igual_a_lectura_completa(trim_sed, sel):
    """v[sel] == v[:][sel] (U1: (T, N, M, K))."""
    completo = d3d.get_var(trim_sed, "U1")
    obtenido = d3d.get_var(trim_sed, "U1", index=sel if len(sel) > 1 else sel[0])
    np.testing.assert_array_equal(obtenido, completo[sel])
    assert obtenido.flags.writeable


def test_serie_temporal_puntual(trim_sed, ref_trim_sed):
    serie = trim_sed["S1"].raw[:, 1, 20]
    esperado = [ref_trim_sed.read("map-series", "S1", t)[1, 20] for t in range(1, 290)]
    np.testing.assert_array_equal(serie, esperado)


@pytest.mark.parametrize("sel", [289, -290, [0, 400], 1.5, "a"])
def test_seleccion_invalida(trim_sed, sel):
    with pytest.raises((IndexError, TypeError)):
        trim_sed["S1"][sel]


def test_corrected_eliminado(trim_sed):
    with pytest.raises(TypeError, match="water_level"):
        d3d.get_var(trim_sed, "S1", corrected=True)


def test_cerrar_con_vistas_vivas(data_dir):
    """close() no falla aunque existan vistas sin copia del archivo."""
    data = d3d.read(str(data_dir / "trim-mud.dat"))
    v = data._reader.view_element("map-series", "S1", 0)
    copia = np.array(v)
    data.close()
    del v
    assert copia.size > 0
