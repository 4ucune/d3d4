"""Exportación netCDF (`d3d.to_netcdf`, `TrimFile.save_nc`).

El motor 'scipy' (netCDF3) se prueba siempre que haya scipy; el motor
'netcdf4' solo si el paquete netCDF4 está instalado.
"""
import numpy as np
import pytest

import d3d
from _nefis_writer import NefisWriter, pack

ENGINES = ["scipy", "netcdf4"]


def _abrir(path, engine):
    """→ (dimensiones {nombre: tamaño}, variables {nombre: (datos, dims, attrs)}, globales)."""
    if engine == "scipy":
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as f:
            dec = lambda v: v.decode() if isinstance(v, bytes) else v
            dims = dict(f.dimensions)
            vs = {n: (np.array(v[:]), v.dimensions, {k: dec(a) for k, a in v._attributes.items()})
                  for n, v in f.variables.items()}
            return dims, vs, {k: dec(a) for k, a in f._attributes.items()}
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as f:
        f.set_auto_maskandscale(False)
        dims = {k: len(d) for k, d in f.dimensions.items()}
        vs = {n: (np.array(v[:]), v.dimensions, {k: v.getncattr(k) for k in v.ncattrs()})
              for n, v in f.variables.items()}
        return dims, vs, {k: f.getncattr(k) for k in f.ncattrs()}


@pytest.fixture(params=ENGINES)
def engine(request):
    if request.param == "scipy":
        pytest.importorskip("scipy.io")
    else:
        pytest.importorskip("netCDF4")
    return request.param


def test_datos_crudos_y_estructura(trim_sed, engine, tmp_path):
    p = d3d.to_netcdf(trim_sed, tmp_path / "a.nc", index=slice(0, 10), engine=engine)
    dims, vs, g = _abrir(p, engine)
    assert dims["N"] == 3 and dims["M"] == 47 and dims["K"] == 10 and dims["time"] == 10
    for name in ("S1", "U1", "V1"):
        np.testing.assert_array_equal(vs[name][0], trim_sed[name].raw[0:10])
    np.testing.assert_array_equal(vs["KCS"][0], trim_sed["KCS"][0])
    assert vs["U1"][1] == ("time", "N", "M", "K")
    assert vs["S1"][2]["units"] == "m" and vs["U1"][2]["units"] == "m s-1"
    assert vs["U1"][2]["delft3d_location"] == "U" and "coordinates" not in vs["U1"][2]
    assert vs["S1"][2]["coordinates"] == "XZ YZ"
    assert vs["XZ"][2]["standard_name"] == "projection_x_coordinate"
    assert float(vs["S1"][2]["missing_value"]) == -999.0
    assert g["Conventions"] == "CF-1.8"
    assert g["delft3d_ITDATE"].startswith("20010601")


def test_tiempo_y_sigma(trim_sed, engine, tmp_path):
    p = d3d.to_netcdf(trim_sed, tmp_path / "t.nc", index=[0, 100, 288], engine=engine)
    _, vs, _ = _abrir(p, engine)
    np.testing.assert_allclose(vs["time"][0], d3d.get_time(trim_sed, as_seconds=True)[[0, 100, 288]])
    assert vs["time"][2]["units"] == "seconds since 2001-06-01 00:00:00"
    k, _, ka = vs["K"]
    np.testing.assert_allclose(k, -(np.arange(10) + 0.5) / 10, atol=1e-12)   # THICK = 0.1 x 10
    assert ka["standard_name"] == "ocean_sigma_coordinate"
    assert ka["formula_terms"] == "sigma: K eta: S1 depth: DPS0"


def test_sin_terminos_sigma_no_declara_standard_name(trim_sed, tmp_path):
    pytest.importorskip("scipy.io")
    p = d3d.to_netcdf(trim_sed, tmp_path / "s.nc", variables=["U1"], index=0, engine="scipy")
    _, vs, _ = _abrir(p, "scipy")
    assert "standard_name" not in vs["K"][2]
    assert "XZ" not in vs                         # U1 no está en puntos Z: sin coordenadas Z


def test_derivados(trim_sed, engine, tmp_path):
    p = d3d.to_netcdf(trim_sed, tmp_path / "d.nc", variables=["S1"], index=slice(5, 8),
                      derived=["velocity_xy", "water_level"], engine=engine)
    _, vs, _ = _abrir(p, engine)
    x, y = d3d.velocity_xy(trim_sed, index=slice(5, 8))
    np.testing.assert_array_equal(vs["ux"][0], x)
    np.testing.assert_array_equal(vs["uy"][0], y)
    np.testing.assert_array_equal(vs["water_level"][0], d3d.water_level(trim_sed, index=slice(5, 8)))
    assert vs["ux"][2]["standard_name"] == "sea_water_x_velocity"


def test_texto(trim_sed, engine, tmp_path):
    p = d3d.to_netcdf(trim_sed, tmp_path / "c.nc", variables=["NAMCON"], engine=engine)
    _, vs, _ = _abrir(p, engine)
    datos = vs["NAMCON"][0]
    esperado = trim_sed["NAMCON"][0]
    assert b"".join(datos[0].tolist()).strip() == esperado[0].strip()


def test_save_nc_compatibilidad_y_renombrado(trim_sed, tmp_path):
    if _sin_netcdf4():
        pytest.importorskip("scipy.io")
    trim_sed.save_nc(str(tmp_path / "b.nc"),
                     variables={"nivel": "S1", "x": "XZ", "prof": "DPS0", "nada": "NO_EXISTE"})
    _, vs, _ = _abrir(tmp_path / "b.nc", "scipy" if _sin_netcdf4() else "netcdf4")
    assert {"nivel", "x", "prof", "YZ"} <= set(vs) and "XZ" not in vs
    assert vs["nivel"][2]["coordinates"] == "x YZ"


def _sin_netcdf4():
    try:
        import netCDF4  # noqa: F401
        return False
    except ImportError:
        return True


def test_limite_de_memoria_scipy(trim_sed, tmp_path):
    pytest.importorskip("scipy.io")
    with pytest.raises(MemoryError):
        d3d.to_netcdf(trim_sed, tmp_path / "m.nc", engine="scipy", max_memory_bytes=1000)


def test_argumentos_invalidos(trim_sed, tmp_path):
    with pytest.raises(ValueError):
        d3d.to_netcdf(trim_sed, tmp_path / "x.nc", engine="otro")
    with pytest.raises(ValueError):
        d3d.to_netcdf(trim_sed, tmp_path / "x.nc", derived=["vorticidad"], engine="scipy")
    with pytest.raises(KeyError):
        d3d.to_netcdf(trim_sed, tmp_path / "x.nc", variables=["NO_EXISTE"], engine="scipy")


def test_sin_motor_netcdf(trim_sed, tmp_path, monkeypatch):
    """Sin netCDF4 ni scipy: error claro (no un fallo a mitad de escritura)."""
    import builtins
    real_import = builtins.__import__

    def falso(name, *a, **k):
        if name.split(".")[0] in ("netCDF4", "scipy"):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", falso)
    with pytest.raises(ImportError, match="netCDF4 o scipy"):
        d3d.to_netcdf(trim_sed, tmp_path / "x.nc", variables=["S1"], index=0)
    assert not (tmp_path / "x.nc").exists()


def test_ubicaciones():
    from d3d.export import variable_location
    assert variable_location("U1") == "U" and variable_location("V1") == "V"
    assert variable_location("S1") == "Z" and variable_location("XCOR") == "corner"
    assert variable_location("NUEVA", "Something in U-point") == "U"
    assert variable_location("NUEVA", "Concentration in zeta point") == "Z"
    assert variable_location("NUEVA", "otra cosa") == "unknown"


# ---------------------------------------------------------------------------
# Malla cuadrada (N == M): antes save_nc nombraba dims por tamaño y las mezclaba
# ---------------------------------------------------------------------------

@pytest.fixture
def trim_cuadrado(tmp_path):
    n = m = 4
    k = 2
    w = NefisWriter()
    for nm, typ, sv, dims in [("ITDATE", "INTEGER", 4, [2]), ("DT", "REAL", 4, [1]),
                              ("TUNIT", "REAL", 4, [1]), ("NMAX", "INTEGER", 4, [1]),
                              ("MMAX", "INTEGER", 4, [1]), ("KMAX", "INTEGER", 4, [1]),
                              ("THICK", "REAL", 4, [k]), ("LAYER_MODEL", "CHARACTE", 16, [1]),
                              ("KCS", "INTEGER", 4, [n, m]), ("ALFAS", "REAL", 4, [n, m]),
                              ("XZ", "REAL", 8, [n, m]), ("YZ", "REAL", 8, [n, m]),
                              ("ITMAPC", "INTEGER", 4, [1]),
                              ("S1", "REAL", 4, [n, m]), ("U1", "REAL", 4, [n, m, k])]:
        w.element(nm, typ, sv, dims, description="water level in zeta point" if nm == "S1" else "")
    w.cell("map-const", ["ITDATE", "DT", "TUNIT", "NMAX", "MMAX", "KMAX", "THICK",
                         "LAYER_MODEL", "KCS", "ALFAS", "XZ", "YZ"])
    w.cell("map-info-series", ["ITMAPC"])
    w.cell("map-series", ["S1", "U1"])
    for g in ("map-const", "map-info-series", "map-series"):
        w.group(g, g, [0])
    nn, mm = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    w.data_var("map-const", "map-const", {1: b"".join([
        pack([20200101, 0], "<i4"), pack([1.0], "<f4"), pack([60.0], "<f4"),
        pack([n], "<i4"), pack([m], "<i4"), pack([k], "<i4"), pack([0.5, 0.5], "<f4"),
        b"SIGMA-MODEL     ", pack(np.ones((n, m)), "<i4"), pack(np.zeros((n, m)), "<f4"),
        pack(mm * 10.0, "<f8"), pack(nn * 10.0, "<f8")])})
    w.data_var("map-info-series", "map-info-series", {t: pack([t], "<i4") for t in (1, 2)})
    s1 = {t: (100 * nn + mm + t).astype("<f4") for t in (1, 2)}      # S1[n, m] = 100n + m + t
    w.data_var("map-series", "map-series",
               {t: pack(s1[t], "<f4") + pack(np.zeros((n, m, k)), "<f4") for t in (1, 2)})
    w.write(tmp_path / "trim-q.def", tmp_path / "trim-q.dat")
    data = d3d.read(str(tmp_path / "trim-q.dat"))
    yield data, s1
    data.close()


def test_malla_cuadrada_no_mezcla_ejes(trim_cuadrado, tmp_path):
    pytest.importorskip("scipy.io")
    data, s1 = trim_cuadrado
    p = d3d.to_netcdf(data, tmp_path / "q.nc", variables=["S1", "U1"], engine="scipy")
    dims, vs, _ = _abrir(p, "scipy")
    assert vs["S1"][1] == ("time", "N", "M")
    np.testing.assert_array_equal(vs["S1"][0][1], s1[2])      # S1[n, m] = 100 n + m + 2
    assert vs["S1"][0][1][2, 1] == 100 * 2 + 1 + 2
    np.testing.assert_allclose(vs["time"][0], [60.0, 120.0])  # ITMAPC·DT·TUNIT
    assert dims["K"] == 2
