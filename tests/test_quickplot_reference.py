"""Validación contra exportaciones de QUICKPLOT ("Save data" → .mat).

Cada .mat de la carpeta es una prueba. El campo se identifica por `data.Name`
y el registro por `data.Time` (datenum), así que se pueden añadir instantes y
campos sin tocar este archivo. Opcional: se ejecuta solo si existen

    D3D_QUICKPLOT_DIR  carpeta con los .mat
    D3D_QUICKPLOT_DEF  ruta del .def del trim (por defecto D3D_TUMACO_DEF)
    D3D_QUICKPLOT_DAT  ruta del .dat del trim (por defecto D3D_TUMACO_DAT)

Sin variables de entorno: si existe `tests/data/quickplot`, sus .mat se
comparan con `trim-sed` de los datos de prueba (exportaciones pequeñas que
se pueden distribuir); si no, se usa `Resultados/matlab` con Tumaco.

Convenciones de QUICKPLOT (d3d_trimfil.m), verificadas con Tumaco:
- ejes (M, N[, K]); d3d usa (N, M[, K]).
- campos 2D en celdas (nivel, transportes): X/Y = esquinas XCOR/YCOR (M, N) y
  Val (M-1, N-1), con Val[i, j] = celda (m, n) = (i+1, j+1).
- campos 3D: X/Y = centros XZ/YZ y Val (M, N, K); NaN donde |KCS| != 1.
- 'total transport' = (SSUU + SBUU, SSVV + SBVV) antes de llevarlo a centros.

Lecturas directas (nivel, WPHY, concentraciones) se exigen idénticas; las que
pasan por uv2cen/cur2ca, con tolerancia por el redondeo float32 de d3d.

`test_comparador_*` verifican el comparador con un .mat FABRICADO a partir de
d3d con 04_fluidmud (no es una validación contra QUICKPLOT).
"""
import os
from pathlib import Path

import numpy as np
import pytest

import d3d
from d3d import transforms as tr

from conftest import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
_QP_LOCAL = Path(__file__).resolve().parent / "data" / "quickplot"
QP_DIR = Path(os.environ.get("D3D_QUICKPLOT_DIR")
              or (_QP_LOCAL if _QP_LOCAL.is_dir() else ROOT / "Resultados" / "matlab"))
_LOCAL = QP_DIR == _QP_LOCAL and not os.environ.get("D3D_QUICKPLOT_DEF")
DEF = (str(DATA_DIR / "trim-sed.def") if _LOCAL else
       os.environ.get("D3D_QUICKPLOT_DEF") or os.environ.get("D3D_TUMACO_DEF"))
DAT = (str(DATA_DIR / "trim-sed.dat") if _LOCAL else
       os.environ.get("D3D_QUICKPLOT_DAT") or os.environ.get("D3D_TUMACO_DAT"))
MATLAB_EPOCH = 719529          # datenum de 1970-01-01
ATOL_DERIVADO = 1e-6

MATS = sorted(QP_DIR.glob("*.mat")) if QP_DIR.is_dir() else []
HAY_TRIM = bool(DEF and DAT and Path(DEF).exists() and Path(DAT).exists())


# ---------------- comparador ----------------

def cargar(path):
    """→ dict con los campos de la estructura `data` de QUICKPLOT."""
    from scipy.io import loadmat
    s = loadmat(str(path))["data"][0, 0]
    out = {k: s[k] for k in s.dtype.names}
    out["Name"] = str(np.asarray(out["Name"]).ravel()[0])
    return out


# Transportes: nombre de QUICKPLOT → pares (U, V) que se suman en las caras.
TRANSPORTES = {
    "bed load transport": [("SBUU", "SBVV")],
    "total transport": [("SSUU", "SSVV"), ("SBUU", "SBVV")],
    "mean bed load transport": [("SBUUA", "SBVVA")],
    "mean d.a. suspended transport": [("SSUUA", "SSVVA")],
    "mean total transport": [("SSUUA", "SSVVA"), ("SBUUA", "SBVVA")],
}


def grupo(ref):
    """Grupo NEFIS del campo: los 'mean ...' están en 'map-avg-series' (tiempo ITAVGS)."""
    return "map-avg-series" if ref["Name"].startswith("mean ") else "map-series"


def registro(data, ref):
    """Índice 0-based del registro del grupo del campo con el tiempo de `ref`."""
    g = grupo(ref)
    ms = int(round((float(np.asarray(ref["Time"]).ravel()[0]) - MATLAB_EPOCH) * 86400000))
    t = d3d.get_time(data, g).astype("int64")
    hit = np.flatnonzero(t == ms)
    assert hit.size == 1, f"tiempo {np.datetime64(ms, 'ms')} no está en {g!r}"
    return int(hit[0])


def _mn(a):
    """(N, M, ...) → (M, N, ...) como QUICKPLOT."""
    return np.swapaxes(np.asarray(a, dtype=float), 0, 1)


def _celdas(campo_mn):
    """Campo (M, N) → (M-1, N-1) como Val de QUICKPLOT en celdas."""
    return campo_mn[1:, 1:]


def _mascara_kcs(data, a):
    """NaN donde |KCS| != 1; `a` en (M, N[, K])."""
    act = tr.active_mask(data["KCS"][0]).T
    a = a.copy()
    a[~np.broadcast_to(act.reshape(act.shape + (1,) * (a.ndim - 2)), a.shape)] = np.nan
    return a


def _vector_suma(data, i, pares):
    """Suma de pares de componentes U/V crudos → (x, y) en centros (N, M, ...).

    Como d3d_trimfil.m: suma en las caras, después uv2cen, cur2ca y máscara KCS.
    """
    u = sum(tr.invalid_to_nan(data[a].raw[i]) for a, _ in pares)
    v = sum(tr.invalid_to_nan(data[b].raw[i]) for _, b in pares)
    uc, vc = tr.uv2cen(u, v, n_axis=0, m_axis=1)
    x, y = tr.cur2ca(uc, vc, data["ALFAS"].raw[0], n_axis=0, m_axis=1)
    act = tr.active_mask(data["KCS"][0])
    return (tr.apply_mask(x, act, n_axis=0, m_axis=1),
            tr.apply_mask(y, act, n_axis=0, m_axis=1))


def _transporte(data, i, pares):
    if data[pares[0][0]].shape[-1] != 1:
        pytest.skip("transporte con varias fracciones: falta verificar cómo las combina QUICKPLOT")
    x, y = _vector_suma(data, i, pares)
    return _celdas(_mn(np.hypot(x, y)[..., 0]))


def _constituyente(data, nombre):
    """Índice de `nombre` en NAMCON (sin distinguir mayúsculas) o None."""
    if "NAMCON" not in data:
        return None
    nam = [s.decode("ascii", "replace").strip().lower() for s in np.ravel(data["NAMCON"][0])]
    return nam.index(nombre.lower()) if nombre.lower() in nam else None


def esperado(data, i, ref):
    """→ lista de (etiqueta, d3d, referencia, exacto) para el campo de `ref`."""
    base, _, sufijo = ref["Name"].partition(", ")
    magnitud = sufijo == "magnitude"
    if base == "water level":
        return [("nivel", _celdas(_mn(d3d.water_level(data, index=i))), ref["Val"], True),
                ("nivel get_var", _celdas(_mn(d3d.get_var(data, "S1", index=i))), ref["Val"], True),
                ("nivel por nombre", _celdas(_mn(d3d.get_var(data, base, index=i))), ref["Val"], True)]
    if base == "vertical velocity":
        return [("w", _mascara_kcs(data, _mn(data["WPHY"].raw[i])), ref["Val"], True),
                ("w get_var", _mn(d3d.get_var(data, "WPHY", index=i)), ref["Val"], True)]
    if base in ("velocity", "horizontal velocity"):
        x, y = (_mn(c) for c in d3d.velocity_xy(data, index=i))
        if base == "horizontal velocity":
            mag = d3d.get_var(data, "horizontal velocity", index=i, component="mag")
            return [("|u|", np.hypot(x, y), ref["Val"], False),
                    ("|u| derivada", _mn(mag), ref["Val"], False)]
        w = _mascara_kcs(data, _mn(data["WPHY"].raw[i]))
        if magnitud:
            return [("|u,w|", np.sqrt(x ** 2 + y ** 2 + w ** 2), ref["Val"], False)]
        return [("ux", x, ref["XComp"], False), ("uy", y, ref["YComp"], False),
                ("w", w, ref["ZComp"], True)]
    if base in TRANSPORTES:
        pares = TRANSPORTES[base]
        mag = d3d.get_var(data, base, index=i, component="mag")
        return [("|s|", _transporte(data, i, pares), ref["Val"], False),
                ("|s| derivada", _celdas(_mn(mag)), ref["Val"], False)]
    c = _constituyente(data, base)
    if c is not None:
        r1 = tr.invalid_to_nan(data["R1"].raw[i][..., c])
        return [(base, _mascara_kcs(data, _mn(r1)), ref["Val"], True),
                (base + " get_var", _mn(d3d.get_var(data, "R1", index=i)[..., c]), ref["Val"], True),
                (base + " por nombre", _mn(d3d.get_var(data, base, index=i)), ref["Val"], True)]
    pytest.skip(f"campo de QUICKPLOT no soportado por esta prueba: {ref['Name']!r}")


def comparar_coordenadas(data, ref):
    val = ref["Val"] if "Val" in ref else ref["XComp"]
    if val.shape == tuple(s - 1 for s in ref["X"].shape):         # celdas: esquinas
        x, y = _mn(data["XCOR"][0]), _mn(data["YCOR"][0])
        X, Y = ref["X"], ref["Y"]
    else:                                                          # centros
        x, y = _mn(data["XZ"][0]), _mn(data["YZ"][0])
        X, Y = ref["X"][..., 0] if ref["X"].ndim == 3 else ref["X"], \
            ref["Y"][..., 0] if ref["Y"].ndim == 3 else ref["Y"]
    # La máscara por defecto de XCOR/XZ debe dejar NaN exactamente donde QUICKPLOT.
    np.testing.assert_array_equal(np.isnan(x), np.isnan(X), err_msg="máscara de X distinta")
    np.testing.assert_array_equal(np.isnan(y), np.isnan(Y), err_msg="máscara de Y distinta")
    ok = ~np.isnan(X)
    np.testing.assert_allclose(x[ok], X[ok], rtol=0, atol=1e-6, err_msg="X distinta")
    np.testing.assert_allclose(y[ok], Y[ok], rtol=0, atol=1e-6, err_msg="Y distinta")


def comparar(data, ref):
    """Lanza AssertionError con un mensaje claro si algo no coincide."""
    i = registro(data, ref)
    comparar_coordenadas(data, ref)
    for etiqueta, ours, theirs, exacto in esperado(data, i, ref):
        theirs = np.asarray(theirs, dtype=float)
        assert ours.shape == theirs.shape, f"{etiqueta}: forma {ours.shape} != {theirs.shape}"
        msg = f"{ref['Name']} / {etiqueta} distinto de QUICKPLOT (registro {i + 1})"
        if exacto:
            np.testing.assert_array_equal(ours, theirs, err_msg=msg)
        else:
            np.testing.assert_allclose(ours, theirs, rtol=0, atol=ATOL_DERIVADO,
                                       equal_nan=True, err_msg=msg)


# ---------------- contra QUICKPLOT ----------------

@pytest.fixture(scope="module")
def trim_qp():
    data = d3d.read(DEF, dat_path=DAT)
    yield data
    data.close()


PIDE_MAT = ("Para validar contra MATLAB, abre {trim} en QUICKPLOT, guarda con "
            "'Save data' (.mat) campos como 'water level', 'velocity', "
            "'bed load transport', 'total transport' o 'mean total transport' y "
            "copia los .mat a {dir} (ver tests/data/README.md).")


def _pedir(motivo):
    trim = Path(DEF).name if DEF else "el trim (D3D_QUICKPLOT_DEF/DAT)"
    return f"{motivo} {PIDE_MAT.format(trim=trim, dir=QP_DIR)}"


def corresponde(data, ref):
    """None si el .mat se exportó de este trim; si no, el motivo."""
    n, m = data.grid_shape
    malla = np.shape(ref["X"])[:2]
    if malla != (m, n):
        return f"su malla (M, N) = {malla} no es la del trim {(m, n)}."
    try:
        registro(data, ref)
    except AssertionError as e:
        return f"{e}."
    return None


@pytest.mark.parametrize("mat", MATS or [None],
                         ids=[m.stem for m in MATS] or ["sin_exportaciones"])
def test_contra_quickplot(request, mat):
    """Si faltan los .mat o no corresponden al trim, se omite pidiéndolos."""
    if not HAY_TRIM:
        pytest.skip("trim de referencia no disponible: definir D3D_QUICKPLOT_DEF/DAT "
                    "(o D3D_TUMACO_DEF/DAT), o poner los datos en tests/data.")
    if mat is None:
        pytest.skip(_pedir(f"No hay exportaciones de QUICKPLOT en {QP_DIR}."))
    pytest.importorskip("scipy.io")
    data = request.getfixturevalue("trim_qp")
    ref = cargar(mat)
    motivo = corresponde(data, ref)
    if motivo:
        pytest.skip(_pedir(f"{mat.name} no se exportó de {Path(DEF).name}: {motivo}"))
    comparar(data, ref)


# ---------------- comparador (sin QUICKPLOT) ----------------

def _fabricar(data, path, i, nombre, perturbar=False):
    """Escribe un .mat con la estructura de QUICKPLOT a partir de d3d."""
    from scipy.io import savemat
    t = d3d.get_time(data)[i].astype("int64") / 86400000.0 + MATLAB_EPOCH
    s = {"Time": t, "Name": nombre}
    if nombre == "water level":
        s["X"], s["Y"] = _mn(data["XCOR"][0]), _mn(data["YCOR"][0])
        s["Val"] = _celdas(_mn(d3d.water_level(data, index=i)))
    else:
        x, y = (_mn(c) for c in d3d.velocity_xy(data, index=i))
        w = _mascara_kcs(data, _mn(data["WPHY"][i]))
        k = x.shape[2]
        s["X"] = np.repeat(_mn(data["XZ"][0])[..., None], k, axis=2)
        s["Y"] = np.repeat(_mn(data["YZ"][0])[..., None], k, axis=2)
        s["XComp"], s["YComp"], s["ZComp"] = x, y, w
    if perturbar:
        campo = "Val" if "Val" in s else "XComp"
        j = tuple(np.argwhere(~np.isnan(s[campo]))[0])
        s[campo][j] += 0.01
    savemat(str(path), {"data": s})


@pytest.mark.parametrize("nombre", ["water level", "velocity"])
def test_comparador_acepta_datos_identicos(trim_sed, tmp_path, nombre):
    pytest.importorskip("scipy.io")
    _fabricar(trim_sed, tmp_path / "r.mat", 144, nombre)
    comparar(trim_sed, cargar(tmp_path / "r.mat"))


@pytest.mark.parametrize("nombre, mensaje", [("water level", "nivel"), ("velocity", "ux")])
def test_comparador_detecta_discrepancia(trim_sed, tmp_path, nombre, mensaje):
    pytest.importorskip("scipy.io")
    _fabricar(trim_sed, tmp_path / "r.mat", 144, nombre, perturbar=True)
    with pytest.raises(AssertionError, match=mensaje):
        comparar(trim_sed, cargar(tmp_path / "r.mat"))
