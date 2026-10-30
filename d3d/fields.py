"""Campos derivados y máscaras por variable, con orden de ejes (T, N, M[, K]).

Cada función lee solo los registros (y capas) pedidos y aplica, de forma
explícita y documentada, las transformaciones de `d3d.transforms`
(réplica de d3d_trimfil.m). El orden de ejes es el de NEFIS / vs_let.m:

    T (registro), N, M, K (capa; en modelos sigma k=0 es la superficie)

Selector `index` (0-based): None = todos; entero = un registro (sin eje T);
slice, lista o máscara booleana = varios registros.
Selector `layer`: None = todas las capas; entero o slice sobre K. Se ignora
en campos 2D.

Máscara al leer (`mask_read`, la usa `Variable[...]`): depende de la
posición de la variable en la malla escalonada, como la columna Loc/ReqLoc
de d3d_trimfil.m. No mueve ni interpola datos; solo pone NaN.
"""
import numpy as np

from . import transforms as tr

__all__ = ["water_level", "vector_xy", "velocity_xy", "derived", "scalar",
           "scalars", "DERIVED", "SCALARS", "location", "mask_read"]

_CHUNK = 50          # registros por bloque (limita los temporales en memoria)


# ---------- máscara por variable ----------

# Posición de cada elemento NEFIS (columnas Val1/Val2 y Loc de DataProps en
# d3d_trimfil.m). Un elemento de ejes (N, M, ...) que no está aquí es de
# centro de celda ('z').
_LOC_U = frozenset("""
    U1 UMNLDF TAUKSI SBUU SSUU SBUUA SSUUA SBCUU SBWUU SSWUU SUCOR DZDUU
    R1FLX_UU R1FLX_UUC QXKR QXKW FXW WSU U_ICE WAVE_FORCE_X ROLLER_FORCE_X
    DPU0 GUU MOM_DUDT MOM_UDENSITY MOM_URESISTANCE MOM_UCORIOLIS MOM_UVISCO
    MOM_UPRESSURE MOM_UTIDEGEN MOM_UWINDFORCE MOM_UBEDSHEAR MOM_UWAVES
    MOM_UDUDX MOM_VDUDY""".split())
_LOC_V = frozenset("""
    V1 VMNLDF TAUETA SBVV SSVV SBVVA SSVVA SBCVV SBWVV SSWVV SVCOR DZDVV
    R1FLX_VV R1FLX_VVC QYKR QYKW FYW WSV V_ICE WAVE_FORCE_Y ROLLER_FORCE_Y
    DPV0 GVV MOM_DVDT MOM_VDENSITY MOM_VRESISTANCE MOM_VCORIOLIS MOM_VVISCO
    MOM_VPRESSURE MOM_VTIDEGEN MOM_VWINDFORCE MOM_VBEDSHEAR MOM_VWAVES
    MOM_VDVDY MOM_UDVDX""".split())
_LOC_D = frozenset({"XCOR", "YCOR", "DP0"})

# Valores "sin dato" que pasan a NaN. S1 usa también 999 (d3d_trimfil.m).
_SENTINELS = {"S1": (-999.0, 999.0)}
_DEFAULT_SENTINELS = (-999.0,)


def location(name, dims, dtype, grid_shape):
    """Posición de un elemento en la malla: 'z', 'u', 'v', 'd' o None.

    None = no se enmascara: archivos sin KCS (trih), elementos sin ejes
    (N, M) iniciales, y enteros o texto (KCS, KFU, ITMAPC, NAMCON, ...).
    """
    if grid_shape is None or np.dtype(dtype).kind != "f":
        return None
    if tuple(int(d) for d in dims[:2]) != tuple(grid_shape):
        return None
    if name in _LOC_U:
        return "u"
    if name in _LOC_V:
        return "v"
    if name in _LOC_D:
        return "d"
    return "z"


def _valid2d(data, loc):
    """Máscara (N, M) de puntos válidos según la posición (en caché)."""
    cache = data.__dict__.setdefault("_mask_cache", {})
    if loc not in cache:
        act = tr.active_mask(_const2d(data, "KCS"))
        # z: |KCS| == 1. u/v/d: como los campos 'staggered' y de esquina de
        # d3d_trimfil.m (ReqLoc 'd'): alguna de las 4 celdas vecinas activa.
        cache[loc] = act if loc == "z" else tr.corner_mask(act)
    return cache[loc]


def _has_dry_mask(data, var):
    if var.name != "S1":
        return False
    kfu, kfv = data.variables.get("KFU"), data.variables.get("KFV")
    return (kfu is not None and kfv is not None
            and kfu.group == kfv.group == var.group)


def mask_read(data, var, records, scalar, rest, out):
    """Aplica en el sitio la máscara de `var` a `out` = var.raw[(records,) + rest].

    Según `var.location` (ver `location`):
      - todos: centinelas → NaN (-999; en S1 también 999)
      - 'z' (centros: S1, R1, RHO, DPS0, XZ, ...): |KCS| != 1 → NaN
      - 'u', 'v' (caras: U1, SBUU, TAUKSI, ...) y 'd' (esquinas: XCOR, DP0):
        NaN si ninguna de las 4 celdas vecinas es activa (conv2 2×2)
      - S1 además: celdas secas (KFU y KFV cerrados en las 4 caras) → NaN,
        como 'water level' en d3d_trimfil.m
    """
    loc = var.location
    if loc is None or out.size == 0:
        return out
    for s in _SENTINELS.get(var.name, _DEFAULT_SENTINELS):
        np.copyto(out, np.nan, where=out == s)
    rec_shape = var.shape[1:]
    valid = _valid2d(data, loc)
    full = np.broadcast_to(valid.reshape(valid.shape + (1,) * (len(rec_shape) - 2)), rec_shape)
    np.copyto(out, np.nan, where=~(full[rest] if rest else full))
    if _has_dry_mask(data, var):
        n, m = rec_shape[:2]
        ii = np.broadcast_to(np.arange(n).reshape(n, 1), (n, m))
        jj = np.broadcast_to(np.arange(m).reshape(1, m), (n, m))
        if rest:
            ii, jj = ii[rest], jj[rest]
        ip, jp = np.maximum(ii - 1, 0), np.maximum(jj - 1, 0)
        reader = data._reader
        for i, t in enumerate(records):
            kfu = reader.view_element(var.group, "KFU", t)
            kfv = reader.view_element(var.group, "KFV", t)
            dry = ((kfu[ii, jj] == 0) & (kfu[ii, jp] == 0)
                   & (kfv[ii, jj] == 0) & (kfv[ip, jj] == 0))
            np.copyto(out if scalar else out[i, ...], np.nan, where=dry)
    return out


# ---------- utilidades ----------

def _var(data, name):
    v = data.variables.get(name)
    if v is None:
        raise KeyError(f"variable '{name}' no encontrada")
    return v


def _records(v, index):
    """→ (lista de registros 0-based, ¿índice escalar?)."""
    return v._time_indices(slice(None) if index is None else index)


def _const2d(data, name):
    v = _var(data, name)
    return np.asarray(v.raw[0]).reshape(v.shape[1:3])


def _is_zmodel(data):
    if "LAYER_MODEL" not in data.variables:
        return False
    txt = np.asarray(data["LAYER_MODEL"].raw[0]).ravel()[0]
    return txt.strip().upper().startswith(b"Z")


def _blocks(records):
    for i in range(0, len(records), _CHUNK):
        yield i, records[i:i + _CHUNK]


def _names(name):
    return (name,) if isinstance(name, str) else tuple(name)


# ---------- campos ----------

def water_level(data, index=None):
    """Nivel de agua S1 en centros de celda → (T, N, M) o (N, M).

    Como d3d_trimfil.m: -999/999 → NaN; celdas inactivas (|KCS| != 1) → NaN;
    celdas secas (KFU y KFV cerrados en las 4 caras, en ese registro) → NaN.
    Coincide con `get_var(data, "S1", index)` (máscara por defecto).
    """
    s1 = _var(data, "S1")
    recs, scalar = _records(s1, index)
    n, m = s1.shape[1:3]
    act = tr.active_mask(_const2d(data, "KCS"))
    has_kf = "KFU" in data.variables and "KFV" in data.variables
    out = np.empty((len(recs), n, m), dtype=np.float32)
    for i, blk in _blocks(recs):
        a = tr.invalid_to_nan(s1.raw[blk], sentinels=(-999.0, 999.0))
        a = tr.apply_mask(a, act, n_axis=1, m_axis=2)
        if has_kf:
            dry = tr.dry_mask(data["KFU"].raw[blk], data["KFV"].raw[blk], n_axis=1, m_axis=2)
            a[dry] = np.nan
        out[i:i + len(blk)] = a
    return out[0] if scalar else out


def vector_xy(data, u_name, v_name, index=None, layer=None):
    """Par de componentes de malla en caras U/V → (x, y) cartesianas en centros.

    Pipeline (d3d_trimfil.m, Loc='u' → ReqLoc='z', VecType='u'):
      1. -999 → NaN (en modelos Z también 0 → NaN)
      2. uv2cen: U media en M, V media en N (primera fila/columna NaN)
      3. cur2ca: rotación con ALFAS (dirección M respecto a +X)
      4. celdas inactivas (|KCS| != 1) → NaN

    Sirve para U1/V1, UMNLDF/VMNLDF, SBUUA/SBVVA, SSUUA/SSVVA, ...
    `u_name`/`v_name` pueden ser tuplas de nombres que se suman en las caras
    antes de interpolar (p. ej. ('SSUU', 'SBUU') = 'total transport').
    Devuelve (x, y), cada uno (T, N, M[, K]) o (N, M[, K]) si `index` es entero.
    """
    us = [_var(data, n) for n in _names(u_name)]
    vs = [_var(data, n) for n in _names(v_name)]
    vu, vv = us[0], vs[0]
    for w in us + vs:
        if w.shape != vu.shape:
            raise ValueError(f"{w.name} {w.shape} y {vu.name} {vu.shape} no tienen la misma forma")
    recs, scalar = _records(vu, index)
    if len(vu.shape) < 3:
        raise ValueError(f"'{vu.name}' no es un campo de malla (forma {vu.shape})")
    step_rest = (slice(None), slice(None))
    extra = vu.shape[3:]
    if extra and layer is not None:
        step_rest = step_rest + (layer,)
    alfas = _const2d(data, "ALFAS")
    act = tr.active_mask(_const2d(data, "KCS"))
    zmodel = _is_zmodel(data)

    xs, ys = [], []
    for _, blk in _blocks(recs):
        key = (blk,) + step_rest
        u = sum(tr.invalid_to_nan(w.raw[key]) for w in us)
        v = sum(tr.invalid_to_nan(w.raw[key]) for w in vs)
        if zmodel:
            u[u == 0] = np.nan
            v[v == 0] = np.nan
        uc, vc = tr.uv2cen(u, v, n_axis=1, m_axis=2)
        x, y = tr.cur2ca(uc, vc, alfas, n_axis=1, m_axis=2)
        xs.append(tr.apply_mask(x, act, n_axis=1, m_axis=2))
        ys.append(tr.apply_mask(y, act, n_axis=1, m_axis=2))
    if not xs:
        return np.array([]), np.array([])
    x = np.concatenate(xs) if len(xs) > 1 else xs[0]
    y = np.concatenate(ys) if len(ys) > 1 else ys[0]
    return (x[0], y[0]) if scalar else (x, y)


def velocity_xy(data, index=None, layer=None, filtered=False):
    """Velocidad horizontal cartesiana en centros de celda.

    filtered=False: U1/V1 (3D, por capa) → (T, N, M, K).
    filtered=True : UMNLDF/VMNLDF (promediada en la vertical y filtrada) → (T, N, M).
    Ver `vector_xy` para el procedimiento.
    """
    if filtered:
        return vector_xy(data, "UMNLDF", "VMNLDF", index=index)
    return vector_xy(data, "U1", "V1", index=index, layer=layer)


# ---------- variables derivadas por nombre (QUICKPLOT / d3d_trimfil.m) ----------

# nombre QUICKPLOT → (componentes U, componentes V, tipo). 'centro': vector_xy
# (uv2cen + cur2ca + KCS). 'staggered': caras sin interpolar ni rotar, con la
# máscara de caras (conv2 2×2), es decir `get_var` por defecto.
DERIVED = {
    "horizontal velocity": (("U1",), ("V1",), "centro"),
    "filtered depth averaged velocity": (("UMNLDF",), ("VMNLDF",), "centro"),
    "bed shear stress": (("TAUKSI",), ("TAUETA",), "centro"),
    "bed load transport": (("SBUU",), ("SBVV",), "centro"),
    "d.a. suspended transport": (("SSUU",), ("SSVV",), "centro"),
    "total transport": (("SSUU", "SBUU"), ("SSVV", "SBVV"), "centro"),
    "mean bed load transport": (("SBUUA",), ("SBVVA",), "centro"),
    "mean d.a. suspended transport": (("SSUUA",), ("SSVVA",), "centro"),
    "mean total transport": (("SSUUA", "SBUUA"), ("SSVVA", "SBVVA"), "centro"),
    "staggered bedload transport": (("SBUU",), ("SBVV",), "staggered"),
}

_SED_GROUPS = ("map-sed-series", "map-avg-series")


def derived(data, name, index=None, component=None):
    """Variable derivada vectorial por nombre → (X, Y, magnitud).

    Ejes (T, N, M[, K]); (N, M[, K]) si `index` es entero. En los transportes
    el eje de fracciones de sedimento se elimina si hay una sola fracción
    (si hay varias se conserva al final: (T, N, M, L)).
    component: None → (X, Y, mag); 'X', 'Y' o 'mag' → solo esa componente.
    """
    if name not in DERIVED:
        raise KeyError(f"variable derivada '{name}' no existe; disponibles: {sorted(DERIVED)}")
    us, vs, kind = DERIVED[name]
    if component not in (None, "X", "Y", "mag", "M"):
        raise ValueError(f"componente inválida: {component!r} ('X', 'Y' o 'mag')")
    if kind == "centro":
        x, y = vector_xy(data, us, vs, index=index)
    else:
        key = slice(None) if index is None else index
        x = sum(_var(data, n)[key] for n in us)
        y = sum(_var(data, n)[key] for n in vs)
    src = _var(data, us[0])
    if src.group in _SED_GROUPS and len(src.shape) == 4 and src.shape[-1] == 1:
        x, y = x[..., 0], y[..., 0]
    if component == "X":
        return x
    if component == "Y":
        return y
    mag = np.hypot(x, y)
    return mag if component in ("mag", "M") else (x, y, mag)


def derived_shape(data, name):
    """Forma de cada componente de `derived(data, name)` (todos los registros)."""
    us, _, _ = DERIVED[name]
    src = _var(data, us[0])
    shape = src.shape
    if src.group in _SED_GROUPS and len(shape) == 4 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


# ---------- escalares de QUICKPLOT ----------

# Campos de QUICKPLOT que en d3d_trimfil.m son la variable NEFIS (Val1) sin
# más cálculo que la máscara. Los que llevan cálculo propio ('water depth',
# 'bed level ...', 'depth averaged ...', presiones, ...) no están.
SCALARS = {
    "water level": "S1",
    "vertical velocity": "WPHY",
    "density": "RHO",
    "non-hydrostatic pressure": "HYDPRES",
    "horizontal viscosity": "VICUV",
    "vertical eddy viscosity": "VICWW",
    "vertical eddy diffusivity": "DICWW",
    "richardson number": "RICH",
    "vorticity": "VORTIC",
    "enstrophy": "ENSTRO",
    "maximum bed shear stress": "TAUMAX",
    "characteristic velocity magnitude": "UMOD",
    "bed shear velocity magnitude": "USTAR",
    "near bed reference concentration": "RCA",
    "equilibrium concentration": "RSEDEQ",
    "settling velocity": "WS",
    "sediment fraction in top layer": "FRAC",
    "mud fraction in top layer": "MUDFRAC",
    "arithmetic mean sediment diameter": "DM",
    "geometric mean sediment diameter": "DG",
    "available mass of sediment": "MSED",
    "sediment fraction": "LYRFRAC",
    "bed porosity": "EPSPOR",
    "air pressure": "PATM",
    "air temperature": "AIRTEM",
    "precipitation rate": "PRECIP",
    "evaporation rate": "EVAP",
    "significant wave height": "HS",
    "wave height": "WAVE_HEIGHT",
    "peak wave period": "TP",
    "orbital velocity amplitude": "UORB",
}


def _constituents(data):
    """Nombres de NAMCON → (elemento, posición): los LSTCI primeros en R1, el
    resto (turbulencia) en RTUR1, como '--constituents' / '--turbquant'."""
    if "NAMCON" not in data.variables:
        return {}
    names = [s.decode("ascii", "replace").strip()
             for s in np.ravel(data["NAMCON"].raw[0])]
    lstci = int(np.ravel(data["LSTCI"].raw[0])[0]) if "LSTCI" in data.variables else 0
    out = {}
    for i, nm in enumerate(names):
        el, j = ("R1", i) if i < lstci else ("RTUR1", i - lstci)
        if nm and el in data.variables:
            out[nm] = (el, j)
    return out


def scalars(data):
    """Escalares de QUICKPLOT presentes en el archivo → {nombre: (elemento, posición)}.

    `posición` es el índice del último eje (constituyentes) o None.
    """
    out = {nm: (el, None) for nm, el in SCALARS.items() if el in data.variables}
    out.update(_constituents(data))
    return out


def find_scalar(data, name):
    """→ (nombre, elemento, posición) sin distinguir mayúsculas, o None."""
    for nm, (el, j) in scalars(data).items():
        if nm.lower() == name.lower():
            return nm, el, j
    return None


def scalar(data, name, index=None, mask=True):
    """Escalar de QUICKPLOT por nombre (p. ej. 'water level', 'Salinity').

    Igual que leer la variable NEFIS (con su máscara salvo mask=False); para
    los constituyentes se toma la posición de NAMCON en el último eje.
    `index` acepta los mismos selectores que `Variable[...]`.
    """
    hit = find_scalar(data, name)
    if hit is None:
        raise KeyError(f"'{name}' no es un campo escalar de QUICKPLOT en este archivo")
    _, el, j = hit
    v = _var(data, el)
    reader = v if mask else v.raw
    a = reader[slice(None) if index is None else index]
    return a if j is None else a[..., j]


def scalar_shape(data, name):
    _, el, j = find_scalar(data, name)
    shape = _var(data, el).shape
    return shape if j is None else shape[:-1]
