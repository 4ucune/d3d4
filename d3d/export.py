"""Exportación a netCDF siguiendo las convenciones CF (1.8).

    d3d.to_netcdf(data, "salida.nc", variables=["S1", "U1", "V1"],
                  index=slice(0, 100), derived=["velocity_xy"])

Qué se escribe
--------------
- Las variables de NEFIS tal como están en el archivo (sin transformar), con
  los ejes (time, N, M[, K]) y atributos: `long_name` (descripción NEFIS),
  `units` (UDUNITS cuando la unidad Delft3D es conocida; la original en
  `delft3d_units`), `delft3d_name`, `delft3d_group`, `delft3d_location`
  (Z, U, V, corner) y `missing_value = -999` en variables reales.
- `coordinates` SOLO para variables en puntos Z (`XZ YZ`) o en esquinas
  (`XCOR YCOR`). Las variables en caras U/V no tienen coordenadas propias en
  el archivo y por eso no llevan el atributo (indicarles XZ/YZ sería falso).
- Tiempo: segundos desde ITDATE (`units = "seconds since ..."`), uno por
  grupo de registros ('time' para map-series/his-series; 'time_<grupo>' para
  otros). Zona horaria del modelo (TZONE en atributos globales).
- Capas sigma: coordenada `K` (centros) y `K_interface` (interfaces) con
  `standard_name = ocean_sigma_coordinate`; `formula_terms` si también se
  exportan S1 y DPS0.
- Campos derivados opcionales (`derived`): 'velocity_xy' → `ux`, `uy`
  (cartesianas en centros, ver d3d.velocity_xy); 'water_level' →
  `water_level` (S1 con celdas inactivas y secas en NaN).

Motores
-------
- 'netcdf4': requiere el paquete netCDF4; NETCDF4 con compresión opcional y
  escritura por bloques de registros (memoria acotada).
- 'scipy': scipy.io.netcdf_file, netCDF3 de 64 bits. Mantiene cada variable
  completa en memoria antes de escribir: adecuado para subconjuntos; se
  rechaza si el total supera `max_memory_bytes`.
- 'auto' (por defecto): netcdf4 si está instalado; si no, scipy.
"""
import datetime as _dt
import re
from pathlib import Path

import numpy as np

from . import fields
from .timeaxis import TIME_COUNTERS, get_time, reference_date

__all__ = ["to_netcdf", "variable_location"]

_CHUNK = 50

# --- unidades Delft3D → UDUNITS ---------------------------------------------
_UNITS = {
    "M": "m", "M/S": "m s-1", "M2/S": "m2 s-1", "KG/M3": "kg m-3",
    "N/M2": "N m-2", "M3/S/M": "m2 s-1", "DEG": "degree", "M2": "m2",
    "KG/M2": "kg m-2", "S": "s", "-": "1", "J/M2": "J m-2", "W/M2": "W m-2",
    "M3/S": "m3 s-1", "KG/M3/S": "kg m-3 s-1", "M2/S2": "m2 s-2", "M2/S3": "m2 s-3",
}

# --- ubicación en la malla escalonada ---------------------------------------
_LOC_BY_NAME = {
    "Z": {"S1", "KCS", "XZ", "YZ", "ALFAS", "DPS0", "DPS", "GSQS", "R1", "RHO", "WPHY",
          "W", "TAUMAX", "VICUV", "VICWW", "DICWW", "RICH", "RTUR1", "PPARTITION",
          "WS", "RCA", "MSED", "LYRFRAC", "DP_BEDLYR"},
    "U": {"U1", "KFU", "KCU", "DPU0", "TAUKSI", "UMNLDF", "SBUU", "SBUUA", "SSUU", "SSUUA"},
    "V": {"V1", "KFV", "KCV", "DPV0", "TAUETA", "VMNLDF", "SBVV", "SBVVA", "SSVV", "SSVVA"},
    "corner": {"XCOR", "YCOR", "DP0"},
}


def variable_location(name, description=""):
    """'Z', 'U', 'V', 'corner' o 'unknown' (tabla por nombre; si no, descripción)."""
    for loc, names in _LOC_BY_NAME.items():
        if name in names:
            return loc
    d = description.lower()
    if re.search(r"\bu[- ]point|\(u point\)", d):
        return "U"
    if re.search(r"\bv[- ]point|\(v point\)", d):
        return "V"
    if "zeta point" in d or "water level point" in d or "water-level point" in d:
        return "Z"
    return "unknown"


def _safe(name):
    """Nombre válido en netCDF/CF (letras, dígitos, '_')."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return s if s[:1].isalpha() else "v_" + s


def _udunits(raw):
    key = raw.strip().strip("[]").strip().upper()
    return _UNITS.get(key)


def _scalar(data, name, default=None):
    if name not in data.variables:
        return default
    v = np.asarray(data[name].raw[0]).ravel()
    if v.dtype.kind == "S":
        return v[0].decode("ascii", "replace").strip()
    return v[0].item() if v.size == 1 else v


# ---------------------------------------------------------------------------
# Plan de exportación (independiente del motor)
# ---------------------------------------------------------------------------

class _Var:
    def __init__(self, name, dims, dtype, attrs, writer, shape, time_dim=None):
        self.name, self.dims, self.dtype, self.attrs = name, dims, dtype, attrs
        self.writer = writer            # writer(sink): escribe los datos en sink[...]
        self.shape, self.time_dim = shape, time_dim


def _plan(data, variables, index, derived, rename):
    nmax = int(_scalar(data, "NMAX", 0) or 0)
    mmax = int(_scalar(data, "MMAX", 0) or 0)
    kmax = int(_scalar(data, "KMAX", 0) or 0)
    layer_model = str(_scalar(data, "LAYER_MODEL", "") or "").upper()
    spherical = str(_scalar(data, "COORDINATES", "") or "").upper().startswith("SPHER")
    t0 = reference_date(data)

    dims, coords, out = {}, {}, []
    time_groups = {}

    def time_dim_for(group):
        if group not in time_groups:
            name = "time" if group in ("map-series", "his-series") else "time_" + _safe(group)
            recs, scalar = data[_first_var_of(data, group)]._time_indices(
                slice(None) if index is None else index)
            if scalar:
                recs = recs[:1]
            sec = get_time(data, group, as_seconds=True)[recs]
            time_groups[group] = (name, recs)
            dims[name] = len(recs)
            coords[name] = _Var(name, (name,), np.float64, {
                "standard_name": "time", "long_name": f"time of group {group}",
                "units": f"seconds since {str(t0).replace('T', ' ')}",
                "calendar": "standard", "axis": "T"},
                lambda sink, s=sec: sink.__setitem__(slice(None), s), (len(recs),))
        return time_groups[group]

    def is_time_group(group):
        return group in TIME_COUNTERS

    def spatial_dims(vname, edims):
        out_d = []
        for i, d in enumerate(edims):
            if i == 0 and d == nmax and len(edims) >= 2 and edims[1] == mmax:
                nm = "N"
            elif i == 1 and d == mmax and out_d[:1] == ["N"]:
                nm = "M"
            elif i == 2 and out_d[:2] == ["N", "M"] and kmax and d == kmax:
                nm = "K"
            elif i == 2 and out_d[:2] == ["N", "M"] and kmax and d == kmax + 1:
                nm = "K_interface"
            else:
                nm = f"{_safe(vname)}_dim{i}"
            if dims.get(nm, d) != d:
                raise ValueError(f"dimensión {nm} con tamaños distintos ({dims[nm]} y {d})")
            dims[nm] = d
            out_d.append(nm)
        return out_d

    def base_attrs(v, name, loc):
        a = {"long_name": v.attributes["description"] or name, "delft3d_name": name,
             "delft3d_group": v.group, "delft3d_location": loc,
             "delft3d_units": v.attributes["units"].strip()}
        u = _udunits(v.attributes["units"])
        if u:
            a["units"] = u
        if loc == "Z" and name not in ("XZ", "YZ"):
            a["coordinates"] = "XZ YZ"
        elif loc == "corner" and name not in ("XCOR", "YCOR"):
            a["coordinates"] = "XCOR YCOR"
        return a

    needs = set()
    src_to_nc = {}                       # nombre NEFIS -> nombre en el netCDF
    for name in variables:
        v = data.variables.get(name)
        if v is None:
            raise KeyError(f"variable '{name}' no encontrada")
        dtype = np.dtype(data._reader.raw_reader._elements[name]["dtype"])
        if dtype.kind == "c":
            raise TypeError(f"'{name}' es complejo: netCDF no lo admite")
        loc = variable_location(name, v.attributes["description"])
        if loc == "Z":
            needs.add("Z")
        elif loc == "corner":
            needs.add("corner")
        edims = [int(d) for d in v.shape[1:]]
        sdims = spatial_dims(name, edims)
        attrs = base_attrs(v, name, loc)
        if dtype.kind == "f":
            attrs["missing_value"] = dtype.type(-999.0)
        if dtype.kind == "S":                       # texto: dimensión de caracteres
            sl = f"{_safe(name)}_strlen"
            dims[sl] = dtype.itemsize
            sdims = sdims + [sl]
        nc_name = _safe(rename.get(name, name))
        src_to_nc[name] = nc_name
        if is_time_group(v.group):
            tname, recs = time_dim_for(v.group)
            shape = (len(recs),) + tuple(edims)

            def writer(sink, v=v, recs=recs):
                for i in range(0, len(recs), _CHUNK):
                    blk = recs[i:i + _CHUNK]
                    sink[i:i + len(blk)] = _as_store(v.raw[blk])
            out.append(_Var(nc_name, tuple([tname] + sdims), dtype, attrs, writer, shape, tname))
        else:
            shape = tuple(edims)
            out.append(_Var(nc_name, tuple(sdims), dtype, attrs,
                            lambda sink, v=v: sink.__setitem__(Ellipsis, _as_store(v.raw[0])), shape))

    # ---- derivados
    for d in derived:
        if d not in ("velocity_xy", "water_level"):
            raise ValueError(f"derivado desconocido: {d!r} ('velocity_xy', 'water_level')")
        needs.add("Z")
        src = "U1" if d == "velocity_xy" else "S1"
        v = data[src]
        tname, recs = time_dim_for(v.group)
        sdims = spatial_dims(src, [int(x) for x in v.shape[1:]])
        shape = (len(recs),) + tuple(v.shape[1:])
        if d == "water_level":
            attrs = {"long_name": "water level (S1) with inactive and dry cells as NaN",
                     "units": "m", "coordinates": "XZ YZ", "delft3d_location": "Z",
                     "comment": "d3d.water_level: -999/999 -> NaN, |KCS|!=1 -> NaN, "
                                "dry cells (KFU/KFV closed on 4 faces) -> NaN"}

            def w(sink, recs=recs):
                for i in range(0, len(recs), _CHUNK):
                    blk = recs[i:i + _CHUNK]
                    sink[i:i + len(blk)] = fields.water_level(data, index=blk)
            out.append(_Var("water_level", tuple([tname] + sdims), np.dtype(np.float32),
                            attrs, w, shape, tname))
        else:
            comment = ("d3d.velocity_xy: U1/V1 -> uv2cen (cell centres) -> rotation with "
                       "ALFAS (cur2ca) -> |KCS|!=1 -> NaN")
            for comp, sn in (("ux", "sea_water_x_velocity"), ("uy", "sea_water_y_velocity")):
                attrs = {"standard_name": sn, "long_name": f"horizontal velocity, "
                         f"{'x' if comp == 'ux' else 'y'} component at cell centres",
                         "units": "m s-1", "coordinates": "XZ YZ", "delft3d_location": "Z",
                         "comment": comment}

                def w(sink, recs=recs, c=comp):
                    for i in range(0, len(recs), _CHUNK):
                        blk = recs[i:i + _CHUNK]
                        x, y = fields.velocity_xy(data, index=blk)
                        sink[i:i + len(blk)] = x if c == "ux" else y
                out.append(_Var(comp, tuple([tname] + sdims), np.dtype(np.float32),
                                attrs, w, shape, tname))

    # ---- coordenadas horizontales
    for flag, (xn, yn) in (("Z", ("XZ", "YZ")), ("corner", ("XCOR", "YCOR"))):
        if flag not in needs:
            continue
        for cn, axis in ((xn, "X"), (yn, "Y")):
            if cn not in data.variables:
                continue
            if cn in src_to_nc:                 # ya exportada (quizá renombrada):
                o = next(o for o in out if o.name == src_to_nc[cn])
                o.attrs.pop("coordinates", None)
                o.attrs["standard_name"] = (
                    ("longitude" if axis == "X" else "latitude") if spherical
                    else ("projection_x_coordinate" if axis == "X" else "projection_y_coordinate"))
                continue
            src_to_nc[cn] = cn
            v = data[cn]
            sdims = spatial_dims(cn, [int(x) for x in v.shape[1:]])
            a = base_attrs(v, cn, "Z" if flag == "Z" else "corner")
            if spherical:
                a["standard_name"] = "longitude" if axis == "X" else "latitude"
                a["units"] = "degrees_east" if axis == "X" else "degrees_north"
            else:
                a["standard_name"] = "projection_x_coordinate" if axis == "X" \
                    else "projection_y_coordinate"
            a["comment"] = "0 en celdas inactivas (valor del archivo)"
            out.append(_Var(cn, tuple(sdims), np.dtype(np.float64), a,
                            lambda sink, v=v: sink.__setitem__(Ellipsis, v.raw[0]),
                            tuple(v.shape[1:])))

    # ---- `coordinates` con los nombres finales (si se renombraron XZ/YZ/...)
    for o in out:
        if "coordinates" in o.attrs:
            o.attrs["coordinates"] = " ".join(src_to_nc.get(c, c)
                                              for c in o.attrs["coordinates"].split())
    s1_nc, dps0_nc = src_to_nc.get("S1"), src_to_nc.get("DPS0")

    # ---- coordenada vertical sigma
    if layer_model.startswith("SIGMA") and "THICK" in data.variables:
        # THICK es REAL*4: se toma el decimal más corto de cada valor (0.04, no
        # 0.03999999910593033) para que las sigmas no arrastren el redondeo.
        thick = np.array([float(str(np.float32(x)))
                          for x in np.asarray(data["THICK"].raw[0]).ravel()])
        iface = -np.concatenate([[0.0], np.cumsum(thick)])
        centers = 0.5 * (iface[:-1] + iface[1:])
        has_terms = s1_nc is not None and dps0_nc is not None
        for dn, vals, ln in (("K", centers, "sigma at layer centres"),
                             ("K_interface", iface, "sigma at layer interfaces")):
            if dn not in dims or dims[dn] != len(vals):
                continue
            a = {"long_name": ln + " (k=0 is the surface layer)", "positive": "up",
                 "units": "1", "axis": "Z"}
            if has_terms:
                a["standard_name"] = "ocean_sigma_coordinate"
                a["formula_terms"] = f"sigma: {dn} eta: {s1_nc} depth: {dps0_nc}"
            else:
                a["comment"] = "formula_terms omitted: export S1 and DPS0 to define it"
            coords[dn] = _Var(dn, (dn,), np.float64, a,
                              lambda sink, x=vals: sink.__setitem__(slice(None), x), (len(vals),))

    return dims, list(coords.values()) + out


def _first_var_of(data, group):
    for n, v in data.variables.items():
        if v.group == group:
            return n
    raise KeyError(f"grupo '{group}' sin variables")


def _as_store(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":                     # (…) S<n> → (…, n) caracteres
        return a.view("S1").reshape(a.shape + (a.dtype.itemsize,))
    return a


def _global_attrs(data, source):
    g = {"Conventions": "CF-1.8",
         "title": "Delft3D-FLOW results exported with d3d",
         "source": f"Delft3D-FLOW NEFIS file: {Path(source).name}",
         "history": f"{_dt.datetime.now().isoformat(timespec='seconds')}: d3d.to_netcdf"}
    for k in ("SIMDAT", "LAYER_MODEL", "COORDINATES", "TZONE", "NMAX", "MMAX", "KMAX",
              "DT", "TUNIT", "GRDANG"):
        val = _scalar(data, k)
        if val is not None:
            g[f"delft3d_{k}"] = val if not isinstance(val, np.ndarray) else val.tolist()
    it = _scalar(data, "ITDATE")
    if it is not None:
        g["delft3d_ITDATE"] = " ".join(str(int(x)) for x in np.atleast_1d(it))
    g["time_zone_note"] = "times are in model time (TZONE hours from UTC), not converted"
    return g


# ---------------------------------------------------------------------------
# Motores
# ---------------------------------------------------------------------------

_DEFAULT_VARS = ("S1", "U1", "V1", "KCS", "ALFAS", "DPS0", "XZ", "YZ", "XCOR", "YCOR")


def to_netcdf(data, path, variables=None, index=None, derived=(), rename=None,
              engine="auto", compress=True, max_memory_bytes=2 * 1024 ** 3):
    """Exporta variables de un TrimFile a netCDF (CF-1.8).

    data: TrimFile (d3d.read).
    variables: nombres NEFIS a exportar. None → S1, U1, V1, KCS, ALFAS, DPS0 y
        las coordenadas, si existen.
    index: selección de registros (0-based) aplicada a cada grupo con tiempo
        (None = todos; entero, slice, lista).
    derived: ('velocity_xy', 'water_level') campos derivados opcionales.
    rename: {nombre NEFIS: nombre en el netCDF}.
    engine: 'auto' | 'netcdf4' | 'scipy'.
    compress: compresión zlib (solo motor netcdf4).
    max_memory_bytes: límite de tamaño total para el motor scipy.
    Devuelve la ruta escrita.
    """
    if variables is None:
        variables = [n for n in _DEFAULT_VARS if n in data.variables]
    variables = list(dict.fromkeys(variables))
    rename = dict(rename or {})
    dims, plan = _plan(data, variables, index, list(derived), rename)
    names = [p.name for p in plan]
    if len(names) != len(set(names)):
        raise ValueError(f"nombres repetidos en la salida: {names}")

    if engine == "auto":
        try:
            import netCDF4  # noqa: F401
            engine = "netcdf4"
        except ImportError:
            engine = "scipy"
    gattrs = _global_attrs(data, data._reader.dat_file())
    path = str(path)
    if engine == "netcdf4":
        _write_netcdf4(path, dims, plan, gattrs, compress)
    elif engine == "scipy":
        try:
            import scipy.io  # noqa: F401
        except ImportError:
            raise ImportError("to_netcdf necesita netCDF4 o scipy: "
                              "pip install netCDF4  (o: pip install scipy)") from None
        total = sum(int(np.prod(p.shape)) * np.dtype(p.dtype).itemsize for p in plan)
        if total > max_memory_bytes:
            raise MemoryError(
                f"exportación de {total / 2**30:.1f} GB con el motor scipy (todo en memoria). "
                f"Seleccione un subconjunto (index=, variables=) o instale netCDF4.")
        _write_scipy(path, dims, plan, gattrs)
    else:
        raise ValueError("engine debe ser 'auto', 'netcdf4' o 'scipy'")
    return path


def _nc_dtype(dt):
    dt = np.dtype(dt)
    return "S1" if dt.kind == "S" else dt.newbyteorder("=")


def _write_netcdf4(path, dims, plan, gattrs, compress):
    import netCDF4
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncatts(gattrs)
        for d, n in dims.items():
            ds.createDimension(d, n)
        for p in plan:
            kw = {}
            if "missing_value" in p.attrs:
                kw["fill_value"] = False
            var = ds.createVariable(p.name, _nc_dtype(p.dtype), p.dims,
                                    zlib=compress and np.dtype(p.dtype).kind != "S", **kw)
            var.set_auto_maskandscale(False)
            var.setncatts(p.attrs)
            p.writer(var)


class _ScipySink:
    """Adaptador: escribe en el arreglo de scipy y conserva la API sink[...]."""
    def __init__(self, var):
        self.var = var

    def __setitem__(self, key, value):
        self.var[key] = value


def _write_scipy(path, dims, plan, gattrs):
    from scipy.io import netcdf_file
    with netcdf_file(path, "w", version=2) as ds:
        for k, v in gattrs.items():
            setattr(ds, k, v)
        for d, n in dims.items():
            ds.createDimension(d, n)
        for p in plan:
            dt = _nc_dtype(p.dtype)
            if dt == "S1":
                dt = "c"
            var = ds.createVariable(p.name, dt, p.dims)
            for k, v in p.attrs.items():
                setattr(var, k, v)
            p.writer(_ScipySink(var))
