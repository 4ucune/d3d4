"""Eje de tiempo de archivos Delft3D-FLOW (trim / trih).

Réplica de `readtim` de `d3d_trimfil.m` y `d3d_trihfil.m` (toolbox MATLAB):

    tiempo hidrodinámico = ITDATE + contador * DT * TUNIT   [segundos]
    tiempo morfológico   = ITDATE + MORFT                    [días]

- ITDATE = [YYYYMMDD, HHMMSS] (constante del archivo).
- DT: paso de tiempo en unidades TUNIT; TUNIT: segundos por unidad.
- El contador depende del grupo (ver `_COUNTERS`): ITMAPC, ITMAPS, ITAVGS
  o ITHISC.
- Los tiempos están en la zona horaria del modelo (constante TZONE, horas
  respecto a UTC); no se convierten, igual que en MATLAB.

Precisión: DT se guarda como REAL*4 (0.2 → 0.2000000029802…). Para no
arrastrar ese error de representación (≈1e-8 relativo, milisegundos tras
muchos pasos), DT se toma como el decimal más corto que representa ese
float32 (0.2), que es el valor que escribió el usuario en el .mdf. El
resultado se redondea a milisegundos.
"""
import numpy as np

__all__ = ["reference_date", "get_time", "TIME_COUNTERS"]

# grupo de datos -> (grupo con el contador, elemento contador)
TIME_COUNTERS = {
    "map-series": ("map-info-series", "ITMAPC"),
    "map-info-series": ("map-info-series", "ITMAPC"),
    "map-sedgs-series": ("map-info-series", "ITMAPC"),
    "map-rol-series": ("map-infrol-serie", "ITMAPS"),
    "map-infrol-serie": ("map-infrol-serie", "ITMAPS"),
    "map-sed-series": ("map-infsed-serie", "ITMAPS"),
    "map-infsed-serie": ("map-infsed-serie", "ITMAPS"),
    "map-avg-series": ("map-infavg-serie", "ITAVGS"),
    "map-infavg-serie": ("map-infavg-serie", "ITAVGS"),
    "his-series": ("his-info-series", "ITHISC"),
    "his-info-series": ("his-info-series", "ITHISC"),
    "his-sed-series": ("his-info-series", "ITHISC"),
    "his-infsed-serie": ("his-info-series", "ITHISC"),
}
_MORPHO = {"map": ("map-infsed-serie", "MORFT"), "his": ("his-infsed-serie", "MORFT")}


def _nf(data):
    """NefisFile subyacente de un TrimFile (o el propio NefisFile)."""
    reader = getattr(data, "_reader", None)
    return reader.raw_reader if reader is not None else data


def _const_group(nf):
    for g in ("map-const", "his-const"):
        if g in nf._grp_dat:
            return g
    raise KeyError("el archivo no tiene grupo 'map-const' ni 'his-const'")


def _float32_decimal(x):
    """Decimal más corto que representa el float32 `x` (0.2f → 0.2)."""
    return float(str(np.float32(x)))


def reference_date(data):
    """Fecha de referencia ITDATE como `numpy.datetime64` (resolución s)."""
    nf = _nf(data)
    itdate = np.asarray(nf.read(_const_group(nf), "ITDATE", 1)).ravel()
    ymd = int(itdate[0])
    hms = int(itdate[1]) if itdate.size > 1 else 0
    y, mo, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    h, mi, s = hms // 10000, (hms // 100) % 100, hms % 100
    return (np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}", "s")
            + np.timedelta64(h * 3600 + mi * 60 + s, "s"))


def _read_series(nf, group, element):
    n = nf.group_info(group)["max_index"]
    return np.array([np.asarray(nf.view(group, element, i)).ravel()[0]
                     for i in range(1, n + 1)])


def get_time(data, group="map-series", kind="hydro", as_seconds=False):
    """Tiempos de los registros de `group`.

    Parámetros
    ----------
    data: TrimFile (de `d3d.read`) o NefisFile.
    group: grupo de datos cuyos registros se quieren fechar
        ('map-series', 'map-sed-series', 'map-avg-series', 'his-series', ...).
        Grupos de un solo registro sin contador (p. ej. 'map-const') → ITDATE.
    kind: 'hydro' (contador·DT·TUNIT) o 'morpho' (MORFT, días).
    as_seconds: si True devuelve segundos (float64) desde ITDATE en lugar
        de fechas.

    Retorna
    -------
    ndarray de `datetime64[ms]` (o float64 en segundos), uno por registro.
    """
    nf = _nf(data)
    if group not in nf._grp_dat:
        raise KeyError(f"grupo '{group}' no encontrado")
    t0 = reference_date(nf)
    n = nf.group_info(group)["max_index"]

    if kind == "hydro":
        if group in TIME_COUNTERS:
            cgroup, counter = TIME_COUNTERS[group]
            if cgroup not in nf._grp_dat:
                cgroup = group                      # como MATLAB: buscar el contador en el propio grupo
        elif n == 1:
            cgroup = None                           # "ONE FIELD": la fecha de referencia
        else:
            raise KeyError(f"no se conoce el contador de tiempo del grupo '{group}'")
        if cgroup is None:
            seconds = np.zeros(1)
        else:
            const = _const_group(nf)
            dt = _float32_decimal(np.asarray(nf.read(const, "DT", 1)).ravel()[0])
            tunit = _float32_decimal(np.asarray(nf.read(const, "TUNIT", 1)).ravel()[0])
            seconds = _read_series(nf, cgroup, counter).astype(np.float64) * dt * tunit
    elif kind == "morpho":
        mgroup, elm = _MORPHO["his" if group.startswith("his") else "map"]
        if mgroup not in nf._grp_dat:
            raise KeyError(f"el archivo no tiene tiempo morfológico ({mgroup}/{elm})")
        seconds = _read_series(nf, mgroup, elm).astype(np.float64) * 86400.0
    else:
        raise ValueError("kind debe ser 'hydro' o 'morpho'")

    if len(seconds) != n and not (n == 1 and len(seconds) == 1):
        raise IOError(f"el contador de tiempo tiene {len(seconds)} registros y el "
                      f"grupo '{group}' {n}")
    if as_seconds:
        return seconds
    ms = np.round(seconds * 1000.0).astype("int64")
    return t0.astype("datetime64[ms]") + ms.astype("timedelta64[ms]")
