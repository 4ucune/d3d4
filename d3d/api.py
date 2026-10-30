from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from . import fields
from .trimfile import TrimFile

__all__ = [
    "read", "open_trim", "info", "get_var",
    "extract_all", "save_nc", "dump_json",
]


def _resolve_pair(path: str) -> tuple:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".dat":
        return p.with_suffix(".def"), p
    if suffix == ".def":
        return p, p.with_suffix(".dat")
    raise ValueError("use un archivo .dat o .def de Delft3D/NEFIS")


def read(path: str, dat_path: Optional[str] = None) -> TrimFile:
    """Abre un par NEFIS .def/.dat (trim-*, trih-*).

    path: el .dat o el .def; el otro se busca con el mismo nombre.
    dat_path: ruta explícita del .dat cuando su nombre no sigue esa
        convención (p. ej. 'trim-x.dat.exe'); `path` debe ser entonces el .def.
    """
    if dat_path is not None:
        def_path, dat_path = Path(path), Path(dat_path)
        if def_path.suffix.lower() != ".def":
            raise ValueError("con dat_path, `path` debe ser el archivo .def")
    else:
        def_path, dat_path = _resolve_pair(path)
    if not def_path.exists():
        raise FileNotFoundError(f"archivo .def no encontrado: {def_path}")
    if not dat_path.exists():
        raise FileNotFoundError(f"archivo .dat no encontrado: {dat_path}")
    return TrimFile(def_path, dat_path)


open_trim = read


def info(data: TrimFile, name: Optional[str] = None) -> None:
    data.info(name)


def get_var(data: TrimFile, name: str, index=None,
            corrected: bool = False, mask: bool = True, component=None):
    """Lee una variable del archivo NEFIS, enmascarada según su posición.

    data: TrimFile abierto con d3d.read()
    name: nombre NEFIS (p. ej. 'S1', 'U1'), de un campo escalar de QUICKPLOT
        (p. ej. 'water level', 'density' o un constituyente de NAMCON como
        'Salinity'; ver d3d.fields.scalars) o de una variable vectorial
        derivada (p. ej. 'bed load transport'; ver d3d.fields.DERIVED).
        `d3d.info(data)` lista todos los disponibles.
    index: selección. None → todos los registros. Cualquier selector de
        `Variable.__getitem__`, p. ej. 5, slice(0, 10), [0, 5, 9] o una
        tupla (tiempo, n, m[, k]) como (slice(None), 1, 20) para una serie
        temporal puntual. Índices 0-based; ejes del registro en el orden
        NEFIS (N, M[, K]). En las derivadas, solo selección de registros.
    mask: True (por defecto) → NaN donde d3d_trimfil.m no muestra datos, según
        la posición de la variable (centro, cara o esquina; S1 también celdas
        secas). False → valores tal como están en el archivo. Ver
        d3d.fields.mask_read. No aplica a las derivadas (siempre enmascaradas).
    component: solo derivadas. None → (X, Y, magnitud); 'X', 'Y' o 'mag'.
    corrected: retirado; con True lanza TypeError.
    """
    if corrected:
        raise TypeError(
            "get_var(..., corrected=True) se eliminó: usar d3d.water_level, "
            "d3d.velocity_xy o d3d.vector_xy (orden (T,N,M[,K])) y d3d.transforms.")
    if name in fields.DERIVED and name not in data.variables:
        return fields.derived(data, name, index=index, component=component)
    if component is not None:
        raise TypeError("component solo se usa con variables derivadas")
    v = data.variables.get(name)
    if v is None:
        if fields.find_scalar(data, name) is not None:
            return fields.scalar(data, name, index=index, mask=mask)
        raise KeyError(f"variable '{name}' no encontrada (ver d3d.info(data))")
    reader = v if mask else v.raw
    return reader[:] if index is None else reader[index]


def extract_all(data: TrimFile, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    return data.extract_all(variables)


def save_nc(data: TrimFile, path: str,
            variables: Optional[Union[Sequence[str], Dict[str, str]]] = None,
            zlib: bool = True) -> None:
    data.save_nc(path, variables=variables, zlib=zlib)


def dump_json(data: TrimFile, path: Optional[str] = None) -> str:
    return data.dump_json(path=path)
