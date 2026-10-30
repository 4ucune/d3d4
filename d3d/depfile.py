"""Lectura de archivos de profundidad `.dep` de Delft3D.

Basado en `d3d_grd_dep/delft3d/DepFile.py`: el `.dep` es ASCII plano
con `(n+1)*(m+1)` valores en formato `E` (12 por fila) y una fila y
columna extra de relleno (`-999`). Se devuelve el array recortado
a `(n, m)`, con el valor faltante convertido a `NaN`.
"""
from pathlib import Path
import re
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["read_dep"]


def _mn_from_grd(grd_path: Union[str, Path]) -> Tuple[int, int]:
    """Lee `(m, n)` del encabezado de un `.grd` de Delft3D."""
    text = Path(grd_path).read_text(encoding="utf-8", errors="ignore")
    mn = re.search(r"\n\s+([\d]+)\s+([\d]+)\n", text)
    if mn is None:
        raise ValueError(f"no se pudo leer MN del .grd: {grd_path}")
    return int(mn.group(1)), int(mn.group(2))


def read_dep(dep_path: Union[str, Path],
             grd_path: Optional[Union[str, Path]] = None,
             m: Optional[int] = None,
             n: Optional[int] = None,
             missing: float = -999.0,
             nan_missing: bool = True) -> np.ndarray:
    """Carga un archivo `.dep` como `ndarray` de forma `(n, m)`.

    Parámetros
    ----------
    dep_path: ruta del archivo `.dep`.
    grd_path: ruta del `.grd` asociado (para obtener `m, n`). Si no
        se da ni `grd_path` ni `m, n`, se busca un `.grd` con el
        mismo nombre junto al `.dep`.
    m, n: dimensiones de la malla (alternativa a `grd_path`).
    missing: valor de relleno/faltante en el archivo (por defecto
        `-999.0`).
    nan_missing: si True (por defecto), convierte `missing` → `NaN`.

    Retorna
    -------
    ndarray float64 de forma `(n, m)` (misma orientación que
    `DepFile.data`). Para comparar con salidas corregidas de
    `d3d.get_var` (forma `(M, N)`), usar `arr.T`.
    """
    if m is None or n is None:
        if grd_path is None:
            candidato = Path(dep_path).with_suffix(".grd")
            if candidato.exists():
                grd_path = candidato
            else:
                raise ValueError(
                    "indica m y n o la ruta del .grd asociado")
        m, n = _mn_from_grd(grd_path)

    # Delft3D parte cada fila de la malla en líneas de hasta 12 valores (la
    # última más corta): se leen todos los números sin mirar las líneas.
    a = np.array(Path(dep_path).read_text(encoding="utf-8", errors="ignore").split(),
                 dtype=np.float64)
    esperado = (m + 1) * (n + 1)
    if a.size != esperado:
        raise ValueError(
            f"el .dep tiene {a.size} valores, se esperaban {esperado} "
            f"para m={m}, n={n}")
    a = a.reshape(n + 1, m + 1)[:-1, :-1]   # quita fila/col de relleno
    if nan_missing:
        a = np.array(a, copy=True)
        a[a == missing] = np.nan
    return a
