"""Configuración común de pruebas.

Los datos de integración (ejemplo `04_fluidmud` de Delft3D: trim-sed,
trim-mud, trih-sed, trih-mud) se buscan, en este orden, en la carpeta de la
variable de entorno `D3D_TEST_DATA`, en `tests/data/04_fluidmud` y en
`Resultados/04_fluidmud`. Si no existen, las pruebas de integración se omiten.
"""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))          # para importar _ref_nefis

_CANDIDATOS = [HERE / "data" / "04_fluidmud", ROOT / "Resultados" / "04_fluidmud"]
DATA_DIR = Path(os.environ.get("D3D_TEST_DATA") or next(
    (c for c in _CANDIDATOS if (c / "trim-sed.dat").exists()), _CANDIDATOS[0]))
NEFIS_FILES = ["trim-mud", "trim-sed", "trih-mud", "trih-sed"]


def _pair(base):
    return DATA_DIR / f"{base}.def", DATA_DIR / f"{base}.dat"


@pytest.fixture(scope="session")
def data_dir():
    if not (DATA_DIR / "trim-sed.dat").exists():
        pytest.skip(f"datos de prueba no disponibles en {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session", params=NEFIS_FILES)
def nefis_pair(request, data_dir):
    """(base, ruta .def, ruta .dat) para cada archivo NEFIS de prueba."""
    d, t = _pair(request.param)
    return request.param, d, t


@pytest.fixture(scope="session")
def trim_sed(data_dir):
    import d3d
    data = d3d.read(str(data_dir / "trim-sed.dat"))
    yield data
    data.close()


@pytest.fixture(scope="session")
def ref_trim_sed(data_dir):
    from _ref_nefis import RefNefis
    d, t = _pair("trim-sed")
    return RefNefis(d, t)
