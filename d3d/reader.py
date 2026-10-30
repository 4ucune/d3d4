from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .nefis_dll import NefisFile
from .models import NEFISElement, NEFISCell, NEFISGroupDef


class NEFISReader:
    def __init__(self, def_path: str, dat_path: Optional[str] = None) -> None:
        self._nf = NefisFile(def_path, dat_path)
        self._def_path = Path(def_path)
        self._dat_path = self._nf.dat_path
        self._init_metadata()

    def _init_metadata(self):
        self._groups = {}
        for name, defined in self._nf.groups():
            info = self._nf.group_info(name)
            self._groups[name] = NEFISGroupDef(
                name=name, defined=defined, cell=info['cell'],
                dim_count=info['dim_count'], dimensions=info['dimensions'],
                order=info['order'], max_index=info['max_index'],
            )

        self._cells = {}
        for name, elements in self._nf.cells():
            self._cells[name] = NEFISCell(name=name, elements=elements)

        elm_raw = self._nf.elements()
        self._elements = {}
        for e in elm_raw:
            desc = e.get('description', '')
            self._elements[e['name']] = NEFISElement(
                name=e['name'], type=e['type'], single_bytes=e['single_bytes'],
                quantity=e['quantity'], unit=e['unit'], description=desc,
                count=len(e['dimensions']), dimensions=e['dimensions'],
            )

    @property
    def groups(self) -> Dict[str, NEFISGroupDef]:
        return dict(self._groups)

    @property
    def elements(self) -> Dict[str, NEFISElement]:
        return dict(self._elements)

    @property
    def cells(self) -> Dict[str, NEFISCell]:
        return dict(self._cells)

    def close(self) -> None:
        self._nf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def view_element(self, group: str, element: str, index: int = 0) -> np.ndarray:
        """Vista de solo lectura sin copia de un registro (`index` 0-based).

        Sin caché de datos: el sistema operativo cachea las páginas del
        archivo y `NefisFile` las tablas de punteros.
        """
        return self._nf.view(group, element, index + 1)  # 0-based → 1-based

    @property
    def raw_reader(self) -> NefisFile:
        return self._nf

    def dat_file(self) -> str:
        return str(self._dat_path)

    def def_file(self) -> str:
        return str(self._def_path)
