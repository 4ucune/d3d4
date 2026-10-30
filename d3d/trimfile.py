from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
from .reader import NEFISReader
from .models import NEFISGroupDef, NEFISCell
from . import fields

logger = logging.getLogger(__name__)


class _Raw:
    """`v.raw[...]`: lectura sin máscara, tal como está en el archivo."""

    def __init__(self, var: "Variable") -> None:
        self._var = var

    def __getitem__(self, key) -> np.ndarray:
        return self._var._read(key)[0]


class Variable:
    def __init__(self, reader: NEFISReader, group: str, name: str, element_info,
                 owner: Optional["TrimFile"] = None) -> None:
        self._reader = reader
        self._group = group
        self._name = name
        self._info = element_info
        self._owner = owner
        gd = reader.groups.get(group)
        self._ntimes = gd.max_index if gd else 0
        self._location = None
        if owner is not None:
            dtype = reader.raw_reader._elements[name]["dtype"]
            self._location = fields.location(name, self.shape[1:], dtype, owner.grid_shape)

    @property
    def name(self) -> str:
        return self._name

    @property
    def group(self) -> str:
        return self._group

    @property
    def shape(self) -> Tuple[int, ...]:
        """Forma de `v[:]`: (registros,) + dimensiones del elemento en el
        orden NEFIS, p. ej. (T, N, M) o (T, N, M, K). Las variables
        constantes tienen un solo registro: (1, N, M).
        """
        return (self._ntimes,) + tuple(int(d) for d in self._info.dimensions)

    @property
    def full_shape(self) -> Tuple[int, ...]:
        """Igual que `shape` (se conserva por compatibilidad)."""
        return self.shape

    @property
    def ntimes(self) -> int:
        return self._ntimes

    @property
    def location(self) -> Optional[str]:
        """Posición en la malla que decide la máscara al leer: 'z' (centro),
        'u'/'v' (caras), 'd' (esquinas) o None (sin máscara)."""
        return self._location

    @property
    def raw(self) -> _Raw:
        """Lectura cruda (sin máscara): `v.raw[t]`, `v.raw[:, n, m]`, ..."""
        return _Raw(self)

    @property
    def attributes(self) -> Dict[str, str]:
        raw_desc = self._info.description.strip() if self._info.description else ''
        raw_q = self._info.quantity.strip() if self._info.quantity else ''
        return {
            'units': self._info.unit.strip() if self._info.unit else '',
            'quantity': raw_q,
            'type': self._info.type.strip(),
            'name': raw_desc or (raw_q if raw_q.isprintable() else '') or self._name,
            'description': raw_desc,
        }

    def _time_indices(self, key) -> Tuple[List[int], bool]:
        """Selector de tiempo → (lista de registros 0-based, ¿escalar?)."""
        nt = self._ntimes
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            k = int(key)
            if k < 0:
                k += nt
            if k < 0 or k >= nt:
                raise IndexError(f"indice fuera de rango: {key}")
            return [k], True
        if isinstance(key, slice):
            return list(range(*key.indices(nt))), False
        if key is Ellipsis:
            return list(range(nt)), False
        arr = np.asarray(key)
        if arr.dtype == bool:
            if arr.shape != (nt,):
                raise IndexError(f"máscara de tiempo de forma {arr.shape}, se esperaba ({nt},)")
            return [int(k) for k in np.flatnonzero(arr)], False
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            out = []
            for k in arr.tolist():
                kk = k + nt if k < 0 else k
                if kk < 0 or kk >= nt:
                    raise IndexError(f"indice fuera de rango: {k}")
                out.append(kk)
            return out, False
        raise TypeError(f"indice de tiempo no soportado: {type(key).__name__}")

    def __getitem__(self, key) -> np.ndarray:
        """Lectura selectiva con sintaxis NumPy: `v[t]`, `v[t0:t1]`,
        `v[[t1, t5]]`, `v[:, n, m]`, `v[10, :, 3:8, 0]`, `v[..., k]`.

        El primer índice selecciona registros (0-based); el resto se aplica
        a las dimensiones de cada registro en el orden en que las devuelve
        NEFIS (p. ej. (N, M) o (N, M, K)). Solo se leen los registros
        pedidos y, dentro de cada uno, las páginas del archivo que toca la
        selección. Devuelve siempre un arreglo nuevo y escribible.

        Aplica la máscara según la posición de la variable (`location`; ver
        `d3d.fields.mask_read`). Sin máscara: `v.raw[...]`.
        """
        out, records, scalar, rest = self._read(key)
        if self._location is not None:
            fields.mask_read(self._owner, self, records, scalar, rest, out)
        return out

    def _read(self, key):
        """Lectura cruda → (arreglo, registros, ¿escalar?, índice del registro)."""
        if not isinstance(key, tuple):
            key = (key,)
        if not key:
            key = (slice(None),)
        tkey, rest = key[0], key[1:]
        if tkey is Ellipsis:
            tkey, rest = slice(None), (Ellipsis,) + rest
        records, scalar = self._time_indices(tkey)

        def leer(t):
            v = self._reader.view_element(self._group, self._name, t)
            return v[rest] if rest else v

        if scalar:
            return np.array(leer(records[0]), order="K"), records, True, rest
        if not records:
            dtype = self._reader.raw_reader._elements[self._name]["dtype"]
            rec = np.broadcast_to(np.zeros((), dtype=dtype), self.shape[1:])
            return np.empty((0,) + np.shape(rec[rest] if rest else rec), dtype=dtype), records, False, rest
        first = leer(records[0])
        out = np.empty((len(records),) + np.shape(first), dtype=first.dtype)
        out[0] = first
        for i, t in enumerate(records[1:], start=1):
            out[i] = leer(t)
        return out, records, False, rest


class TrimFile:
    _GROUP_PRIORITY = [
        'map-series', 'map-const', 'map-info-series',
        'map-sed-series', 'map-infsed-serie',
        'map-avg-series', 'map-infavg-serie', 'map-version',
    ]

    def __init__(self, def_path: str, dat_path: Optional[str] = None) -> None:
        self._reader = NEFISReader(def_path, dat_path)
        self._ds: Dict[str, Variable] = {}
        self._probe_elements()

    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """(N, M) de la malla según KCS; None si no hay KCS (p. ej. trih)."""
        el = self._reader.elements.get("KCS")
        if el is None or len(el.dimensions) < 2:
            return None
        return int(el.dimensions[0]), int(el.dimensions[1])

    def _probe_elements(self) -> None:
        group_by_element = self._group_by_element()
        for name, el in self._reader.elements.items():
            try:
                grp = group_by_element.get(name) or self._find_group(name)
                if grp:
                    self._ds[name] = Variable(self._reader, grp, name, el, owner=self)
                else:
                    logger.debug("elemento '%s' no asignado a ningun grupo", name)
            except Exception as exc:
                logger.warning("error al mapear elemento '%s': %s", name, exc)

    def _group_by_element(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        groups = self._reader.groups
        ordered: List[str] = [g for g in self._GROUP_PRIORITY if g in groups]
        ordered.extend(g for g in groups if g not in ordered)
        for group_name in ordered:
            cell = self._reader.cells.get(group_name)
            if cell is None:
                cell_name = groups[group_name].cell
                cell = self._reader.cells.get(cell_name)
            if cell is None:
                continue
            for element_name in cell.elements:
                out.setdefault(element_name, group_name)
        return out

    def _find_group(self, elm_name: str) -> Optional[str]:
        for g in self._GROUP_PRIORITY:
            if g in self._reader.groups:
                try:
                    self._reader._nf.read(g, elm_name, 1)
                    return g
                except Exception:
                    pass
        for g in self._reader.groups:
            try:
                self._reader._nf.read(g, elm_name, 1)
                return g
            except Exception:
                pass
        return None

    @property
    def groups(self) -> Dict[str, NEFISGroupDef]:
        return self._reader.groups

    @property
    def variables(self) -> Dict[str, Variable]:
        return dict(self._ds)

    @property
    def cells(self) -> Dict[str, NEFISCell]:
        return self._reader.cells

    def keys(self) -> List[str]:
        return list(self._ds.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._ds

    def __getitem__(self, name: str) -> Variable:
        return self._ds[name]

    def read(self, name: str, index=None, mask: bool = True) -> np.ndarray:
        v = self._ds[name]
        reader = v if mask else v.raw
        return reader[:] if index is None else reader[index]

    def get_var(self, name: str, index=None, mask: bool = True) -> np.ndarray:
        return self.read(name, index=index, mask=mask)

    def info(self, name: Optional[str] = None) -> None:
        """Catálogo de lo que se puede extraer del archivo.

        Sin `name`: variables NEFIS por grupo (forma, posición de la máscara,
        unidades), campos escalares y vectoriales de QUICKPLOT disponibles y
        funciones de d3d aplicables. Con `name`: detalle de una variable
        NEFIS, de un campo de QUICKPLOT o de una derivada.
        """
        if name is None:
            self._info_all()
            return
        if name in self._ds:
            v = self._ds[name]
            print(f"{name}:  variable NEFIS")
            print(f"  nombre: {v.attributes['name']}")
            print(f"  grupo:  {v.group}")
            print(f"  forma:  {v.shape}")
            print(f"  pasos:  {v.ntimes}")
            print(f"  posición: {v.location or 'sin máscara'}")
            print(f"  unidad: {v.attributes['units']}")
            print(f"  tipo:   {v.attributes['type']}")
            return
        if name in fields.DERIVED:
            us, vs, kind = fields.DERIVED[name]
            faltan = [n for n in us + vs if n not in self._ds]
            if faltan:
                print(f"'{name}': faltan {faltan} en este archivo")
                return
            print(f"{name}:  campo vectorial (QUICKPLOT) -> (X, Y, magnitud)")
            print(f"  fuentes:  U = {' + '.join(us)}, V = {' + '.join(vs)}")
            print("  cálculo:  " + ("uv2cen + cur2ca (ALFAS) + máscara KCS, en centros"
                                   if kind == "centro" else
                                   "caras sin interpolar ni rotar, máscara de caras (conv2 2x2)"))
            print(f"  forma:    {fields.derived_shape(self, name)} por componente")
            print(f"  unidad:   {self._ds[us[0]].attributes['units']}")
            print(f"  uso:      d3d.get_var(data, {name!r}, index, component=None|'X'|'Y'|'mag')")
            return
        hit = fields.find_scalar(self, name)
        if hit is not None:
            nm, el, j = hit
            v = self._ds[el]
            src = el if j is None else f"{el}[..., {j}]"
            print(f"{nm}:  campo escalar (QUICKPLOT)")
            print(f"  fuente:   {src}  ({v.attributes['name']})")
            print(f"  máscara:  la de {el} (posición {v.location or '-'})")
            print(f"  forma:    {fields.scalar_shape(self, nm)}")
            print(f"  unidad:   {v.attributes['units']}")
            print(f"  uso:      d3d.get_var(data, {nm!r}, index)")
            return
        print(f"Variable '{name}' no encontrada")

    def _info_all(self) -> None:
        def fila(nombre, forma, extra, desc, ancho=34):
            print(f"  {nombre:{ancho}s} {str(forma):22s} {extra:2s} {desc}")

        derivadas = [d for d, (us, vs, _) in fields.DERIVED.items()
                     if all(n in self._ds for n in us + vs)]
        escalares = fields.scalars(self)
        print(f"Archivo: {self._reader.def_file()}")
        print(f"Grupos: {len(self.groups)}   Variables NEFIS: {len(self._ds)}   "
              f"Campos QUICKPLOT: {len(escalares)} escalares, {len(derivadas)} vectoriales")
        print()
        print("=== Variables NEFIS ===   d3d.get_var(data, nombre, index, mask=True)")
        print("    posición (máscara al leer): z centro, u/v cara, d esquina, - sin máscara")
        by_group: dict = {}
        for vname, v in self._ds.items():
            by_group.setdefault(v.group, []).append((vname, v))
        for gname in self.groups:
            print(f"[{gname}]")
            for vname, v in sorted(by_group.get(gname, []), key=lambda x: x[0]):
                fila(vname, v.shape, v.location or "-",
                     f"[{v.attributes['units'].strip('[] ')}]  {v.attributes['name']}")
        print()
        if escalares:
            print("=== Campos escalares (nombres de QUICKPLOT) ===   d3d.get_var(data, nombre, index)")
            for nm, (el, j) in escalares.items():
                src = el if j is None else f"{el}[..., {j}]"
                units = self._ds[el].attributes['units'].strip('[] ')
                fila(nm, fields.scalar_shape(self, nm), "", f"[{units}]  = {src}")
            print()
        if derivadas:
            print("=== Campos vectoriales (nombres de QUICKPLOT) ===   "
                  "d3d.get_var(data, nombre, index, component) -> (X, Y, mag)")
            for d in derivadas:
                us, vs, kind = fields.DERIVED[d]
                units = self._ds[us[0]].attributes['units'].strip('[] ')
                calc = "centros, cartesianas" if kind == "centro" else "caras, sin rotar"
                fila(d, fields.derived_shape(self, d), "",
                     f"[{units}]  {'+'.join(us)} / {'+'.join(vs)} ({calc})")
            print()
        funcs = []
        if "S1" in self._ds:
            funcs.append(("d3d.water_level(data, index)", self._ds["S1"].shape,
                          "S1 con máscara de inactivas y secas"))
        if "U1" in self._ds and "V1" in self._ds:
            funcs.append(("d3d.velocity_xy(data, index, layer)", self._ds["U1"].shape,
                          "(ux, uy): U1/V1 en centros y cartesianas"))
        if "UMNLDF" in self._ds and "VMNLDF" in self._ds:
            funcs.append(("d3d.velocity_xy(data, index, filtered=True)", self._ds["UMNLDF"].shape,
                          "(ux, uy): UMNLDF/VMNLDF"))
        caras_u = sorted(n for n, v in self._ds.items()
                         if v.location == "u" and v.group != "map-const")
        if caras_u:
            funcs.append(("d3d.vector_xy(data, u, v, index, layer)", "",
                          "(x, y) de un par de caras U/V: " + ", ".join(caras_u)))
        for g in ("map-series", "map-sed-series", "map-avg-series"):
            if g in self.groups:
                funcs.append((f"d3d.get_time(data, {g!r})", (self.groups[g].max_index,),
                              "fechas datetime64 de los registros"))
        if funcs:
            print("=== Funciones de d3d ===")
            for f, forma, desc in funcs:
                fila(f, forma, "", desc, ancho=46)
            print()
        print("Detalle de cualquier nombre: d3d.info(data, nombre)")

    def extract_all(self, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        target = variables or [
            "S1", "U1", "V1", "W", "R1", "RHO",
            "XCOR", "YCOR", "XZ", "YZ", "DP0", "THICK",
            "ITDATE", "DT",
        ]
        out: Dict[str, np.ndarray] = {}
        for name in target:
            v = self.variables.get(name)
            if v is not None:
                try:
                    out[name.lower()] = v[:]
                except Exception:
                    out[name.lower()] = v[0]
        try:
            # segundos desde ITDATE: ITMAPC * DT * TUNIT (antes faltaba TUNIT)
            from .timeaxis import get_time
            out["time"] = get_time(self, "map-series", as_seconds=True)
        except Exception as exc:
            logger.warning("no se pudo extraer tiempo: %s", exc)
        return out

    def dump_json(self, path: Optional[str] = None) -> str:
        import json
        out = {}
        for name, v in self.variables.items():
            out[name] = {
                "group": v.group,
                "shape": list(v.shape),
                "ntimes": v.ntimes,
                "attributes": v.attributes,
            }
        text = json.dumps(out, indent=2)
        if path:
            import pathlib
            pathlib.Path(path).write_text(text, encoding="utf-8")
        return text

    def save_nc(self, path: str,
                variables: Optional[Union[Sequence[str], Dict[str, str]]] = None,
                zlib: bool = True) -> None:
        """Exporta a netCDF (se conserva la firma anterior; usa d3d.to_netcdf).

        variables: None → TODAS las variables; lista de nombres; o
        {nombre_en_netcdf: nombre_NEFIS}. Los nombres inexistentes se omiten.
        Antes las dimensiones se nombraban por tamaño (dim_<n>), lo que
        mezclaba ejes cuando M == N; ahora se usan (time, N, M, K, ...).
        """
        from .export import to_netcdf
        if variables is None:
            names, rename = sorted(self.variables), {}
        elif isinstance(variables, dict):
            names = [n for n in variables.values() if n in self.variables]
            rename = {n: a for a, n in variables.items() if n in self.variables}
        else:
            names, rename = [n for n in variables if n in self.variables], {}
        names = [n for n in names
                 if np.dtype(self._reader.raw_reader._elements[n]["dtype"]).kind != "c"]
        to_netcdf(self, path, variables=names, rename=rename, compress=zlib)

    def close(self) -> None:
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
