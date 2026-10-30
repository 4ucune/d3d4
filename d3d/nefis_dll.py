import mmap
import numpy as np
from pathlib import Path
import struct

__all__ = ["NefisFile"]


class NefisFile:
    """Pure Python NEFIS reader - parses .def and .dat files."""

    # .def ELEMENT record (NEFIS5, direcciones de 8 bytes), offsets relativos
    # a `pos` = inicio del registro + 8 (ver vs_use.m):
    # +8: code '       1'(8), +16: name(16), +32: type(8),
    # +40: SizeElm (uint64, bytes totales del elemento),
    # +48: SizeVal (uint32, bytes de un valor),
    # +52: quantity(16), +68: unit(16), +84: description(64),
    # +148: ndim(uint32), +152: dims(5 x uint32)
    #
    # NEFIS5 .dat group data record offsets (8-byte addressing):
    # +0: Link(8), +8: Size(8), +16: Type(8)="       4"/"       5",
    # +24: Name(16), +40: DefName(16), +56: int_attr_names(80),
    # +136: int_attr_vals(20), +156: real_attr_names(80),
    # +236: real_attr_vals(20), +256: str_attr_names(80),
    # +336: str_attr_vals(80), +416: data / pointer table
    # var-dim: cell_size(8) at +416, pointer table at +424

    def __init__(self, def_path, dat_path=None):
        self._def_path = Path(def_path)
        self._dat_path = (
            Path(dat_path) if dat_path
            else self._def_path.with_suffix(".dat")
        )
        if not self._def_path.exists():
            raise FileNotFoundError(f"File not found: {self._def_path}")
        if not self._dat_path.exists():
            raise FileNotFoundError(f"File not found: {self._dat_path}")
        self._elements = {}
        self._cells = {}
        self._groups = {}
        self._grp_dat = {}       # group_name -> group data record info
        self._fid = None
        self._mmap = None        # mapa de memoria del .dat (lectura de datos)
        self._parse_all()

    def _parse_all(self):
        def_data = self._def_path.read_bytes()
        self._parse_definition_file(def_data)
        self._fid = self._dat_path.open("rb")
        self._scan_dat_groups()

    # -------- tablas hash NEFIS (ver vs_use.m) --------

    _NHASH = 997                  # entradas por tabla hash
    _NIL = 0xFFFFFFFFFFFFFFFF     # dirección vacía (8 bytes)
    _HEADER_LEN = 60 + 8          # cabecera de texto + longitud de archivo (uint64)

    @classmethod
    def _check_header(cls, header, path, kind):
        """Solo NEFIS 5 (direcciones de 8 bytes) little-endian ('L')."""
        if len(header) < 60:
            raise IOError(f"archivo NEFIS incompleto: {path}")
        txt = header[:59].decode("ascii", errors="replace").lower()
        if kind not in txt:
            raise IOError(f"no es un archivo NEFIS de {kind}: {path}")
        if b"5." not in header[:59] or header[59:60] != b"L":
            raise NotImplementedError(
                f"formato NEFIS no soportado (se requiere versión 5 little-endian, "
                f"indicador {header[59:60]!r}): {path}")

    @classmethod
    def _hash_records(cls, read_at, table_offset, file_size):
        """Offsets de todos los registros encadenados desde una tabla hash.

        `read_at(offset, n)` devuelve n bytes del archivo. Equivale a
        ReadHashTableGetNonEmptyBuckets de vs_use.m.
        """
        table = struct.unpack("<%dQ" % cls._NHASH,
                              read_at(table_offset, 8 * cls._NHASH))
        queue = [o for o in table if o != cls._NIL]
        seen = set()
        out = []
        while queue:
            off = queue.pop(0)
            if off in seen or off + 8 > file_size:
                continue
            seen.add(off)
            out.append(off)
            link = struct.unpack("<Q", read_at(off, 8))[0]
            if link != cls._NIL:
                queue.append(link)
        return sorted(out)

    # -------- .def parsing --------

    def _parse_definition_file(self, data):
        """Recorre las tablas hash de ELEMENTOS, CELDAS y GRUPOS del .def.

        Cada parser de registro recibe `pos` = inicio del registro + 8.
        """
        self._check_header(data[:60], self._def_path, "definition")
        read_at = lambda off, n: data[off:off + n]
        h = self._HEADER_LEN
        for off in self._hash_records(read_at, h, len(data)):
            self._parse_element_entry(data, off + 8)
        for off in self._hash_records(read_at, h + 8 * self._NHASH, len(data)):
            self._parse_cell_entry(data, off + 8)
        for off in self._hash_records(read_at, h + 16 * self._NHASH, len(data)):
            self._parse_group_entry(data, off + 8)

    def _parse_element_entry(self, data, pos):
        if pos + 172 > len(data):
            return
        name = data[pos+16:pos+32].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        if not name:
            return
        raw_type = data[pos+32:pos+40].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        typ = raw_type if raw_type else "INTEGER"

        max_bytes = struct.unpack_from("<Q", data, pos + 40)[0]     # SizeElm
        single_bytes = struct.unpack_from("<I", data, pos + 48)[0]  # SizeVal
        spb = single_bytes if single_bytes > 0 else 4

        raw_qty = data[pos+52:pos+68].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        raw_unit = data[pos+68:pos+84].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        raw_desc = data[pos+84:pos+148].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )

        ndim = struct.unpack_from("<i", data, pos + 148)[0]
        dims = []
        for i in range(min(ndim, 5)):
            d = struct.unpack_from("<i", data, pos + 152 + i*4)[0]
            dims.append(d)

        # Map type to numpy dtype
        tu = typ.upper()
        valid_types = {"INTEGER": 3, "REAL": 5, "REAL4": 5, "REAL8": 5,
                       "DOUBLE": 5, "CHARACTE": 1, "COMPLEX": 2, "LOGICAL": 4}
        type_code = valid_types.get(tu, 5 if spb in (4, 8) else 3)

        # El tamaño de un valor (SizeVal) decide la precisión: un elemento
        # "REAL" puede ser REAL*4 o REAL*8 (p. ej. DT, XCOR en trim-*.def).
        _int_dt = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
        if "REAL8" in tu or "DOUBLE" in tu:
            dtype = np.float64; spb_std = 8
        elif "REAL" in tu:
            if spb == 8:
                dtype = np.float64; spb_std = 8
            else:
                dtype = np.float32; spb_std = 4
        elif "INT" in tu or "INTEGER" in tu:
            dtype = _int_dt.get(spb, np.int32); spb_std = np.dtype(dtype).itemsize
        elif "CHAR" in tu:
            spb_std = spb if spb > 0 else 1
            dtype = np.dtype(('S', spb_std))
        elif "COMPLEX" in tu:
            if spb == 16:
                dtype = np.complex128; spb_std = 16
            else:
                dtype = np.complex64; spb_std = 8
        elif "LOGICAL" in tu:
            dtype = _int_dt.get(spb, np.int32); spb_std = np.dtype(dtype).itemsize
        else:
            spb_std = spb if spb > 0 else 4
            dtype = np.float64 if spb_std == 8 else np.float32

        spb_actual = single_bytes if single_bytes > 0 else spb_std

        self._elements[name] = {
            "name": name,
            "type": typ,
            "type_code": type_code,
            "dtype": dtype,
            "single_bytes": spb_actual,
            "num_bytes": max_bytes,
            "quantity": raw_qty,
            "unit": raw_unit,
            "description": raw_desc,
            "dimensions": dims,
            "ndim": ndim,
        }

    def _parse_cell_entry(self, data, pos):
        if pos + 56 > len(data):
            return
        name = data[pos+16:pos+32].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        if not name:
            return
        cel_dimens = struct.unpack_from("<i", data, pos + 40)[0]
        if not (0 < cel_dimens <= 500):
            cel_dimens = 0
        elems = []
        for i in range(cel_dimens):
            epos = pos + 44 + i * 16
            if epos + 16 > len(data):
                break
            ename = data[epos:epos+16].rstrip(b"\x00 ").decode(
                "ascii", errors="replace"
            )
            if ename:
                elems.append(ename)
        self._cells[name] = {"name": name, "elements": elems}

    def _parse_group_entry(self, data, pos):
        if pos + 92 > len(data):
            return
        name = data[pos+16:pos+32].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        if not name:
            return
        cell_name = data[pos+32:pos+48].rstrip(b"\x00 ").decode(
            "ascii", errors="replace"
        )
        ndim = struct.unpack_from("<i", data, pos + 48)[0]
        if not (0 <= ndim <= 5):
            ndim = 0
        dims = []
        for i in range(min(ndim, 5)):
            d = struct.unpack_from("<i", data, pos + 52 + i*4)[0]
            dims.append(d)
        orders = []
        for i in range(min(ndim, 5)):
            o = struct.unpack_from("<i", data, pos + 72 + i*4)[0]
            orders.append(o)
        self._groups[name] = {
            "name": name,
            "cell": cell_name,
            "dimensions": dims,
            "ndim": ndim,
            "order": orders,
        }

    # -------- .dat file parsing (NEFIS5, 8-byte addressing) --------

    def _scan_dat_groups(self):
        """Lee la tabla hash de GRUPOS DE DATOS del .dat (ver vs_use.m).

        Registro: +0 Link(8), +8 Size(8), +16 código '       4' (dimensión
        fija) o '       5' (dimensión variable), +24 nombre(16), +40 nombre de
        la definición(16). Los datos (o la tabla de punteros) empiezan en
        +392+3*8 = +416.
        """
        fid = self._fid
        fid.seek(0, 2)
        file_size = fid.tell()
        fid.seek(0)
        self._check_header(fid.read(60), self._dat_path, "data")

        def read_at(off, n):
            fid.seek(off)
            return fid.read(n)

        for off in self._hash_records(read_at, self._HEADER_LEN, file_size):
            rec = read_at(off, 56)
            if len(rec) < 56:
                continue
            rectype = rec[16:24]
            if rectype not in (b"       4", b"       5"):
                continue
            name = rec[24:40].rstrip(b"\x00 ").decode("ascii", errors="replace")
            if not name:
                continue
            self._grp_dat[name] = {
                "offset": off,
                "size": struct.unpack_from("<Q", rec, 8)[0],
                "var_dim": rectype == b"       5",
                "data_offset": off + 392 + 24,   # +416
                "name": name,
                "def_name": rec[40:56].rstrip(b"\x00 ").decode("ascii", errors="replace"),
            }
        fid.seek(0)

    # -------- disposición de los grupos (ver vs_let.m) --------

    def _group_def(self, group_name):
        """Definición del grupo de datos `group_name` (vía su nombre de definición)."""
        grp_dat = self._grp_dat.get(group_name)
        if grp_dat is None:
            raise KeyError(f"group '{group_name}' not found in .dat")
        gdef = self._groups.get(grp_dat.get("def_name") or group_name) \
            or self._groups.get(group_name)
        if gdef is None:
            raise IOError(f"group '{group_name}' has no definition in .def")
        return grp_dat, gdef

    def _layout(self, group_name):
        """Geometría de un grupo de datos, en caché.

        - `shape`: tamaño de cada dimensión del grupo (orden de la definición),
          con la dimensión variable resuelta al último índice escrito.
        - `vd`: posición de la dimensión variable (o None).
        - `strides`: salto en celdas de cada dimensión dentro de un bloque.
          Como vs_let.m: las dimensiones se ordenan por OrderDim (la variable
          va al final salvo que su orden sea 1) y la variable cuenta como
          tamaño 1, porque cada uno de sus índices tiene su propio bloque.
        - `cell_bytes`: suma de SizeElm de los elementos de la celda
          (NBytesPerCell de vs_let.m).
        - `elm_offsets`: desplazamiento de cada elemento dentro de la celda.
        """
        cache = self.__dict__.setdefault("_layout_cache", {})
        if group_name in cache:
            return cache[group_name]
        grp_dat, gdef = self._group_def(group_name)
        cell = self._cells.get(gdef["cell"])
        if cell is None:
            raise IOError(f"cell '{gdef['cell']}' of group '{group_name}' not defined")

        dims = list(gdef.get("dimensions") or [1])
        order = list(gdef.get("order") or [])
        if len(order) != len(dims) or sorted(order) != list(range(1, len(dims) + 1)):
            order = list(range(1, len(dims) + 1))     # orden ausente/incoherente: el natural
        vd = None
        if grp_dat["var_dim"]:
            zeros = [k for k, d in enumerate(dims) if d == 0]
            if len(zeros) != 1:
                raise IOError(f"group '{group_name}': variable dimension not found "
                              f"or not unique (dims={dims})")
            vd = zeros[0]
            dims[vd] = self._pointer_table_max_index(grp_dat)

        key = list(order)
        if vd is not None and key[vd] != 1:
            key[vd] = max(key) + 1
        strides = [0] * len(dims)
        step = 1
        for k in sorted(range(len(dims)), key=lambda k: key[k]):
            strides[k] = step
            step *= 1 if k == vd else dims[k]

        elm_offsets = {}
        off = 0
        for name in cell["elements"]:
            elm_offsets.setdefault(name, off)          # duplicado: el primero (vs_let)
            el = self._elements.get(name)
            if el is None:
                raise IOError(f"element '{name}' of cell '{gdef['cell']}' not defined")
            off += el["num_bytes"]

        lay = {"shape": dims, "order": order, "vd": vd, "strides": strides,
               "cell_bytes": off, "elm_offsets": elm_offsets,
               "def_name": grp_dat.get("def_name") or group_name, "cell": gdef["cell"]}
        cache[group_name] = lay
        return lay

    # -------- tablas de punteros (grupos de dimensión variable) --------

    _PTR_CACHE_MAX = 4096          # tablas de 2 KB → máx. ~8 MB

    def _ptr_table(self, offset):
        """Tabla de 256 direcciones en `offset`, con caché (las tablas no cambian)."""
        cache = self.__dict__.setdefault("_ptr_cache", {})
        tab = cache.get(offset)
        if tab is None:
            mm = self._data_map()
            if offset + 256 * 8 > len(mm):
                raise IOError(f"pointer table truncated at offset {offset}")
            tab = struct.unpack_from("<256Q", mm, offset)
            if len(cache) >= self._PTR_CACHE_MAX:
                cache.clear()
            cache[offset] = tab
        return tab

    def _resolve_pointer_table(self, grp_dat, index_1based):
        """Dirección del bloque del índice variable `index_1based` (0 si no existe).

        El índice se descompone en bytes b3 b2 b1 b0 (vs_let.m: VDByte); la
        tabla de nivel 3 está en +424 del registro del grupo y cada nivel
        apunta a la tabla siguiente; la entrada b0 del nivel 0 es el bloque.
        """
        if not 0 < index_1based < 2 ** 32:
            return 0
        b = [(index_1based >> (8 * k)) & 0xFF for k in range(4)]
        off = grp_dat["data_offset"] + 8
        for level in (3, 2, 1):
            off = self._ptr_table(off)[b[level]]
            if off == self._NIL:
                return 0
        off = self._ptr_table(off)[b[0]]
        return 0 if off == self._NIL else off

    def _pointer_table_max_index(self, grp_dat):
        """Último índice escrito (1-based) de un grupo variable; 0 si ninguno.

        Igual que vs_use.m: en cada nivel se sigue la última entrada no vacía.
        """
        off = grp_dat["data_offset"] + 8
        index = 0
        for level in (3, 2, 1, 0):
            tab = self._ptr_table(off)
            used = [k for k, v in enumerate(tab) if v != self._NIL]
            if not used:
                return 0
            index += used[-1] << (8 * level)
            off = tab[used[-1]]
        return index

    # -------- lectura --------

    def _normalize_index(self, group_name, lay, index):
        """Índice 1-based → tupla con un índice por dimensión del grupo."""
        nd = len(lay["shape"])
        if isinstance(index, (int, np.integer)):
            if nd != 1:
                raise ValueError(
                    f"group '{group_name}' has {nd} dimensions {lay['shape']}: "
                    f"pass a tuple of {nd} 1-based indices")
            idx = (int(index),)
        else:
            idx = tuple(int(i) for i in index)
            if len(idx) != nd:
                raise ValueError(f"group '{group_name}' expects {nd} indices, got {len(idx)}")
        for k, (i, n) in enumerate(zip(idx, lay["shape"])):
            if not 1 <= i <= n:
                raise IndexError(f"index {idx} out of range for group '{group_name}' "
                                 f"with shape {lay['shape']} (1-based)")
        return idx

    def _data_map(self):
        """mmap de solo lectura del .dat, creado al primer uso.

        Permite leer sin `seek` compartido (seguro entre hilos) y exponer
        vistas sin copia: el sistema operativo solo lee del disco las páginas
        que se tocan. El tamaño queda fijado al abrir el mapa.
        """
        if self._mmap is None:
            with self._dat_path.open("rb") as f:
                self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def view(self, group_name, element_name, index=1):
        """Como `read`, pero devuelve una VISTA de solo lectura sobre el archivo
        (sin copiar). Indexarla lee solo las páginas necesarias.

        La vista depende del archivo abierto: no usarla después de `close()`.
        Para conservar datos, copiar (`np.array(v)`) o usar `read`.
        """
        el = self._elements.get(element_name)
        if el is None:
            raise KeyError(f"element '{element_name}' not found")
        grp_dat = self._grp_dat.get(group_name)
        if grp_dat is None:
            raise IOError(f"group data '{group_name}' not found in .dat")
        lay = self._layout(group_name)
        if element_name not in lay["elm_offsets"]:
            raise IOError(f"element '{element_name}' not in group '{group_name}'")

        idx = self._normalize_index(group_name, lay, index)
        vd = lay["vd"]
        cell = sum((i - 1) * st for k, (i, st) in enumerate(zip(idx, lay["strides"]))
                   if k != vd)
        if vd is not None:
            base = self._resolve_pointer_table(grp_dat, idx[vd])
            if base == 0:
                raise IndexError(f"index {idx} not written in group '{group_name}'")
        else:
            base = grp_dat["data_offset"]

        offset = base + cell * lay["cell_bytes"] + lay["elm_offsets"][element_name]
        elem_size = el["num_bytes"]
        mm = self._data_map()
        if offset + elem_size > len(mm):
            raise IndexError(f"index {idx} out of range for '{element_name}' "
                             f"(file truncated?)")

        dtype = np.dtype(el["dtype"])
        count = elem_size // dtype.itemsize
        shape = [count]
        dims = el["dimensions"]
        if dims:
            prod = int(np.prod(dims))
            if prod and count % prod == 0:
                extra = count // prod
                shape = [extra] + list(dims) if extra > 1 else list(dims)
        return np.ndarray(shape, dtype=dtype, buffer=mm, offset=offset, order="F")

    def read(self, group_name, element_name, index=1):
        """Lee un elemento de una celda del grupo (copia propia y escribible).

        `index` es 1-based (convención de vs_let.m): un entero para grupos de
        una dimensión, o una tupla con un índice por dimensión del grupo, en
        el orden de su definición. Devuelve un arreglo con las dimensiones
        del elemento en orden Fortran (como los almacena NEFIS).
        """
        return np.array(self.view(group_name, element_name, index), order="K")

    # -------- Public API --------

    def groups(self):
        """[(nombre del grupo de datos, nº de índices de su 1.ª dim. variable o fija)]."""
        out = []
        for name in self._grp_dat:
            try:
                info = self.group_info(name)
            except (KeyError, IOError):
                continue                       # grupo sin definición utilizable
            out.append((name, info["max_index"]))
        return out

    def group_info(self, group_name):
        lay = self._layout(group_name)
        g = self._groups[lay["def_name"]]
        vd = lay["vd"]
        return {
            "cell": lay["cell"],
            "max_index": lay["shape"][vd if vd is not None else 0],
            "shape": list(lay["shape"]),
            "var_dim": vd,
            "order": lay["order"],
            "dim_count": len(lay["shape"]),
            "dimensions": list(g.get("dimensions", [])),
            "def_name": lay["def_name"],
        }

    def cells(self):
        return [(n, c["elements"]) for n, c in self._cells.items()]

    def elements(self):
        seen = set()
        out = []
        for e in self._elements.values():
            if e["name"] not in seen:
                seen.add(e["name"])
                out.append(e)
        return out

    @property
    def dat_path(self):
        return self._dat_path

    def close(self):
        if self._fid is not None:
            self._fid.close()
            self._fid = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # aún hay vistas (`view`) vivas: el mapa se libera al
                # desaparecer la última referencia
                pass
            self._mmap = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
