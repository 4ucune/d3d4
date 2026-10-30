"""Escritor NEFIS 5 (little-endian, direcciones de 8 bytes) MÍNIMO, solo para pruebas.

Genera pares .def/.dat sintéticos con estructuras que los archivos reales de
prueba no tienen: grupos de dimensión fija con varias celdas, grupos
multidimensionales con distinto OrderDim, colisiones en las tablas hash, varios
grupos de datos con la misma definición, índices variables > 255.

El diseño de los registros sigue vs_use.m / vs_let.m:

.def: cabecera(60) + longitud(8) + 3 tablas hash de 997 direcciones
      (ELEMENTOS, CELDAS, GRUPOS) + registros:
  elemento: Link(8) Size(8) '       1' Nombre(16) Tipo(8) SizeElm(8) SizeVal(4)
            Cantidad(16) Unidad(16) Descripción(64) NDim+5 dims (6 x uint32)
  celda:    Link Size '       2' Nombre(16) CellSize(8) NElm(4) Nombres(16*n)
  grupo:    Link Size '       3' Nombre(16) Celda(16) NDim+5 dims+5 órdenes (11 x uint32)
.dat: cabecera(60) + longitud(8) + 1 tabla hash (GRUPOS DE DATOS) + registros:
  grupo de datos: Link Size '       4'|'       5' Nombre(16) Definición(16)
            + 360 bytes de atributos (= 392 + 3*8 = 416 bytes)
  fijo ('4'):     datos a partir de +416
  variable ('5'): +416 CellSize(8), +424 tabla de punteros de nivel 3 (256 x 8);
                  índice i (1-based) -> bytes b3 b2 b1 b0; niveles 3->2->1->0;
                  la entrada b0 del nivel 0 apunta al bloque de datos.

La DISPOSICIÓN de las celdas dentro de un bloque NO la decide este escritor:
la prueba entrega los bytes ya ordenados, calculados a mano. Así la prueba
no reutiliza la fórmula de offsets del lector.
"""
import struct
from pathlib import Path

import numpy as np

NHASH = 997
NIL = 0xFFFFFFFFFFFFFFFF
HDR = 60 + 8

_TYPES = {"REAL": "REAL    ", "INTEGER": "INTEGER ", "CHARACTE": "CHARACTE",
          "COMPLEX": "COMPLEX ", "LOGICAL": "LOGICAL "}


def nefis_bucket(name):
    """Cubeta 0-based según vs_use.m: mod(sum(prod(reshape(Label+1,[4 4]))),997)."""
    codes = [c + 1 for c in name.ljust(16).encode("ascii")[:16]]
    return sum(int(np.prod(codes[4 * k:4 * k + 4], dtype=object)) for k in range(4)) % NHASH


def _txt(s, n):
    b = s.encode("ascii")
    assert len(b) <= n, s
    return b.ljust(n, b" ")


class _HashedFile:
    """Archivo con tablas hash; los registros se añaden al final y se encadenan."""

    def __init__(self, header_text, ntables, bucket=nefis_bucket):
        hdr = header_text.encode("ascii").ljust(59, b" ") + b"L"
        self.buf = bytearray(hdr + b"\0" * 8)
        self.tables = [len(self.buf) + 8 * NHASH * t for t in range(ntables)]
        self.buf += struct.pack(f"<{NHASH * ntables}Q", *([NIL] * NHASH * ntables))
        self.last = {}            # (tabla, cubeta) -> offset del último registro
        self.bucket = bucket

    def add(self, table, name, record):
        off = len(self.buf)
        self.buf += record
        b = self.bucket(name)
        key = (table, b)
        if key in self.last:      # colisión: encadenar desde el último registro
            struct.pack_into("<Q", self.buf, self.last[key], off)
        else:
            struct.pack_into("<Q", self.buf, self.tables[table] + 8 * b, off)
        self.last[key] = off
        return off

    def append(self, data):
        off = len(self.buf)
        self.buf += data
        return off

    def finish(self, path):
        struct.pack_into("<Q", self.buf, 60, len(self.buf))
        Path(path).write_bytes(bytes(self.buf))


class NefisWriter:
    """Uso:
        w = NefisWriter()
        w.element("S1", "REAL", 4, [3, 4], unit="[ M ]")
        w.cell("c", ["S1"])
        w.group("g", "c", dims=[0], order=[1])          # 0 = dimensión variable
        w.data_fixed("g", "g", raw_bytes)               # grupo fijo
        w.data_var("g2", "g", {1: bytes, 2: bytes})     # grupo variable
        w.write(def_path, dat_path)
    """

    def __init__(self, bucket=nefis_bucket):
        self.elements, self.cells, self.groups, self.data = {}, {}, {}, []
        self.bucket = bucket

    def element(self, name, typ, sizeval, dims, quantity="", unit="", description=""):
        self.elements[name] = dict(typ=typ, sizeval=sizeval, dims=list(dims),
                                   quantity=quantity, unit=unit, description=description)

    def elm_bytes(self, name):
        e = self.elements[name]
        return int(np.prod(e["dims"])) * e["sizeval"]

    def cell(self, name, elements):
        self.cells[name] = list(elements)

    def cell_bytes(self, name):
        return sum(self.elm_bytes(e) for e in self.cells[name])

    def group(self, name, cell, dims, order=None):
        self.groups[name] = dict(cell=cell, dims=list(dims),
                                 order=list(order or range(1, len(dims) + 1)))

    def data_fixed(self, name, defname, raw):
        self.data.append(("4", name, defname, bytes(raw)))

    def data_var(self, name, defname, blocks):
        """blocks: {índice 1-based: bytes de un bloque}"""
        self.data.append(("5", name, defname, dict(blocks)))

    # ---------- escritura ----------

    def write(self, def_path, dat_path):
        d = _HashedFile("Deltares, NEFIS Definition File; 5.00.00", 3, self.bucket)
        for name, e in self.elements.items():
            dims = e["dims"] + [0] * (5 - len(e["dims"]))
            rec = (struct.pack("<QQ", NIL, 180) + b"       1" + _txt(name, 16)
                   + _txt(_TYPES[e["typ"]], 8)
                   + struct.pack("<QI", self.elm_bytes(name), e["sizeval"])
                   + _txt(e["quantity"], 16) + _txt(e["unit"], 16)
                   + _txt(e["description"], 64)
                   + struct.pack("<6I", len(e["dims"]), *dims))
            assert len(rec) == 180
            d.add(0, name, rec)
        for name, elms in self.cells.items():
            rec = (struct.pack("<QQ", NIL, 52 + 16 * len(elms)) + b"       2"
                   + _txt(name, 16) + struct.pack("<QI", self.cell_bytes(name), len(elms))
                   + b"".join(_txt(n, 16) for n in elms))
            d.add(1, name, rec)
        for name, g in self.groups.items():
            nd = len(g["dims"])
            rec = (struct.pack("<QQ", NIL, 100) + b"       3" + _txt(name, 16)
                   + _txt(g["cell"], 16)
                   + struct.pack("<11I", nd, *(g["dims"] + [0] * (5 - nd)),
                                 *(g["order"] + [0] * (5 - nd))))
            d.add(2, name, rec)
        d.finish(def_path)

        t = _HashedFile("Deltares, NEFIS Data File; 5.00.00", 1, self.bucket)
        for kind, name, defname, payload in self.data:
            head = struct.pack("<QQ", NIL, 0) + (b"       " + kind.encode()) \
                + _txt(name, 16) + _txt(defname, 16) + b"\0" * 360
            assert len(head) == 416
            if kind == "4":
                off = t.add(0, name, head + payload)
                struct.pack_into("<Q", t.buf, off + 8, 416 + len(payload))
            else:
                cellsize = self.cell_bytes(self.groups[defname]["cell"])
                l3 = [NIL] * 256
                off = t.add(0, name, head + struct.pack("<Q", cellsize)
                            + struct.pack("<256Q", *l3))
                struct.pack_into("<Q", t.buf, off + 8, 424 + 2048)
                tables = {(): off + 424}          # prefijo de bytes -> offset de tabla
                for idx, block in sorted(payload.items()):
                    b = [(idx >> (8 * k)) & 0xFF for k in range(4)]   # b0..b3
                    prefix = ()
                    for level in (3, 2, 1):
                        tab = tables[prefix]
                        slot = tab + 8 * b[level]
                        prefix = prefix + (b[level],)
                        if prefix not in tables:
                            tables[prefix] = t.append(struct.pack("<256Q", *([NIL] * 256)))
                            struct.pack_into("<Q", t.buf, slot, tables[prefix])
                    data_off = t.append(block)
                    struct.pack_into("<Q", t.buf, tables[prefix] + 8 * b[0], data_off)
        t.finish(dat_path)


def pack(values, dtype):
    """Bytes de un arreglo en orden Fortran (como lo escribe Delft3D)."""
    return np.asarray(values, dtype=dtype).tobytes(order="F")
