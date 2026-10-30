"""Lector NEFIS de REFERENCIA, solo para pruebas.

Traducción directa y deliberadamente simple de `vs_use.m` (cabecera, tablas
hash de 997 entradas, registros de elemento/celda/grupo) y `vs_let.m`
(offsets de celda y tablas de punteros de grupos de dimensión variable) del
toolbox MATLAB de Delft3D. No comparte código con `d3d` para que sirva de
oráculo independiente.

Limitaciones: lee el archivo completo en memoria y solo admite grupos con una
dimensión (todos los trim/trih conocidos). No usar en producción.
"""
import struct, numpy as np
from pathlib import Path
NHASH = 997

class RefNefis:
    def __init__(self, def_path, dat_path):
        self.dat = Path(dat_path).read_bytes()
        self.dfn = Path(def_path).read_bytes()
        hdr = self.dat[:60]
        flag = chr(hdr[59])
        self.bo = '<' if flag == 'L' else '>'
        self.ab = 8 if b'5.' in hdr else 4
        self.A = 'Q' if self.ab == 8 else 'I'
        self.nil = 2**64-1 if self.ab == 8 else 2**32-1
        self.hl = 60 + self.ab
        self._groups_dat(); self._groups_def(); self._cells(); self._elements()

    def _u(self, buf, off, fmt):
        return struct.unpack_from(self.bo + fmt, buf, off)

    def _buckets(self, buf, loc):
        ht = self._u(buf, loc, f'{NHASH}{self.A}')
        q = [h for h in ht if h != self.nil]; out = []
        while q:
            o = q.pop(0); out.append(o)
            link = self._u(buf, o, self.A)[0]
            if link != self.nil: q.append(link)
        return sorted(out)

    def _s(self, buf, a, n):
        return buf[a:a+n].decode('ascii', 'replace').rstrip('\x00 ')

    def _groups_dat(self):
        ab = self.ab; self.grpdat = {}
        for o in self._buckets(self.dat, self.hl):
            size = self._u(self.dat, o+ab, self.A)[0]
            code = self.dat[o+2*ab:o+3*ab]
            p = o+3*ab
            name = self._s(self.dat, p, 16); defn = self._s(self.dat, p+16, 16)
            vardim = code[-1:] == b'5'
            if vardim:
                ds = self._u(self.dat, o+392+3*ab, self.A)[0]
            else:
                ds = size-392-3*ab
            self.grpdat[name] = dict(offset=o, vardim=vardim, defname=defn, datasize=ds)

    def _groups_def(self):
        ab = self.ab; self.grpdef = {}
        for o in self._buckets(self.dfn, self.hl+2*ab*NHASH):
            p = o+3*ab
            name = self._s(self.dfn, p, 16); cel = self._s(self.dfn, p+16, 16)
            d = self._u(self.dfn, p+32, '11I'); nd = d[0] or 1
            self.grpdef[name] = dict(cell=cel, size=list(d[1:1+nd]), order=list(d[6:6+nd]))
        for g, gd in self.grpdat.items():
            df = self.grpdef[gd['defname']]
            gd['size'] = list(df['size']); gd['order'] = df['order']; gd['cell'] = df['cell']
            if gd['vardim']:
                vd = [i for i, s in enumerate(gd['size']) if s == 0]
                gd['size'][vd[0]] = self._maxindex(gd)

    def _maxindex(self, gd):
        ab = self.ab; loc = gd['offset']+392+4*ab; dim = 0
        for nbyte in (3, 2, 1, 0):
            pl = self._u(self.dat, loc, f'256{self.A}')
            ks = [i for i, v in enumerate(pl) if v != self.nil]
            if not ks: break
            k = ks[-1]
            if nbyte > 0:
                dim += k*256**nbyte; loc = pl[k]
            else:
                dim += k
        return dim

    def _cells(self):
        ab = self.ab; self.cells = {}
        for o in self._buckets(self.dfn, self.hl+ab*NHASH):
            p = o+3*ab
            name = self._s(self.dfn, p, 16)
            csz = self._u(self.dfn, p+16, self.A)[0]
            ne = self._u(self.dfn, p+16+ab, 'I')[0]
            loc = o+20+4*ab
            elms = [self._s(self.dfn, loc+16*k, 16) for k in range(ne)]
            self.cells[name] = dict(cellsize=csz, elms=elms)

    def _elements(self):
        ab = self.ab; self.elms = {}
        for o in self._buckets(self.dfn, self.hl):
            p = o+3*ab
            name = self._s(self.dfn, p, 16)
            typ = self.dfn[p+16:p+24].decode()
            selm = self._u(self.dfn, p+24, self.A)[0]
            sval = self._u(self.dfn, p+24+ab, 'I')[0]
            q = p+24+ab+4
            desc = self.dfn[q:q+96].decode('ascii', 'replace')
            dims = self._u(self.dfn, q+96, '6I'); nd = dims[0] or 1
            self.elms[name] = dict(type=typ.strip(), sizeelm=selm, sizeval=sval,
                quantity=desc[:16].rstrip(), units=desc[16:32].rstrip(),
                description=desc[32:96].rstrip(), size=list(dims[1:1+nd]))

    def dtype(self, e):
        t, v = e['type'], e['sizeval']
        if t == 'REAL': return np.dtype(self.bo+('f4' if v == 4 else 'f8'))
        if t == 'INTEGER': return np.dtype(self.bo+{1:'i1',2:'i2',4:'i4',8:'i8'}[v])
        if t == 'LOGICAL': return np.dtype(self.bo+{1:'i1',2:'i2',4:'i4',8:'i8'}[v])
        if t == 'COMPLEX': return np.dtype(self.bo+('c8' if v == 8 else 'c16'))
        if t == 'CHARACTE': return np.dtype(f'S{v}')
        raise ValueError(t)

    def read(self, group, elm, index=1):
        """index 1-based, sobre la (única) dimensión del grupo."""
        gd = self.grpdat[group]; ab = self.ab
        cel = self.cells[gd['cell']]; e = self.elms[elm]
        eoff = 0
        for n in cel['elms']:
            if n == elm: break
            eoff += self.elms[n]['sizeelm']
        cellbytes = sum(self.elms[n]['sizeelm'] for n in cel['elms'])
        off = gd['offset']+392+3*ab
        if gd['vardim']:
            b = [(index//p) % 256 for p in (1, 256, 65536, 16777216)]
            pl = self._u(self.dat, off+ab, f'256{self.A}')
            for bt in (3, 2, 1):
                pl = self._u(self.dat, pl[b[bt]], f'256{self.A}')
            off = pl[b[0]]
            if off == self.nil: raise IndexError
        else:
            off += (index-1)*cellbytes
        raw = self.dat[off+eoff: off+eoff+e['sizeelm']]
        a = np.frombuffer(raw, dtype=self.dtype(e))
        return a.reshape(e['size'], order='F').copy()
