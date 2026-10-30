"""Lectura NEFIS de bajo nivel (`d3d.nefis_dll.NefisFile`) contra el lector
de referencia independiente (traducción de vs_use.m / vs_let.m).

Criterio: metadatos y valores idénticos (forma, dtype y bits) en todos los
grupos, elementos y pasos de tiempo de los archivos de prueba.
"""
import numpy as np
import pytest

from _ref_nefis import RefNefis
from d3d.nefis_dll import NefisFile


@pytest.fixture(scope="module")
def pair(nefis_pair):
    base, d, t = nefis_pair
    nf = NefisFile(d, t)
    yield base, RefNefis(d, t), nf
    nf.close()


def test_referencia_autoconsistente(pair):
    """El oráculo debe ser coherente consigo mismo antes de usarlo."""
    _, ref, _ = pair
    for name, e in ref.elms.items():
        if e["type"] != "CHARACTE":
            assert e["sizeelm"] == int(np.prod(e["size"])) * e["sizeval"], name
    for name, c in ref.cells.items():
        total = sum(ref.elms[n]["sizeelm"] for n in c["elms"])
        # vs_use admite relleno hasta múltiplo de 4 bytes
        assert c["cellsize"] in (total, 4 * -(-total // 4)), name


def test_grupos_definidos_y_de_datos(pair):
    _, ref, nf = pair
    assert set(nf._groups) == {g["defname"] for g in ref.grpdat.values()}
    assert set(nf._grp_dat) == set(ref.grpdat)


def test_numero_de_registros(pair):
    _, ref, nf = pair
    for g, gd in ref.grpdat.items():
        assert nf.group_info(g)["max_index"] == gd["size"][0], g


def test_celdas(pair):
    _, ref, nf = pair
    for name, c in ref.cells.items():
        assert nf._cells[name]["elements"] == c["elms"], name


@pytest.mark.parametrize("campo,clave_d3d,clave_ref", [
    ("dims", "dimensions", "size"),
    ("bytes de un valor", "single_bytes", "sizeval"),
    ("bytes del elemento", "num_bytes", "sizeelm"),
    ("quantity", "quantity", "quantity"),
    ("unidades", "unit", "units"),
    ("descripción", "description", "description"),
])
def test_metadatos_de_elementos(pair, campo, clave_d3d, clave_ref):
    _, ref, nf = pair
    assert set(nf._elements) == set(ref.elms)
    malos = {}
    for name, e in ref.elms.items():
        a, b = nf._elements[name][clave_d3d], e[clave_ref]
        if isinstance(b, str):
            a, b = a.strip(), b.strip()
        if a != b:
            malos[name] = (a, b)
    assert not malos, f"{campo} distinto: {malos}"


def test_dtypes(pair):
    _, ref, nf = pair
    for name, e in ref.elms.items():
        assert np.dtype(nf._elements[name]["dtype"]) == ref.dtype(e).newbyteorder("="), name


def test_valores_todos_los_pasos(pair):
    """Todos los elementos de todos los grupos, todos los registros."""
    _, ref, nf = pair
    n = 0
    for g, gd in ref.grpdat.items():
        for el in ref.cells[gd["cell"]]["elms"]:
            for idx in range(1, gd["size"][0] + 1):
                esperado = ref.read(g, el, idx)
                obtenido = nf.read(g, el, idx)
                assert obtenido.shape == esperado.shape, (g, el, idx)
                assert obtenido.dtype == esperado.dtype.newbyteorder("="), (g, el, idx)
                np.testing.assert_array_equal(obtenido, esperado,
                                              err_msg=f"{g}/{el}[{idx}]")
                n += 1
    assert n > 0


def test_indice_fuera_de_rango(pair):
    _, ref, nf = pair
    g = next(g for g, gd in ref.grpdat.items() if gd["vardim"])
    el = ref.cells[ref.grpdat[g]["cell"]]["elms"][0]
    with pytest.raises(IndexError):
        nf.read(g, el, ref.grpdat[g]["size"][0] + 1)


def test_elemento_inexistente(pair):
    _, ref, nf = pair
    g = next(iter(ref.grpdat))
    with pytest.raises(KeyError):
        nf.read(g, "NO_EXISTE", 1)


# ---------- formatos no soportados / archivos dañados ----------

def _copia(tmp_path, data_dir, base="trih-mud"):
    import shutil
    for ext in (".def", ".dat"):
        shutil.copy(data_dir / f"{base}{ext}", tmp_path / f"x{ext}")
    return tmp_path / "x.def", tmp_path / "x.dat"


def test_big_endian_no_soportado(tmp_path, data_dir):
    d, t = _copia(tmp_path, data_dir)
    b = bytearray(t.read_bytes())
    b[59:60] = b"B"
    t.write_bytes(b)
    with pytest.raises(NotImplementedError):
        NefisFile(d, t)


def test_dat_truncado(tmp_path, data_dir):
    d, t = _copia(tmp_path, data_dir)
    t.write_bytes(b"abc")
    with pytest.raises(IOError):
        NefisFile(d, t)


def test_def_y_dat_intercambiados(tmp_path, data_dir):
    d, t = _copia(tmp_path, data_dir)
    with pytest.raises(IOError):
        NefisFile(t, d)
