"""Estructuras NEFIS que los archivos reales de prueba no contienen, generadas
con `_nefis_writer`. La disposición de las celdas se calcula a mano en cada
prueba (no se reutiliza la fórmula del lector).

Convención de índices de `NefisFile.read`: 1-based, como vs_let.m.
  - grupo de 1 dimensión: entero
  - grupo multidimensional: tupla con un índice por dimensión, en el orden de
    la definición del grupo
"""
import numpy as np
import pytest

from _nefis_writer import NefisWriter, pack
from _ref_nefis import RefNefis
from d3d.nefis_dll import NefisFile


# ---------------------------------------------------------------------------
# Celda "c": un elemento de cada tipo; los valores codifican el registro k
# ---------------------------------------------------------------------------

def _define_elements(w):
    w.element("A", "REAL", 4, [3, 2], "nivel", "[ M ]", "real*4 2D")
    w.element("B", "INTEGER", 4, [1], "", "[ - ]", "entero")
    w.element("C", "CHARACTE", 10, [1], "", "", "texto")
    w.element("D", "REAL", 8, [2], "", "[ S ]", "real*8")
    w.element("E", "COMPLEX", 8, [2], "", "", "complejo*8")
    w.element("F", "INTEGER", 2, [2, 2], "", "", "entero*2")
    w.cell("c", ["A", "B", "C", "D", "E", "F"])


def _valores(k):
    return {
        "A": (np.arange(6).reshape(3, 2) + 100 * k).astype("<f4"),
        "B": np.array([k], "<i4"),
        "C": np.array([f"cel{k}".ljust(10).encode()], "S10"),
        "D": np.array([k + 0.5, -k], "<f8"),
        "E": np.array([k + 1j, -k - 2j], "<c8"),
        "F": (np.array([[1, 2], [3, 4]]) * k).astype("<i2"),
    }


def _celda(k):
    v = _valores(k)
    return b"".join(pack(v[n], v[n].dtype) for n in "ABCDEF")


VAR_IDX = list(range(1, 301))           # cruza 255/256 (byte b1 de la tabla de punteros)


@pytest.fixture(scope="module")
def basico(tmp_path_factory):
    d = tmp_path_factory.mktemp("basico")
    w = NefisWriter()
    _define_elements(w)
    w.group("gv", "c", [0])             # dimensión variable
    w.group("gf", "c", [3])             # dimensión fija, 3 celdas consecutivas
    w.data_var("gv", "gv", {k: _celda(k) for k in VAR_IDX})
    w.data_fixed("gf", "gf", b"".join(_celda(k) for k in (1, 2, 3)))
    w.write(d / "s.def", d / "s.dat")
    nf = NefisFile(d / "s.def", d / "s.dat")
    yield nf, d
    nf.close()


def test_escritor_validado_con_referencia(basico):
    _, d = basico
    r = RefNefis(d / "s.def", d / "s.dat")
    assert r.grpdat["gv"]["size"] == [300]
    for k in (1, 256, 300):
        np.testing.assert_array_equal(r.read("gv", "A", k), _valores(k)["A"])


@pytest.mark.parametrize("k", [1, 2, 255, 256, 257, 300])
@pytest.mark.parametrize("elm", list("ABCDEF"))
def test_grupo_variable(basico, k, elm):
    nf, _ = basico
    esperado = _valores(k)[elm]
    obtenido = nf.read("gv", elm, k)
    assert obtenido.dtype == esperado.dtype.newbyteorder("=")
    np.testing.assert_array_equal(obtenido.reshape(esperado.shape), esperado)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("elm", list("ABCDEF"))
def test_grupo_fijo_varias_celdas(basico, k, elm):
    """Antes: cell_size solo sumaba hasta el elemento → offset erróneo para k>1."""
    nf, _ = basico
    esperado = _valores(k)[elm]
    np.testing.assert_array_equal(nf.read("gf", elm, k).reshape(esperado.shape), esperado)


def test_numero_de_registros(basico):
    nf, _ = basico
    assert nf.group_info("gv")["max_index"] == 300
    assert nf.group_info("gf")["max_index"] == 3


@pytest.mark.parametrize("grupo,k", [("gv", 301), ("gf", 4), ("gv", 0), ("gf", 0)])
def test_indice_fuera_de_rango(basico, grupo, k):
    nf, _ = basico
    with pytest.raises(IndexError):
        nf.read(grupo, "B", k)


def test_metadatos(basico):
    nf, _ = basico
    e = nf._elements["A"]
    assert (e["quantity"], e["unit"].strip(), e["description"]) == ("nivel", "[ M ]", "real*4 2D")
    assert nf._elements["F"]["dtype"] == np.int16
    assert nf._elements["E"]["dtype"] == np.complex64


# ---------------------------------------------------------------------------
# Grupos multidimensionales y OrderDim
# ---------------------------------------------------------------------------

def _celda_b(valor):
    """Celda con un solo entero."""
    return pack([valor], "<i4")


@pytest.fixture(scope="module")
def multidim(tmp_path_factory):
    d = tmp_path_factory.mktemp("multidim")
    w = NefisWriter()
    w.element("B", "INTEGER", 4, [1])
    w.cell("cb", ["B"])
    ni, nj, nt = 2, 3, 4
    # (a) fijo [2,3], OrderDim [1,2]: primera dimensión más rápida
    w.group("f12", "cb", [ni, nj], [1, 2])
    w.data_fixed("f12", "f12", b"".join(
        _celda_b(10 * i + j) for j in range(1, nj + 1) for i in range(1, ni + 1)))
    # (b) fijo [2,3], OrderDim [2,1]: segunda dimensión más rápida
    w.group("f21", "cb", [ni, nj], [2, 1])
    w.data_fixed("f21", "f21", b"".join(
        _celda_b(10 * i + j) for i in range(1, ni + 1) for j in range(1, nj + 1)))
    # (c) variable en la 2.ª dimensión [2,0]: un bloque por t con las celdas i=1..2
    w.group("v_last", "cb", [ni, 0], [1, 2])
    w.data_var("v_last", "v_last", {t: b"".join(_celda_b(100 * t + i) for i in range(1, ni + 1))
                                    for t in range(1, nt + 1)})
    # (d) variable en la 1.ª dimensión [0,3] con OrderDim 1: un bloque por t con j=1..3
    w.group("v_first", "cb", [0, nj], [1, 2])
    w.data_var("v_first", "v_first", {t: b"".join(_celda_b(100 * t + j) for j in range(1, nj + 1))
                                      for t in range(1, nt + 1)})
    w.write(d / "m.def", d / "m.dat")
    nf = NefisFile(d / "m.def", d / "m.dat")
    yield nf, (ni, nj, nt)
    nf.close()


@pytest.mark.parametrize("grupo", ["f12", "f21"])
def test_fijo_2d_order_dim(multidim, grupo):
    nf, (ni, nj, _) = multidim
    for i in range(1, ni + 1):
        for j in range(1, nj + 1):
            assert nf.read(grupo, "B", (i, j))[0] == 10 * i + j, (grupo, i, j)


def test_variable_en_ultima_dimension(multidim):
    nf, (ni, _, nt) = multidim
    assert nf.group_info("v_last")["shape"] == [ni, nt]
    for t in range(1, nt + 1):
        for i in range(1, ni + 1):
            assert nf.read("v_last", "B", (i, t))[0] == 100 * t + i


def test_variable_en_primera_dimension(multidim):
    nf, (_, nj, nt) = multidim
    assert nf.group_info("v_first")["shape"] == [nt, nj]
    for t in range(1, nt + 1):
        for j in range(1, nj + 1):
            assert nf.read("v_first", "B", (t, j))[0] == 100 * t + j


def test_indice_entero_en_grupo_multidimensional(multidim):
    nf, _ = multidim
    with pytest.raises(ValueError):
        nf.read("f12", "B", 1)


@pytest.mark.parametrize("idx", [(3, 1), (1, 4), (0, 1)])
def test_indice_multidimensional_fuera_de_rango(multidim, idx):
    nf, _ = multidim
    with pytest.raises(IndexError):
        nf.read("f12", "B", idx)


# ---------------------------------------------------------------------------
# Varios grupos de datos con la misma definición; colisiones en tablas hash
# ---------------------------------------------------------------------------

def test_grupos_de_datos_con_misma_definicion(tmp_path):
    w = NefisWriter()
    _define_elements(w)
    w.group("defg", "c", [0])
    w.data_var("run1", "defg", {k: _celda(k) for k in (1, 2)})
    w.data_var("run2", "defg", {k: _celda(10 + k) for k in (1, 2, 3)})
    w.write(tmp_path / "x.def", tmp_path / "x.dat")
    with NefisFile(tmp_path / "x.def", tmp_path / "x.dat") as nf:
        assert {g for g, _ in nf.groups()} == {"run1", "run2"}
        assert nf.group_info("run1")["max_index"] == 2
        assert nf.group_info("run2")["max_index"] == 3
        assert nf.read("run1", "B", 2)[0] == 2
        assert nf.read("run2", "B", 3)[0] == 13


def test_colisiones_en_tablas_hash(tmp_path):
    """Todos los registros en la misma cubeta: se recorren por el campo Link."""
    w = NefisWriter(bucket=lambda name: 7)
    _define_elements(w)
    w.group("gv", "c", [0])
    w.group("gf", "c", [2])
    w.data_var("gv", "gv", {k: _celda(k) for k in (1, 2)})
    w.data_fixed("gf", "gf", _celda(1) + _celda(2))
    w.write(tmp_path / "x.def", tmp_path / "x.dat")
    with NefisFile(tmp_path / "x.def", tmp_path / "x.dat") as nf:
        assert set(nf._elements) == set("ABCDEF")
        assert set(nf._grp_dat) == {"gv", "gf"}
        assert nf.read("gf", "D", 2)[0] == 2.5
        assert nf.read("gv", "C", 2)[0] == b"cel2      "
