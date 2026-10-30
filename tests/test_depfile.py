"""`d3d.read_dep` con archivos sintéticos (formato .dep/.grd de Delft3D)."""
import numpy as np
import pytest

from d3d import read_dep

M, N = 4, 2   # puntos de malla del .grd (encabezado "M N")


def _escribir(tmp_path, valores, crlf=True):
    nl = "\r\n" if crlf else "\n"
    grd = tmp_path / "malla.grd"
    grd.write_text(f"Coordinate System = Cartesian{nl}       {M}       {N}{nl}"
                   f"       0       0       0{nl} ETA=    1   1.0 2.0 3.0 4.0{nl}",
                   newline="")
    dep = tmp_path / "malla.dep"
    dep.write_text(nl.join(" ".join(f"{v: .7E}" for v in fila) for fila in valores) + nl,
                   newline="")
    return dep, grd


def _campo():
    a = np.arange((N + 1) * (M + 1), dtype=float).reshape(N + 1, M + 1)
    a[-1, :] = -999.0
    a[:, -1] = -999.0
    a[0, 1] = -999.0          # un faltante dentro del dominio
    return a


@pytest.mark.parametrize("crlf", [True, False])
def test_lee_con_grd_vecino(tmp_path, crlf):
    a = _campo()
    dep, _ = _escribir(tmp_path, a, crlf)
    out = read_dep(dep)
    assert out.shape == (N, M)
    esperado = a[:-1, :-1].copy()
    esperado[esperado == -999] = np.nan
    np.testing.assert_array_equal(out, esperado)


def test_m_n_explicitos_y_sin_nan(tmp_path):
    a = _campo()
    dep, _ = _escribir(tmp_path, a)
    out = read_dep(dep, m=M, n=N, nan_missing=False)
    np.testing.assert_array_equal(out, a[:-1, :-1])


def test_tamano_incorrecto(tmp_path):
    dep, _ = _escribir(tmp_path, _campo())
    with pytest.raises(ValueError):
        read_dep(dep, m=M + 1, n=N)


def test_sin_grd_ni_dimensiones(tmp_path):
    dep = tmp_path / "solo.dep"
    dep.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        read_dep(dep)


def test_filas_partidas_en_lineas_de_12(tmp_path):
    """Formato real de Delft3D: cada fila de M+1 valores se escribe en líneas
    de hasta 12 valores (la última más corta); antes fallaba con loadtxt."""
    m, n = 15, 3
    a = np.arange((n + 1) * (m + 1), dtype=float).reshape(n + 1, m + 1)
    a[-1, :] = -999.0
    a[:, -1] = -999.0
    lineas = []
    for fila in a:
        for k in range(0, fila.size, 12):
            lineas.append(" ".join(f"{v: .7E}" for v in fila[k:k + 12]))
    dep = tmp_path / "real.dep"
    dep.write_text("\n".join(lineas) + "\n")
    out = read_dep(dep, m=m, n=n)
    np.testing.assert_array_equal(out, a[:-1, :-1])
