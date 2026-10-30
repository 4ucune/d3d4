"""Transformaciones científicas de campos Delft3D-FLOW (funciones puras NumPy).

La lectura (`get_var` / `Variable[...]`) solo aplica la máscara de cada
variable (ver `d3d.fields.mask_read`); interpolar y rotar es siempre
explícito. Estas funciones replican las de `d3d_trimfil.m`.

Malla escalonada de Delft3D (verificado con DPU0/DPV0 de un trim real):
  - S1, KCS, ALFAS, DPS0, ... : centro de celda (n, m)
  - U1[n, m]  : cara entre los centros (n, m) y (n, m+1)   (dirección M / ξ)
  - V1[n, m]  : cara entre los centros (n, m) y (n+1, m)   (dirección N / η)
  - ALFAS     : ángulo (grados, antihorario) de la dirección M respecto a +X
Los arreglos siguen el orden NEFIS (…, N, M[, K]); las funciones reciben de
forma EXPLÍCITA la posición de los ejes N y M (`n_axis`, `m_axis`).
"""
import numpy as np

__all__ = ["invalid_to_nan", "active_mask", "corner_mask", "apply_mask", "dry_mask",
           "uv2cen", "cur2ca"]


def invalid_to_nan(a, sentinels=(-999.0,)):
    """Copia en coma flotante con los valores centinela convertidos a NaN.

    Conserva float32/float64; los enteros pasan a float64. d3d_trimfil.m usa
    -999 para velocidades y -999/999 para niveles y profundidades.
    """
    a = np.asarray(a)
    out = np.array(a, dtype=a.dtype if a.dtype.kind == "f" else np.float64, copy=True)
    for s in sentinels:
        out[out == s] = np.nan
    return out


def active_mask(kcs):
    """Celdas activas de cálculo: |KCS| == 1 (0 = inactiva, 2 = frontera abierta)."""
    return np.abs(np.asarray(kcs)) == 1


def corner_mask(active):
    """Esquinas (y caras) activas: alguna de las 4 celdas vecinas es activa.

    d3d_trimfil.m (ReqLoc 'd'): act = conv2(act, [1 1; 1 1], 'valid') > 0 sobre
    la máscara de celdas extendida con una fila y una columna que repiten la
    última (idx1). En (N, M): esquina(n, m) = any(active[n:n+2, m:m+2]).
    """
    a = np.pad(np.asarray(active, dtype=bool), ((0, 1), (0, 1)), mode="edge")
    return a[:-1, :-1] | a[1:, :-1] | a[:-1, 1:] | a[1:, 1:]


def _expand(mask2d, ndim, n_axis, m_axis):
    """Vista de un arreglo (N, M) con forma difundible sobre `ndim` dimensiones."""
    n_axis %= ndim
    m_axis %= ndim
    if n_axis == m_axis:
        raise ValueError("n_axis y m_axis deben ser distintos")
    shape = [1] * ndim
    shape[n_axis], shape[m_axis] = mask2d.shape
    arr = mask2d if n_axis < m_axis else mask2d.T
    return arr.reshape(shape)


def apply_mask(a, mask2d, n_axis, m_axis):
    """NaN donde `mask2d` (N, M) es False. Devuelve una copia flotante."""
    a = invalid_to_nan(a, sentinels=())
    m = np.asarray(mask2d, dtype=bool)
    if (a.shape[n_axis], a.shape[m_axis]) != m.shape:
        raise ValueError(f"máscara {m.shape} incompatible con ejes N,M "
                         f"{(a.shape[n_axis], a.shape[m_axis])}")
    a[np.broadcast_to(~_expand(m, a.ndim, n_axis, m_axis), a.shape)] = np.nan
    return a


def _shift_prev(a, axis):
    """a[..., i-1, ...] con el primer índice repetido (MATLAB: a(:,[1 1:end-1]))."""
    idx = np.r_[0, np.arange(a.shape[axis] - 1)]
    return np.take(a, idx, axis=axis)


def dry_mask(kfu, kfv, n_axis, m_axis):
    """Celdas secas: las 4 caras cerradas (d3d_trimfil.m, nivel de agua).

    seca(n, m) = kfu(n, m) == 0 y kfu(n, m-1) == 0 y kfv(n, m) == 0 y kfv(n-1, m) == 0
    """
    kfu = np.asarray(kfu)
    kfv = np.asarray(kfv)
    return ((kfu == 0) & (_shift_prev(kfu, m_axis) == 0)
            & (kfv == 0) & (_shift_prev(kfv, n_axis) == 0))


def _center_mean(a, axis):
    """Media de a[i] y a[i-1] ignorando NaN; índice 0 → NaN (uv2cen.m)."""
    a = np.asarray(a)
    a = np.array(a, dtype=a.dtype if a.dtype.kind == "f" else np.float64)
    cur = np.take(a, np.arange(1, a.shape[axis]), axis=axis)
    prev = np.take(a, np.arange(0, a.shape[axis] - 1), axis=axis)
    cnt = np.isfinite(cur).astype(a.dtype) + np.isfinite(prev)
    out = np.full(a.shape, np.nan, dtype=a.dtype)
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(1, None)
    # Como uv2cen.m: si ambas son NaN, 0/max(0,1) = 0 (no NaN). La máscara KCS
    # que se aplica después cubre las celdas inactivas.
    out[tuple(sl)] = (np.nan_to_num(cur) + np.nan_to_num(prev)) / np.maximum(cnt, 1)
    return out


def uv2cen(u, v, n_axis, m_axis):
    """Componentes en caras U/V → centros de celda (uv2cen.m).

    U(n, m) = media(u(n, m), u(n, m-1)),  V(n, m) = media(v(n, m), v(n-1, m)),
    ignorando NaN (si ambas son NaN el resultado es 0, igual que MATLAB).
    La primera fila (N) de V y la primera columna (M) de U son NaN.
    Las entradas con centinelas deben convertirse antes a NaN.
    """
    return _center_mean(u, m_axis), _center_mean(v, n_axis)


def cur2ca(u, v, alfas_deg, n_axis, m_axis):
    """Rotación de componentes de malla (ξ, η) a cartesianas (x, y) (cur2ca.m).

        x = u·cos α − v·sin α,   y = u·sin α + v·cos α

    con α = ALFAS (grados, dirección M respecto a +X), arreglo (N, M).
    """
    u = np.asarray(u)
    v = np.asarray(v)
    a = np.deg2rad(np.asarray(alfas_deg, dtype=np.float64))
    dt = np.result_type(u.dtype, v.dtype, np.float32)
    c = _expand(np.cos(a).astype(dt), u.ndim, n_axis, m_axis)
    s = _expand(np.sin(a).astype(dt), u.ndim, n_axis, m_axis)
    return u * c - v * s, u * s + v * c
