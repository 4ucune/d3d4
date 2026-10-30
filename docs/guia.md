# Guía de `d3d`

Lectura y procesamiento de resultados NEFIS de Delft3D-FLOW (`trim-*`,
`trih-*`) en Python puro. Esta guía describe la API **tal como existe en la
versión 0.1.0.dev0**.

## 1. Qué hace y qué no hace

- Lee los archivos `.def` (definiciones) y `.dat` (datos) de NEFIS 5
  (direcciones de 8 bytes, little-endian) sin la DLL de NEFIS ni MATLAB.
  La estructura se obtiene de las tablas hash del formato, igual que
  `vs_use.m`; los offsets siguen `vs_let.m`.
- Al leer aplica **la máscara que corresponde a cada variable** según su
  posición en la malla (centro, cara o esquina), como `d3d_trimfil.m`; no
  mueve ni interpola datos (§4). Los valores tal como están en el archivo se
  piden con `mask=False` o `v.raw[...]`. Interpolar a centros y rotar son
  funciones explícitas (§7).
- No soporta NEFIS 4 (direcciones de 4 bytes), big-endian ni archivos
  def+dat combinados: los rechaza con `NotImplementedError` en lugar de
  leerlos mal.

## 2. Instalación

Requisitos: Python ≥ 3.10 y NumPy. Opcionales: scipy (motor netCDF3 de
`to_netcdf` y lectura de `.mat` en las pruebas), netCDF4 (exportación
comprimida y por bloques) y pytest.

```bash
pip install git+https://github.com/4ucune/d3d4.git      # desde GitHub
pip install -e .                                        # copia local: núcleo
pip install -e ".[scipy,netcdf,test]"                   # con opcionales y pruebas
```

Windows con Git Bash y un entorno en `.venv`:

```bash
source .venv/Scripts/activate
python -m pip install -e ".[scipy,netcdf,test]"
python -m pytest
```

Versiones verificadas: Python 3.10.12/numpy 2.2.6; Python 3.13.18/numpy
2.5.3/scipy 1.18.1; y Windows con Python 3.13.17/numpy 2.5.4/scipy 1.18.1/
netCDF4 1.7.5.1 (incluye el motor netCDF4).

## 3. Abrir archivos e inspeccionarlos

```python
import d3d

data = d3d.read("trim-sed.dat")            # o el .def; busca el par con el mismo nombre
data = d3d.read("trim-x.def", dat_path="trim-x.dat.exe")   # .dat con otro nombre

d3d.info(data)                 # catálogo: variables NEFIS, campos de QUICKPLOT, funciones
d3d.info(data, "U1")           # detalle de una variable (o de un campo de QUICKPLOT)
data.keys()                    # nombres de variables
v = data["U1"]
v.shape                        # (T, N, M, K) — la forma exacta de v[:]
v.group                        # 'map-series'
v.attributes                   # units, quantity, type, description
v.location                     # 'z', 'u', 'v', 'd' o None: máscara al leer (§4)
v.raw                          # lectura sin máscara: v.raw[480]
data.grid_shape                # (N, M) según KCS; None en trih
data.groups                    # definición de grupos NEFIS
data.close()                   # o: with d3d.read(...) as data:
```

Abrir no lee datos: un trim de 80 GB se abre en centésimas de segundo.

## 4. Leer variables

`data[nombre][selector]` y `d3d.get_var(data, nombre, index=selector)` usan
sintaxis NumPy. El primer índice selecciona registros (0-based); el resto se
aplica a las dimensiones del registro en orden NEFIS:

| Variable típica | Forma de `v[:]` |
|---|---|
| S1, KFU, KFV, TAUKSI | (T, N, M) |
| U1, V1, WPHY, RHO | (T, N, M, K) |
| W, VICWW, DICWW, RICH | (T, N, M, K+1) — interfaces |
| R1 | (T, N, M, K, LSTCI) |
| KCS, ALFAS, XZ, DPS0 (map-const) | (1, N, M) |

```python
u = data["U1"]
u[480]                 # un registro: (N, M, K)
u[0:961:24]            # cada 24 registros
u[:, 200, 150, 0]      # serie temporal en una celda y capa: (T,)
u[..., 0]              # capa 0 de todos los registros
```

Solo se leen del disco los registros pedidos y las páginas que toca la
selección (archivo mapeado en memoria). Cada lectura devuelve un arreglo
nuevo y escribible. Medido en un trim de 80 GB (VM + OneDrive, en frío):
un registro de U1 (5 MB) 0.12 s; serie puntual de 961 registros ≈ 11 s.

### Máscara por defecto

`data[nombre][...]`, `d3d.get_var(...)` y `data.read(...)` ponen NaN donde
QUICKPLOT/`d3d_trimfil.m` no muestra datos. La máscara depende de dónde vive
la variable en la malla (`v.location`, §5) y respeta cualquier selector
(incluida una serie puntual):

| `location` | Variables (ejemplos) | Máscara (`d3d.fields.mask_read`) |
|---|---|---|
| `z` centro | S1, R1, RHO, WPHY, DPS0, XZ, YZ, ALFAS, TAUMAX | −999 → NaN; \|KCS\| ≠ 1 → NaN |
| `z`, solo S1 | S1 | además 999 → NaN y celdas secas (KFU y KFV cerrados en las 4 caras) → NaN, como `'water level'` |
| `u` / `v` caras | U1, V1, SBUU, SSUUA, TAUKSI, DPU0, UMNLDF | −999 → NaN; NaN si ninguna de las 4 celdas vecinas está activa (conv2 2×2, como los campos `staggered`) |
| `d` esquina | XCOR, YCOR, DP0 | −999 → NaN; misma regla de 4 celdas vecinas |
| `None` | KCS, KFU, KCU, ITMAPC, NAMCON, THICK, trih | sin máscara (enteros, texto, no-malla) |

```python
s1 = d3d.get_var(data, "S1", index=480)              # = d3d.water_level(data, index=480)
crudo = d3d.get_var(data, "S1", index=480, mask=False)
crudo = data["S1"].raw[480]                          # equivalente
```

La máscara no interpola: U1 sigue en las caras. Para valores en centros y
componentes cartesianas, §7. Coste medido en Tumaco (páginas en caché): un
registro de S1 0.003 s (crudo 0.0002 s); serie puntual de 961 registros
0.02 s. Comprobado contra QUICKPLOT: las coordenadas XCOR/YCOR y XZ/YZ
enmascaradas tienen NaN exactamente donde las exportaciones de QUICKPLOT (§9).

## 5. Malla de Delft3D y ubicación de las variables

Delft3D usa una malla escalonada. En un arreglo (N, M):

| Ubicación (`location`) | Variables (ejemplos) | Posición de [n, m] |
|---|---|---|
| Z (`z`) | S1, KCS, ALFAS, DPS0, XZ, YZ, WPHY, R1 | centro de la celda (n, m) |
| U (`u`) | U1, KFU, KCU, DPU0, TAUKSI, SBUU | cara entre los centros (n, m) y (n, m+1) |
| V (`v`) | V1, KFV, KCV, DPV0, TAUETA, SBVV | cara entre los centros (n, m) y (n+1, m) |
| esquina (`d`) | XCOR, YCOR, DP0 | vértice de la malla |

La lista de variables en caras y esquinas está en `d3d/fields.py`
(`_LOC_U`, `_LOC_V`, `_LOC_D`, tomada de la tabla de `d3d_trimfil.m`); toda
otra variable real con ejes (N, M) se trata como centro. Las enteras (KCS,
KFU, ...) no se enmascaran.

- `ALFAS`: ángulo (grados, antihorario) de la dirección M respecto a +X.
- `KCS`: 1 = celda activa, 0 = inactiva, 2 = frontera abierta.
- Modelos sigma: la capa k = 0 es la superficial; `THICK` da la fracción de
  la profundidad de cada capa (suma 1, aunque la etiqueta diga `.01*%`).
- Profundidades (`DP*`) positivas hacia abajo.

Verificación con un trim real (Tumaco, malla girada): ALFAS coincide con el
ángulo de la dirección M calculado de XZ/YZ (mediana 0.0003°);
`DPU0[n,m] = min(DPS0[n,m], DPS0[n,m+1])` y
`DPV0[n,m] = min(DPS0[n,m], DPS0[n+1,m])` exactamente, lo que confirma la
posición de U1 y V1.

## 6. Tiempo

```python
t = d3d.get_time(data)                       # datetime64[ms], grupo 'map-series'
s = d3d.get_time(data, as_seconds=True)      # segundos desde ITDATE
d3d.get_time(data, "map-sed-series")         # contador ITMAPS
d3d.get_time(data, "map-avg-series")         # contador ITAVGS
d3d.get_time(data, kind="morpho")            # ITDATE + MORFT (días)
d3d.get_time(trih, "his-series")             # trih: ITHISC
d3d.reference_date(data)                     # ITDATE
```

Fórmula (`readtim` de `d3d_trimfil.m` / `d3d_trihfil.m`):
`ITDATE + contador · DT · TUNIT` segundos. Los tiempos están en la zona
horaria del modelo (TZONE, horas respecto a UTC) y **no** se convierten.
DT y TUNIT son REAL*4; se usa el decimal más corto que los representa (0.2,
no 0.2000000029…) y el resultado se redondea a milisegundos.

## 7. Campos derivados y transformaciones

Todas las salidas nuevas usan el orden `(T, N, M[, K])`.

```python
wl = d3d.water_level(data, index=slice(0, 100))
ux, uy = d3d.velocity_xy(data, index=480)                  # (N, M, K)
ux, uy = d3d.velocity_xy(data, index=slice(None), layer=0) # superficie, todos los registros
fx, fy = d3d.velocity_xy(data, filtered=True)              # UMNLDF/VMNLDF
sx, sy = d3d.vector_xy(data, "SBUUA", "SBVVA")             # cualquier par U/V
```

| Función | Procedimiento (réplica de `d3d_trimfil.m`) |
|---|---|
| `water_level` | −999/999 → NaN; \|KCS\| ≠ 1 → NaN; celdas secas (KFU y KFV cerrados en las 4 caras) → NaN |
| `vector_xy` | −999 → NaN (modelos Z: también 0); `uv2cen`; `cur2ca` con ALFAS; \|KCS\| ≠ 1 → NaN |

`vector_xy` acepta tuplas de nombres que se suman en las caras antes de
interpolar: `vector_xy(data, ("SSUU", "SBUU"), ("SSVV", "SBVV"))` es el
`'total transport'` de QUICKPLOT.

### Campos escalares por nombre

`get_var` acepta también los nombres escalares de QUICKPLOT que en
`d3d_trimfil.m` son una variable NEFIS sin más cálculo que la máscara
(`d3d.fields.SCALARS`: `water level` = S1, `vertical velocity` = WPHY,
`density` = RHO, `maximum bed shear stress` = TAUMAX, ...) y los
constituyentes por su nombre en NAMCON (los LSTCI primeros en R1, la
turbulencia en RTUR1), sin distinguir mayúsculas. Devuelven lo mismo que la
variable NEFIS (mismos selectores `index`, `mask=False` disponible):

```python
nivel = d3d.get_var(data, "water level", index=480)        # = data["S1"][480]
sal = d3d.get_var(data, "Salinity", index=480)             # = data["R1"][480][..., 0]
d3d.fields.scalars(data)                                   # {nombre: (elemento, posición)}
```

Aún no están los campos con cálculo propio (`water depth`, `bed level ...`
con signo cambiado, `depth averaged velocity`, presiones hidrostáticas).

### Variables derivadas por nombre

Con los nombres de QUICKPLOT, `get_var` devuelve `(X, Y, magnitud)` en orden
`(T, N, M[, K])` (o una sola componente con `component='X'|'Y'|'mag'`):

```python
X, Y, mag = d3d.get_var(data, "bed load transport", index=480)   # (N, M)
mag = d3d.get_var(data, "total transport", component="mag")      # (T, N, M)
X, Y, mag = d3d.get_var(data, "horizontal velocity", index=480)   # (N, M, K)
```

| Nombre | Fuentes U / V | Cálculo |
|---|---|---|
| `horizontal velocity` | U1 / V1 | `vector_xy` (centros, cartesianas) |
| `filtered depth averaged velocity` | UMNLDF / VMNLDF | `vector_xy` |
| `bed shear stress` | TAUKSI / TAUETA | `vector_xy` |
| `bed load transport` | SBUU / SBVV | `vector_xy` |
| `d.a. suspended transport` | SSUU / SSVV | `vector_xy` |
| `total transport` | SSUU+SBUU / SSVV+SBVV | suma en caras + `vector_xy` |
| `mean bed load transport` | SBUUA / SBVVA | `vector_xy` |
| `mean d.a. suspended transport` | SSUUA / SSVVA | `vector_xy` |
| `mean total transport` | SSUUA+SBUUA / SSVVA+SBVVA | suma en caras + `vector_xy` |
| `staggered bedload transport` | SBUU / SBVV | en las caras, sin interpolar ni rotar, con la máscara de caras |

En los transportes, el eje de fracciones de sedimento se elimina si hay una
sola fracción; con varias se conserva al final `(T, N, M, L)`. La magnitud de
`staggered bedload transport` combina componentes de caras distintas (como
la antigua API); úsese con cuidado. `d3d.info(data)` lista las derivadas
disponibles en el archivo y `d3d.info(data, "bed load transport")` su
cálculo. Comprobado contra QUICKPLOT con Tumaco: `horizontal velocity`,
`bed load transport` y `total transport` (registros 1, 500 y 961) y
`mean total transport` (único registro de `map-avg-series`, diferencia
máxima 6e-12 m³/s/m).

Los campos `mean ...` están en `map-avg-series`: su `index` recorre los
registros de ese grupo (en Tumaco uno solo), no los de `map-series`. Su
fecha (`d3d.get_time(data, "map-avg-series")`, contador ITAVGS) es el final
del periodo promediado. El promedio lo calcula Delft3D durante la
simulación; d3d y QUICKPLOT solo lo leen, suman las fracciones/componentes
en las caras, lo llevan a centros y lo rotan.

Piezas sueltas en `d3d.transforms` (ejes N y M explícitos; partir de datos
crudos):

```python
from d3d import transforms as tr
u = tr.invalid_to_nan(data["U1"].raw[480])             # (N, M, K)
v = tr.invalid_to_nan(data["V1"].raw[480])
uc, vc = tr.uv2cen(u, v, n_axis=0, m_axis=1)
x, y = tr.cur2ca(uc, vc, data["ALFAS"].raw[0], n_axis=0, m_axis=1)
x = tr.apply_mask(x, tr.active_mask(data["KCS"][0]), n_axis=0, m_axis=1)
esquinas = tr.corner_mask(tr.active_mask(data["KCS"][0]))   # máscara de caras/esquinas
```

Notas:
- `uv2cen`, como en MATLAB, devuelve 0 (no NaN) cuando las dos caras son NaN;
  la máscara KCS posterior cubre las celdas inactivas.
- La rotación usa x = u·cos α − v·sin α, y = u·sin α + v·cos α. En
  coordenadas cartesianas proyectadas, x e y son las direcciones de la
  proyección (≈ Este/Norte salvo la convergencia de meridianos).

### Retirado: `corrected=True`

`get_var(..., corrected=True)` lanza `TypeError`. Equivalencias:

| Antes | Ahora |
|---|---|
| `get_var(d, "S1", corrected=True)` → (T, M, N) | `get_var(d, "S1")` (T, N, M); para (T, M, N): `.transpose(0, 2, 1)` |
| `get_var(d, "U1"/"V1", corrected=True)` → X/Y (T, N, M, K) | `velocity_xy(d)` o `get_var(d, "horizontal velocity")` |
| `get_var(d, "SBUUA"/"SBVVA", corrected=True)` → (T, M, N) | `get_var(d, "mean bed load transport", component="X"/"Y")` (T, N, M) |
| `get_var(d, "DPS0"/"XZ"/..., corrected=True)` (máscara KCS) | `get_var(d, "DPS0")` (máscara por defecto) |

Las variables derivadas por nombre volvieron (ver arriba) con el orden
`(T, N, M)` en lugar del antiguo `(T, M, N)`, y sin quedarse solo con la
primera fracción de sedimento.

## 8. Exportar a netCDF

```python
d3d.to_netcdf(data, "salida.nc",
              variables=["S1", "U1", "V1", "DPS0"],    # nombres NEFIS
              index=slice(0, 100),                      # registros
              derived=["velocity_xy", "water_level"],   # opcional
              rename={"S1": "nivel"},                   # opcional
              engine="auto")                            # netcdf4 | scipy
```

- Variables **crudas** (sin la máscara de lectura: valores del archivo) con
  `(time, N, M[, K])`, `long_name`, `units` (UDUNITS),
  `delft3d_name`, `delft3d_group`, `delft3d_units`, `delft3d_location` y
  `missing_value = -999` en reales.
- `coordinates = "XZ YZ"` solo en variables en puntos Z y `"XCOR YCOR"` en
  esquinas; las variables en caras U/V no llevan `coordinates`.
- Coordenada `time` en segundos desde ITDATE; capas sigma `K` /
  `K_interface` con `ocean_sigma_coordinate` y `formula_terms` cuando se
  exportan S1 y DPS0.
- Motor scipy (netCDF3) escribe todo en memoria y se detiene con
  `MemoryError` por encima de `max_memory_bytes` (2 GB por defecto).
- `data.save_nc(...)` conserva su firma anterior y usa este exportador.

## 9. Pruebas y validación

```bash
python -m pytest                                    # todo lo disponible
D3D_TEST_DATA=/ruta/04_fluidmud python -m pytest    # datos de integración en otra carpeta
D3D_TUMACO_DEF=... D3D_TUMACO_DAT=... python -m pytest tests/test_tumaco_3d.py
D3D_QUICKPLOT_DIR=... D3D_QUICKPLOT_DEF=... D3D_QUICKPLOT_DAT=... \
    python -m pytest tests/test_quickplot_reference.py
```

Los datos se buscan en `tests/data/` (ver `tests/data/README.md`). El
repositorio incluye el ejemplo `04_fluidmud` (299 pruebas al clonar); lo que
falte se omite con un aviso. Si `tests/data/quickplot` no tiene
exportaciones de `trim-sed`, o un `.mat` se exportó de otro trim, la prueba
de QUICKPLOT se omite y explica cómo exportarlos.

- `tests/_ref_nefis.py`: lector de referencia independiente (traducción de
  `vs_use.m`/`vs_let.m`; no importa `d3d`).
- `tests/_nefis_writer.py`: generador de archivos NEFIS sintéticos.
- `tests/test_masking.py`: máscara por variable y derivadas por nombre
  contra traducciones literales de `d3d_trimfil.m`.
- `tests/test_quickplot_reference.py`: cada `.mat` de QUICKPLOT ("Save
  data") es un caso; el campo se identifica por `data.Name` y el registro
  por `data.Time` (en el grupo del campo: `map-avg-series` para los
  `mean ...`). Sin variables de entorno usa `tests/data/quickplot` con
  `trim-sed`.

Así se validó esta versión:

- **Lector NEFIS**: cada variable comparada valor a valor con un lector de
  referencia independiente (traducción de `vs_use.m`/`vs_let.m`) sobre el
  ejemplo `04_fluidmud` de Delft3D, y con archivos NEFIS sintéticos para las
  estructuras que ese ejemplo no contiene (grupos fijos de varias celdas,
  grupos multidimensionales, OrderDim, colisiones en tablas hash, más de
  255 registros).
- **Máscaras y derivadas por nombre**: comparadas con traducciones literales
  de `d3d_trimfil.m` (incluido `conv2` 2×2 escrito con bucles).
- **QUICKPLOT**: exportaciones "Save data" (`.mat`) de un trim real 3D
  (Tumaco) para `water level`, `velocity` (componentes y magnitud),
  `horizontal velocity`, `vertical velocity`, la concentración de
  `sediment_sand`, `bed load transport`, `total transport` y
  `mean total transport`. QUICKPLOT guarda los campos 2D en celdas como
  `Val` (M-1, N-1), con `Val[i, j]` = celda (m, n) = (i+1, j+1) y X/Y en
  las esquinas; en d3d, `campo[1:, 1:].T`.
  Resultado con Tumaco (registros 1, 500 y 961): nivel, velocidad vertical y
  concentración idénticos; velocidad horizontal ≤ 2.1e-7 m/s y transportes
  ≤ 5e-11 (redondeo float32); misma máscara (incluidas 10 563 celdas secas
  en el registro 1). Cada caso se comprueba dos veces: con el cálculo literal
  y con la API (`get_var` enmascarado y las derivadas por nombre); además, el
  patrón de NaN de XCOR/YCOR o XZ/YZ debe ser idéntico al de QUICKPLOT.

## 10. Errores frecuentes

| Mensaje | Causa probable | Qué hacer |
|---|---|---|
| `FileNotFoundError: archivo .def no encontrado` | Falta el `.def` o tiene otro nombre | Copiarlo junto al `.dat`, o `d3d.read(def, dat_path=dat)` |
| `ValueError: use un archivo .dat o .def` | El `.dat` no termina en `.dat` (p. ej. `.dat.exe`) | `d3d.read("x.def", dat_path="x.dat.exe")` |
| `NotImplementedError: formato NEFIS no soportado` | NEFIS 4 o big-endian | No soportado; convertir o usar MATLAB |
| `IOError: no es un archivo NEFIS de data` | `.def` y `.dat` intercambiados o archivo no NEFIS | Revisar las rutas |
| `IOError: archivo NEFIS incompleto` / `IndexError ... file truncated?` | Simulación interrumpida o copia incompleta | Volver a copiar; leer solo registros completos |
| `IndexError: indice fuera de rango` | Registro inexistente | Ver `data["S1"].shape[0]` |
| `KeyError: variable 'X' no encontrada` | Nombre NEFIS distinto | `data.keys()` / `d3d.info(data)` |
| `TypeError: get_var(..., corrected=True) se eliminó` | Código escrito para la ruta retirada | Usar `water_level` / `velocity_xy` / `vector_xy` (§7) |
| `MemoryError: exportación de N GB con el motor scipy` | netCDF3 escribe en memoria | `index=`/`variables=` más pequeños o instalar netCDF4 |
| Valores −999 con `mask=False` / `.raw` | Centinela de Delft3D (celda/cara sin dato) | Leer con la máscara por defecto o `d3d.transforms.invalid_to_nan` |
| NaN inesperados en una variable | Máscara por defecto (§4) | Revisar `v.location`; valores del archivo con `mask=False` |
| `TypeError: component solo se usa con variables derivadas` | `component=` con una variable NEFIS | Usar un nombre de §7 o quitar `component` |

## 11. Limitaciones

- TrimFile/Variable: grupos de una dimensión (todos los trim/trih conocidos).
  Grupos multidimensionales: `data._reader.raw_reader.read(grupo, elem, (i, j))`.
- El número de registros se fija al abrir (no ve pasos escritos después por
  una simulación en curso).
- `get_time(kind="morpho")` usa MORFT; no aplica las opciones `morstt`/
  `morfac` de QUICKPLOT.
