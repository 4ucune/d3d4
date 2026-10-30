# Cambios

El benchmark y los scripts de ejemplo que se citan abajo forman parte del
entorno de desarrollo y no se distribuyen en este repositorio. Las pruebas
sí (`tests/`), con los datos del ejemplo `04_fluidmud`; las exportaciones de
QUICKPLOT y el trim de Tumaco no (ver `tests/data/README.md`). Los recuentos
de pruebas se obtuvieron con los datos completos.

## Sin publicar — 2026-10-30

### Cambia el comportamiento (rompe compatibilidad)

- **La lectura enmascara por defecto**: `get_var`,
  `data[nombre][...]`, `TrimFile.read/get_var` y `extract_all` ponen NaN
  según la posición de cada variable en la malla, como `d3d_trimfil.m`:
  centros (`z`): −999 y |KCS| ≠ 1; S1 además 999 y celdas secas (igual que
  `water_level`); caras (`u`/`v`) y esquinas (`d`): −999 y puntos sin
  ninguna celda vecina activa (conv2 2×2). Enteros, texto y archivos trih
  no cambian. Los valores del archivo: `mask=False` o `v.raw[...]`.
  Sustituye la decisión D11 ("la lectura no transforma"); interpolar y rotar
  siguen siendo explícitos.
- `to_netcdf`/`save_nc` siguen escribiendo valores crudos.

### Nuevo

- `Variable.location` ('z', 'u', 'v', 'd' o None), `Variable.raw`,
  `TrimFile.grid_shape`, `get_var(..., mask=...)`, `d3d.fields.location`,
  `d3d.fields.mask_read`, `d3d.transforms.corner_mask`.
- Variables derivadas por nombre de QUICKPLOT en `get_var` → `(X, Y, mag)` o
  `component='X'|'Y'|'mag'`, en orden `(T, N, M[, K])`: `horizontal velocity`,
  `filtered depth averaged velocity`, `bed shear stress`,
  `bed load transport`, `d.a. suspended transport`, `total transport`,
  `mean bed load transport`, `mean d.a. suspended transport`,
  `mean total transport`, `staggered bedload transport`
  (`d3d.fields.DERIVED`). Frente a la versión retirada: orden (T, N, M) en
  vez de (T, M, N) y todas las fracciones de sedimento (antes solo la
  primera). `corrected=True` sigue retirado.
- `vector_xy` acepta tuplas de nombres que se suman en las caras.
- `info()` es un catálogo: todas las variables NEFIS (con posición de la
  máscara), campos escalares y vectoriales de QUICKPLOT disponibles y
  funciones de d3d aplicables; `info(nombre)` acepta cualquiera de ellos.
- Campos escalares de QUICKPLOT por nombre en `get_var` (`water level`,
  `density`, `vertical velocity`, ... y constituyentes de NAMCON como
  `Salinity`/`Sediment_sand`, sin distinguir mayúsculas): la variable NEFIS
  con su máscara (`d3d.fields.SCALARS`, `d3d.fields.scalars`). Validado
  contra QUICKPLOT: `water level` y `sediment_sand`.

### Corregido

- Una selección de tiempo vacía devolvía forma `(0,)`; ahora `(0, N, M, ...)`.
- `read_dep` fallaba con los `.dep` reales de Delft3D (cada fila partida en
  líneas de 12 valores, la última más corta: `np.loadtxt` exige columnas
  constantes). Ahora lee todos los números. Verificado con
  `Batimetria_Tumaco_2025_Canal072020_4T.dep`: coincide con DP0 de Tumaco
  (diferencia máxima 4e-6 m en 94 523 puntos).

### Pruebas y validación

- `tests/test_masking.py` (76 pruebas): máscara de cada posición contra
  traducciones literales de `d3d_trimfil.m` (conv2 con bucles), selección
  enmascarada = lectura completa enmascarada, derivadas por nombre.
- `tests/test_quickplot_reference.py` comprueba también la API
  (`get_var` enmascarado y derivadas) y que XCOR/YCOR y XZ/YZ enmascaradas
  tengan NaN donde QUICKPLOT: 18 exportaciones de Tumaco, todas coinciden.
- Pruebas de lectura cruda pasan a `mask=False` / `.raw`.
- QUICKPLOT: `mean total transport` (magnitud, `map-avg-series`) coincide
  con d3d (diferencia máxima 6e-12 m³/s/m, mismas celdas NaN). La prueba
  busca el registro en el grupo de cada campo y admite los tres `mean ...`.
- Suite: 324 aprobadas con Tumaco; 299 aprobadas y 25 omitidas sin él.

## Sin publicar — 2026-10-29


### Eliminado (rompe compatibilidad)

- `get_var(..., corrected=True)`: ahora lanza `TypeError` indicando las
  funciones nuevas. Usar `water_level`, `velocity_xy`, `vector_xy` y
  `d3d.transforms` (orden `(T, N, M[, K])`; tabla de equivalencias en la guía,
  §7).
- Variables derivadas antiguas de `get_var`: `'bed load transport'`,
  `'mean d.a. suspended transport'`, `'staggered bedload transport'`, y el
  parámetro `component`.
- `d3d.clear_cache()` / `NEFISReader.clear_cache()` (no hacían nada desde que
  se quitó la caché de datos).
- Parámetro `dll_path` de `read`, `TrimFile`, `NEFISReader` y `NefisFile` (se
  ignoraba: el lector es Python puro).
- `TrimFile.metadata()`, `NEFISReader.read_step()`,
  `NEFISReader.read_element()`, `NefisFile.element_info()`, `open_nefis()` y
  `models.NEFISGroupData` (sin uso).
- `d3d/_compat.py` (`convolve2d`): solo lo usaba la ruta `corrected=True`.
  scipy sigue siendo opcional (motor netCDF3 y pruebas con `.mat`).
- Validación con `vs_let` (`tools/exportar_referencia_matlab.m`,
  `tests/test_matlab_reference.py`, `tests/_matlab_ref.py`): sustituida por la
  validación con QUICKPLOT. También `tools/ci/pruebas.yml`.

### Pruebas y validación

- Las pruebas que comparaban con `corrected=True` ahora comparan
  `velocity_xy` y `vector_xy` con traducciones literales de
  `uv2cen`/`cur2ca`/máscara KCS. Suite tras la limpieza: 243 aprobadas con
  Tumaco; 219 aprobadas y 24 omitidas sin él; 208 aprobadas y 35 omitidas sin
  scipy.

- Nueva prueba `tests/test_quickplot_reference.py`: compara exportaciones de
  QUICKPLOT (`.mat` de "Save data") con `d3d`; cada archivo de
  `Resultados/matlab` es un caso. Con Tumaco, registros 1, 500 y 961: nivel de
  agua, velocidad vertical y concentración idénticos; velocidad horizontal y
  transportes de fondo y total con diferencias ≤ 2.1e-7 (redondeo float32).
- Suite ejecutada en Windows (Python 3.13.17, numpy 2.5.4, scipy 1.18.1,
  netCDF4 1.7.5.1, pytest 9.2.0): 253 aprobadas y 1 omitida (`vs_let`) con
  Tumaco; 229 aprobadas y 25 omitidas sin él. Primera ejecución del motor
  netCDF4 de `to_netcdf`.

## 0.1.0.dev0 — 2026-10-28


### Errores corregidos que cambian resultados

- **Nivel de agua corregido (`get_var(..., "S1", corrected=True)`)**: la
  máscara de celdas secas desplazaba KFU y KFV en los ejes intercambiados.
  En un trim real (Tumaco) difería de MATLAB en 971–2375 celdas por registro.
  Revisar análisis que usaran esta salida.
- **Tiempo de `extract_all`**: era `ITMAPC·DT`; ahora `ITMAPC·DT·TUNIT`
  segundos desde ITDATE (con TUNIT = 60 los valores anteriores eran 60 veces
  menores).
- **Metadatos de elementos**: el registro se leía con 4 bytes de desfase
  (`quantity` con basura, descripciones truncadas, tamaño de valor erróneo).
- **Variables de texto (CHARACTE)**: se leían como `S4` con formas
  inventadas; algunas lanzaban excepción.
- **Último grupo de cada `.def`** (`TEMPOUT`, `his-version`) no aparecía.
- **Elemento fantasma** `/-1 Active/Non-a` creado a partir de una
  descripción.
- **Grupos de dimensión fija con varias celdas** y **grupos
  multidimensionales / OrderDim**: offsets erróneos (no presentes en los
  trim conocidos; detectado con archivos sintéticos).

### Cambios de comportamiento

- `Variable.shape` y `full_shape` = forma real de `v[:]` (antes invertían
  las dimensiones y omitían el tiempo).
- Las lecturas devuelven arreglos nuevos y escribibles (antes, `index=i`
  devolvía un arreglo de solo lectura compartido por una caché).
- Sin caché de datos (retenía hasta 32 registros completos); `clear_cache()`
  se conserva.
- `NefisFile.groups()` lista los grupos de **datos**.
- `get_var(..., corrected=True)` emite `DeprecationWarning` (misma salida).
- `save_nc` usa el nuevo exportador: dimensiones `(time, N, M, K, ...)` en
  lugar de `dim_<tamaño>` (que mezclaba ejes cuando M == N); ya no exige
  netCDF4.
- `import d3d` ya no requiere scipy.
- Formatos no soportados (NEFIS 4, big-endian) y archivos truncados producen
  errores explícitos en lugar de lecturas erróneas.

### Nuevo

- Lectura NEFIS por tablas hash (`vs_use.m`) y offsets según `vs_let.m`;
  índices multidimensionales en `NefisFile.read(grupo, elem, (i, j))`.
- Lectura selectiva con sintaxis NumPy (`data["U1"][:, n, m, k]`) mediante
  mapeo de memoria.
- `d3d.read(..., dat_path=...)` para `.dat` con otro nombre.
- `d3d.get_time`, `d3d.reference_date`.
- `d3d.water_level`, `d3d.vector_xy`, `d3d.velocity_xy` (orden
  `(T, N, M[, K])`) y `d3d.transforms`.
- `d3d.to_netcdf` (CF-1.8; motores netCDF4 y scipy).
- `pyproject.toml`, pruebas (`tests/`), benchmark (`benchmarks/`), script de
  referencia MATLAB (`tools/`), guía (`docs/guia.md`).
