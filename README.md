# d3d

Lectura de archivos NEFIS de Delft3D-FLOW (`trim-*`, `trih-*`) en **Python puro**:
no requiere la DLL de NEFIS ni MATLAB. Solo depende de NumPy.

Al leer, cada variable recibe la **máscara que le corresponde según su
posición en la malla** (centro, cara o esquina), como el toolbox MATLAB
(`d3d_trimfil.m`). Incluye **variables derivadas** con los nombres de
QUICKPLOT (`bed load transport`, `total transport`,
`mean d.a. suspended transport`, `staggered bedload transport`, ...).

> Estado: desarrollo (0.1.0.dev0). Guía completa en [docs/guia.md](docs/guia.md)
> y cambios en [CHANGELOG.md](CHANGELOG.md).

## Instalación

Desde GitHub:

```bash
pip install git+https://github.com/4ucune/d3d4.git
pip install "d3d[scipy,netcdf] @ git+https://github.com/4ucune/d3d4.git"   # con opcionales
```

Desde una copia local del repositorio:

```bash
pip install -e .            # núcleo (numpy)
pip install -e .[scipy]     # opcional: motor netCDF3 de to_netcdf
pip install -e .[netcdf]    # opcional: netCDF4 (compresión, exportaciones grandes)
pip install -e .[test]      # pytest (y scipy para leer los .mat de QUICKPLOT)
```

En Windows con un entorno virtual en `.venv` (Git Bash):

```bash
.venv/Scripts/python.exe -m pip install -e ".[scipy,netcdf,test]"
.venv/Scripts/python.exe -m pytest
```

Requiere Python ≥ 3.10 y NumPy.

## Uso básico

```python
import d3d

datos = d3d.read("trim-sed.dat")      # .dat o .def; ambos deben existir
# si el .dat tiene otro nombre: d3d.read("trim-x.def", dat_path="trim-x.dat.exe")

datos.info()                          # catálogo: variables NEFIS, campos de QUICKPLOT y funciones
d3d.info(datos, "total transport")    # detalle de cualquier nombre

s1 = d3d.get_var(datos, "S1", index=0)            # (N, M), con máscara: NaN en inactivas y secas
s1_crudo = d3d.get_var(datos, "S1", index=0, mask=False)   # valores del archivo
u = d3d.get_var(datos, "U1")                      # (T, N, M, K), máscara de caras
xc = d3d.get_var(datos, "XCOR", index=0)          # (N, M), máscara de esquinas

# Campos escalares con nombre de QUICKPLOT (la variable NEFIS con su máscara)
nivel = d3d.get_var(datos, "water level", index=0)    # = S1
arena = d3d.get_var(datos, "Sediment_sand", index=0)  # = R1[..., i] según NAMCON

# Variables derivadas → (X, Y, magnitud), en centros y en cartesianas
X, Y, mag = d3d.get_var(datos, "mean d.a. suspended transport", index=0)
x = d3d.get_var(datos, "bed load transport", index=0, component="X")

# Acceso directo por nombre
v = datos["S1"]
v.shape          # (T, N, M): forma exacta de v[:]
v.ntimes         # número de registros
v.location       # 'z' → máscara de centro + celdas secas
v[0]             # primer registro (enmascarado)
v[-1]            # último
v[0:10]          # primeros 10
v[:, 1, 20]      # serie temporal en una celda
v.raw[0]         # primer registro sin máscara
v.attributes["units"], v.attributes["name"]

datos.close()    # o usar `with d3d.read(...) as datos:`
```

Todos los arreglos siguen el orden **`(T, N, M[, K])`** de NEFIS (registro,
fila N, columna M, capa K; en modelos sigma k = 0 es la superficie). Las
lecturas devuelven arreglos nuevos y escribibles, y solo leen del disco los
registros y páginas que toca la selección.

## Máscara por variable

Como en `d3d_trimfil.m`, la máscara depende de dónde vive la variable
(`v.location`). La máscara no interpola ni rota: U1 sigue en sus caras.

| `location` | Variables (ejemplos) | NaN donde |
|---|---|---|
| `z` centro | S1, R1, RHO, WPHY, DPS0, XZ, YZ | −999; \|KCS\| ≠ 1 |
| `z`, solo S1 | S1 (`'water level'`) | además 999 y celdas secas (KFU/KFV cerrados en las 4 caras) |
| `u`/`v` caras | U1, V1, SBUU, SSUUA, TAUKSI, DPU0 | −999; ninguna de las 4 celdas vecinas activa (conv2 2×2) |
| `d` esquinas | XCOR, YCOR, DP0 | −999; ninguna de las 4 celdas vecinas activa |
| `None` | KCS, KFU, ITMAPC, NAMCON, THICK, archivos trih | sin máscara |

Sin máscara: `get_var(..., mask=False)`, `datos.read(nombre, mask=False)` o
`v.raw[...]`.

## API

### Nivel alto (`d3d.*`)

| Función | Descripción |
|---------|-------------|
| `d3d.read(ruta, dat_path=None)` | Abre `.dat` o `.def`; devuelve `TrimFile` |
| `d3d.get_var(datos, nombre, index=None, mask=True, component=None)` | Variable NEFIS enmascarada (o cruda con `mask=False`); para derivadas, `(X, Y, mag)` o una componente |
| `d3d.info(datos, nombre=None)` | Variables, posición, unidades y derivadas disponibles |
| `d3d.get_time(datos, grupo="map-series")` | Tiempos `datetime64[ms]` (ITDATE + contador·DT·TUNIT) |
| `d3d.water_level(datos, index)` | S1 enmascarado (igual que `get_var(datos, "S1")`) |
| `d3d.velocity_xy(datos, index, layer)` | U1/V1 en centros y cartesianas |
| `d3d.vector_xy(datos, u, v, index, layer)` | Cualquier par U/V (o suma de pares) en centros y cartesianas |
| `d3d.to_netcdf(datos, ruta, ...)` / `d3d.save_nc` | Exporta a netCDF CF-1.8 (valores crudos) |
| `d3d.extract_all(datos, variables)` | Varias variables (enmascaradas) en un dict |
| `d3d.dump_json(datos, ruta)` | Metadatos a JSON |
| `d3d.read_dep(ruta_dep, grd_path=None, m=None, n=None)` | Archivos `.dep` |
| `d3d.transforms` | Piezas sueltas: `invalid_to_nan`, `active_mask`, `corner_mask`, `dry_mask`, `uv2cen`, `cur2ca`, `apply_mask` |

### Nivel medio (`TrimFile`)

```python
datos.groups            # {nombre: NEFISGroupDef}
datos.variables         # {nombre: Variable}
datos.cells             # {nombre: NEFISCell}
datos.grid_shape        # (N, M) según KCS; None en trih
datos["S1"]             # Variable
datos.read("S1", index=0, mask=True)
datos.info("U1")        # detalle (incluye la posición)
datos.info("bed load transport")   # fuentes y cálculo de una derivada
datos.extract_all()
datos.save_nc("salida.nc")
datos.dump_json("metadatos.json")
```

### Variable

```python
v = datos["U1"]
v.name, v.group         # 'U1', 'map-series'
v.shape                 # (T, N, M, K)
v.ntimes                # T
v.location              # 'u'
v.attributes            # units, quantity, type, name, description
v[480]                  # (N, M, K) enmascarado
v[:, 200, 150, 0]       # serie temporal (T,)
v.raw[480]              # sin máscara
```

### Bajo nivel (`NefisFile`)

```python
nf = datos._reader.raw_reader
nf.groups()                       # [(grupo, registros), ...]
nf.group_info("map-series")       # cell, max_index, dimensions, ...
nf.cells()                        # [(celda, [elementos]), ...]
nf.elements()                     # metadatos de los elementos
nf.read("map-const", "XCOR", 1)   # lectura cruda (índice 1-based)
```

## Catálogo: `d3d.info(datos)`

Muestra, para el archivo abierto:

1. **Variables NEFIS** por grupo: forma `(T, ...)`, posición de la máscara
   (`z`, `u`, `v`, `d`, `-`), unidades y descripción. Son todas las del `.def`.
2. **Campos escalares de QUICKPLOT** disponibles (`water level`, `density`,
   `vertical velocity`, constituyentes de NAMCON como `Salinity` o
   `Sediment_sand`, turbulencia, ...), con la variable NEFIS de la que salen.
   Son la variable NEFIS con su máscara (`d3d.fields.SCALARS`); los campos
   de QUICKPLOT que llevan cálculo propio (`water depth`, `bed level`,
   promedios en la vertical) todavía no están.
3. **Campos vectoriales de QUICKPLOT** (tabla siguiente).
4. **Funciones de d3d** aplicables (`water_level`, `velocity_xy`,
   `vector_xy` con los pares de caras disponibles, `get_time` por grupo).

`d3d.info(datos, nombre)` da el detalle de cualquiera de ellos.

## Variables derivadas

Nombres de QUICKPLOT; `get_var` devuelve `(X, Y, magnitud)` en `(T, N, M[, K])`
(`(N, M[, K])` con `index` entero). En los transportes se elimina el eje de
fracciones cuando hay una sola.

| Variable | Fuentes U / V | Cálculo (`d3d_trimfil.m`) |
|----------|---------------|---------------------------|
| `horizontal velocity` | U1 / V1 | uv2cen + cur2ca (ALFAS) + máscara KCS |
| `filtered depth averaged velocity` | UMNLDF / VMNLDF | ídem |
| `bed shear stress` | TAUKSI / TAUETA | ídem |
| `bed load transport` | SBUU / SBVV | ídem |
| `d.a. suspended transport` | SSUU / SSVV | ídem |
| `total transport` | SSUU+SBUU / SSVV+SBVV | suma en caras + ídem |
| `mean bed load transport` | SBUUA / SBVVA | ídem |
| `mean d.a. suspended transport` | SSUUA / SSVVA | ídem |
| `mean total transport` | SSUUA+SBUUA / SSVVA+SBVVA | suma en caras + ídem |
| `staggered bedload transport` | SBUU / SBVV | en caras, sin interpolar ni rotar, máscara conv2 2×2 |

`get_var(..., corrected=True)` se retiró (lanza `TypeError`); sus
equivalencias están en la guía (§7).

## Grupos disponibles (trim)

| Grupo | Contenido |
|-------|-----------|
| `map-const` | Constantes del modelo (geometría, máscaras, coordenadas, capas) |
| `map-series` | Resultados en el tiempo (S1, U1, V1, WPHY, R1, RHO, KFU, KFV, ...) |
| `map-info-series` | Contador de tiempo `ITMAPC` |
| `map-sed-series` | Resultados de sedimento (SBUU, SSUU, DPS, ...) |
| `map-infsed-serie` | Información de sedimento (`ITMAPS`, MORFAC, MORFT) |
| `map-avg-series` | Promedios (SBUUA, SSUUA, ...) |
| `map-infavg-serie` | Contador de promedios (`ITAVGS`) |

## Inventario del paquete `d3d/`

| Archivo | Rol |
|---------|-----|
| `__init__.py` | Punto de entrada: `read`, `get_var`, `info`, `get_time`, campos, `to_netcdf`, ... |
| `api.py` | Funciones de alto nivel (`read`, `get_var`, `info`, `extract_all`, `save_nc`, `dump_json`) |
| `trimfile.py` | `TrimFile` y `Variable` (lectura selectiva, máscara por defecto, `.raw`, `info`) |
| `fields.py` | Máscara por variable (`location`, `mask_read`), `water_level`, `vector_xy`, `velocity_xy`, derivadas por nombre (`DERIVED`) |
| `transforms.py` | Transformaciones puras de `d3d_trimfil.m` (`uv2cen`, `cur2ca`, máscaras) |
| `timeaxis.py` | `get_time`, `reference_date` |
| `export.py` | `to_netcdf` (CF-1.8; motores netCDF4 y scipy) |
| `depfile.py` | `read_dep` |
| `reader.py` | `NEFISReader`: capa media sobre `NefisFile` |
| `nefis_dll.py` | `NefisFile`: lector NEFIS 5 en Python puro (tablas hash de `vs_use.m`, offsets de `vs_let.m`, mmap) |
| `models.py` | Dataclasses `NEFISElement`, `NEFISCell`, `NEFISGroupDef` |

## Pruebas

```bash
pytest
```

Al clonar se ejecutan 299 pruebas: las sintéticas (lector NEFIS con
archivos generados por `tests/_nefis_writer.py`, transformaciones, `.dep`,
exportación) y las de integración con el ejemplo `04_fluidmud`, incluido en
`tests/data/`. Las que necesitan archivos que no se distribuyen se omiten
con un aviso que explica qué añadir ([tests/data/README.md](tests/data/README.md)):

| Datos | ¿Incluidos? | Pruebas |
|---|---|---|
| `tests/data/04_fluidmud/`: trim/trih del ejemplo `04_fluidmud` de Delft3D (27 MB) | sí | lector contra la referencia independiente (`tests/_ref_nefis.py`), máscaras (`tests/test_masking.py`), campos, tiempo, API |
| `tests/data/quickplot/`: `.mat` de QUICKPLOT exportados de `trim-sed` | no: la prueba los pide | `tests/test_quickplot_reference.py` |
| trim 3D grande y sus `.mat` (`D3D_TUMACO_DEF/DAT`, `D3D_QUICKPLOT_DIR`) | no | `tests/test_tumaco_3d.py`, QUICKPLOT con ese trim |

## Validación

Resultados obtenidos con los datos completos (detalle en la
[guía](docs/guia.md), §9):

- **Lector NEFIS**: comparado valor a valor con un lector de referencia
  independiente (traducción de `vs_use.m` / `vs_let.m`) sobre el ejemplo
  `04_fluidmud` de Delft3D y sobre archivos NEFIS sintéticos con las
  estructuras que ese ejemplo no contiene.
- **Máscaras y campos derivados**: comparados con traducciones literales de
  `d3d_trimfil.m`.
- **QUICKPLOT**: exportaciones "Save data" de un trim real 3D (Tumaco,
  424×300×10, 961 registros; registros 1, 500 y 961): nivel de agua,
  velocidad vertical y concentración idénticos; velocidad y transportes
  (incluido `mean total transport`) con diferencias ≤ 2e-7; NaN de las
  coordenadas enmascaradas en las mismas posiciones que QUICKPLOT.

## Rendimiento

Con Tumaco (424×300×10, 961 registros, páginas en caché): un registro de S1
enmascarado 0.003 s; serie puntual de 961 registros 0.02 s; un registro de
U1 enmascarado 0.004 s.

## Limitaciones conocidas

- Solo NEFIS 5 con direcciones de 8 bytes y little-endian (verificado con
  archivos NEFIS 5.00). NEFIS4 / big-endian / archivo único no soportados.
- `get_time` no convierte a UTC (usa la zona horaria del modelo, TZONE).
- `TrimFile`/`Variable` manejan grupos de una dimensión (todos los trim/trih
  conocidos); los multidimensionales se leen con `NefisFile.read(..., (i, j))`.
- `get_var` sin `index` carga la variable completa en memoria; usar la
  lectura selectiva para archivos grandes.
- El número de registros se fija al abrir el archivo.
- La posición de cada variable para la máscara sale de una lista tomada de
  `d3d_trimfil.m`; una variable en caras que no esté en ella se enmascara
  como centro (`v.location` lo muestra).
- Derivadas con varias fracciones de sedimento: se devuelven por fracción;
  no se ha comprobado contra QUICKPLOT con más de una fracción.

## Licencia

MIT. Ver [LICENSE](LICENSE).

Los algoritmos de lectura y de transformación replican el comportamiento de
`vs_use.m`, `vs_let.m` y `d3d_trimfil.m` del toolbox MATLAB de Delft3D
(Deltares, [openearth/delft3d_matlab](https://github.com/openearth/delft3d_matlab)),
que no se incluye en este repositorio.
