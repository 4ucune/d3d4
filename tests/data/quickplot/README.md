# Exportaciones de QUICKPLOT de `trim-sed`

Esta carpeta está vacía a propósito: añade aquí los `.mat` exportados con
QUICKPLOT desde `tests/data/04_fluidmud/trim-sed.dat`.

1. En MATLAB: `addpath` del toolbox de Delft3D y `d3d_qp` (QUICKPLOT).
2. Abre `tests/data/04_fluidmud/trim-sed.dat`.
3. Elige un campo (`water level`, `velocity`, `bed load transport`,
   `total transport`, `mean total transport`, ...) y un instante.
4. "Save data" → `.mat` con el nombre que propone QUICKPLOT (por ejemplo
   `water level.mat`; para otros instantes, `water level_145.mat`).
5. Copia el `.mat` aquí y ejecuta `pytest tests/test_quickplot_reference.py`.

Cada `.mat` es un caso de prueba. Si falta o se exportó de otro trim, la
prueba se omite y lo indica.
