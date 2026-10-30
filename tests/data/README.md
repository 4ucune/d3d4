# Datos de prueba

Las pruebas que necesitan archivos reales los buscan aquí. Si no están, se
omiten con un aviso que explica qué falta (las pruebas sintéticas se
ejecutan siempre).

## `04_fluidmud/` — integración (incluido)

Archivos NEFIS del ejemplo `04_fluidmud` de Delft3D-FLOW (malla 3 × 47,
27 MB), resultado de ejecutar ese caso de ejemplo de Delft3D 4 (Deltares):

```
04_fluidmud/trim-sed.def   04_fluidmud/trim-sed.dat
04_fluidmud/trim-mud.def   04_fluidmud/trim-mud.dat
04_fluidmud/trih-sed.def   04_fluidmud/trih-sed.dat
04_fluidmud/trih-mud.def   04_fluidmud/trih-mud.dat
```

Otra ubicación: variable de entorno `D3D_TEST_DATA`.

## `quickplot/` — validación contra MATLAB (por añadir)

Vacía por ahora: ver [quickplot/README.md](quickplot/README.md) para
exportarlos. Mientras falten, la prueba se omite y pide los archivos; si un
`.mat` se exportó de otro trim (otra malla u otro instante), se omite
indicándolo.

Exportaciones de QUICKPLOT ("Save data" → `.mat`) de `04_fluidmud/trim-sed`:
cada `.mat` es un caso de prueba, identificado por el nombre del campo
(`water level`, `velocity`, `bed load transport`, `total transport`,
`mean total transport`, ...) y por su instante. Otro trim y otra carpeta:
`D3D_QUICKPLOT_DIR`, `D3D_QUICKPLOT_DEF`, `D3D_QUICKPLOT_DAT`.

## Archivos grandes (no se distribuyen)

`tests/test_tumaco_3d.py` y la validación con un trim 3D grande usan
`D3D_TUMACO_DEF` / `D3D_TUMACO_DAT` (y `D3D_QUICKPLOT_DIR` para sus `.mat`).
