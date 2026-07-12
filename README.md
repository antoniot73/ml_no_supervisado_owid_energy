# Aprendizaje no supervisado aplicado al dataset OWID Energy

Proyecto académico reproducible para analizar perfiles energéticos nacionales mediante dos técnicas de aprendizaje no supervisado:

- **K-means**, para identificar grupos de países con perfiles energéticos semejantes.
- **PCA**, para reducir dimensionalidad e interpretar las principales fuentes de variación entre países.

El análisis utiliza datos de **Our World in Data (OWID)** correspondientes al año **2022**.

---

## Objetivo

Responder la pregunta:

> ¿Qué perfiles energéticos pueden identificarse entre los países y qué variables explican las principales diferencias de sus sistemas energéticos?

El proyecto analiza simultáneamente:

- consumo de energía por habitante;
- intensidad energética;
- demanda eléctrica por habitante;
- participación fósil;
- participación renovable;
- participación nuclear;
- participación baja en carbono;
- composición por carbón, petróleo y gas.

---

## Resultados principales

### K-means

Se evaluaron valores de `k` entre 2 y 8. La mayor silueta promedio se obtuvo con:

```text
k = 2
silueta ≈ 0.328
```

Este resultado respalda una solución descriptiva de dos clústeres, aunque la separación es moderada.

Los perfiles identificados fueron:

1. **Perfil fósil dominante**
   - consumo medio aproximado: **39,425 kWh por habitante**;
   - participación fósil media: **86.1 %**;
   - participación baja en carbono: **13.9 %**;
   - demanda eléctrica media: **5,556 kWh por habitante**.

2. **Perfil de mayor consumo con más fuentes bajas en carbono**
   - consumo medio aproximado: **60,362 kWh por habitante**;
   - participación fósil media: **48.8 %**;
   - participación baja en carbono: **51.2 %**;
   - demanda eléctrica media: **14,047 kWh por habitante**.

La intensidad energética media es prácticamente igual entre los clústeres:

```text
Clúster 0: 1.366 kWh por dólar internacional
Clúster 1: 1.363 kWh por dólar internacional
```

Por tanto, la separación se explica principalmente por la composición de la matriz, el consumo por habitante y la demanda eléctrica, no por diferencias relevantes de intensidad energética.

### PCA

Las dos primeras componentes explican:

```text
CP1: 37.11 %
CP2: 23.61 %
CP1 + CP2: 60.72 %
```

La varianza acumulada supera 80 % con cuatro componentes:

```text
CP1 + CP2 + CP3 + CP4 = 86.05 %
```

Interpretación:

- **CP1** contrapone participación fósil frente a participación renovable y baja en carbono.
- **CP2** representa principalmente nivel e intensidad del consumo: energía por habitante, intensidad energética, demanda eléctrica y participación del gas.

La representación bidimensional conserva 60.72 % de la varianza y omite 39.28 %, por lo que debe interpretarse como un resumen parcial.

---

## Casos representativos

Algunos países ilustran diferencias importantes:

- **Qatar**: consumo muy alto y participación fósil cercana a 99.81 %.
- **Islandia**: consumo muy alto, pero participación baja en carbono cercana a 82.08 %.
- **Bangladesh**: consumo bajo y participación fósil cercana a 99.31 %.

Estos casos muestran que el nivel de consumo no determina por sí solo la composición de la matriz energética.

---

## Dataset analítico

La muestra final contiene:

```text
79 países
13 columnas
10 variables de modelado
Año: 2022
Valores faltantes: 0
```

Archivos principales:

```text
data/owid-energy-data.csv
data/owid-energy-codebook.csv
outputs/tablas/dataset_analitico_2022_completo.csv
outputs/tablas/dataset_analitico_2022_estandarizado.csv
outputs/tablas/diccionario_10_variables_modelado.csv
```

Los archivos OWID se vinculan mediante:

```python
codebook["column"]
```

y los nombres de las columnas de `owid-energy-data.csv`.

---

## Criterios de selección de variables

Las diez variables se eligieron por:

1. relevancia para consumo, intensidad y composición energética;
2. variabilidad entre países;
3. interpretabilidad;
4. cobertura suficiente en 2022;
5. compatibilidad con K-means y PCA.

Grupos:

- **Nivel de consumo**:
  - `energy_per_capita`
  - `electricity_demand_per_capita`

- **Intensidad energética**:
  - `energy_per_gdp`

- **Composición general**:
  - `fossil_share_energy`
  - `renewables_share_energy`
  - `nuclear_share_energy`
  - `low_carbon_share_energy`

- **Desagregación fósil**:
  - `coal_share_energy`
  - `oil_share_energy`
  - `gas_share_energy`

Todas las variables se estandarizan antes del modelado.

---

## Estructura del proyecto

```text
data/        Dataset y diccionario de datos
notebooks/   Notebook ejecutado y exportaciones
outputs/     Tablas, gráficas y reportes
src/         Código modular del pipeline
tests/       Pruebas automáticas
run_tests.ps1
requirements.txt
README.md
```

---

## Instalación

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux o macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Ejecución del pipeline

Desde la raíz del proyecto:

```powershell
python -m src.main
```

El pipeline genera las tablas y gráficas en:

```text
outputs/tablas/
outputs/graficas/
```

---

## Ejecución de pruebas

### Opción recomendada en Windows

```powershell
.\run_tests.ps1
```

### Ejecución manual

```powershell
python -m compileall src tests
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.main
```

Resultado esperado:

```text
Ran 12 tests
OK
```

Las pruebas verifican:

- existencia y dimensiones de los archivos OWID;
- dataset analítico de 79 países y 13 columnas;
- ausencia de valores faltantes;
- vínculo con el codebook;
- estandarización;
- reproducibilidad de K-means;
- selección de `k = 2`;
- silueta aproximada de 0.328;
- PCA y varianza acumulada;
- cuatro componentes para superar 80 %;
- generación de tablas y gráficas.

---

## Ejecución del notebook

```powershell
python -m jupyter notebook notebooks\practica_no_supervisado_owid_energy.ipynb
```

Para ejecutar todas las celdas y actualizar las salidas:

```powershell
python -m jupyter nbconvert `
  --to notebook `
  --execute `
  --inplace `
  --ExecutePreprocessor.timeout=900 `
  notebooks\practica_no_supervisado_owid_energy.ipynb
```

---

## Conversión a HTML

```powershell
python -m jupyter nbconvert `
  --to html `
  --execute `
  --ExecutePreprocessor.timeout=900 `
  --output "practica_no_supervisado_owid_energy.html" `
  --output-dir ".\notebooks" `
  ".\notebooks\practica_no_supervisado_owid_energy.ipynb"
```

---

## Privacidad y rutas locales

El notebook utiliza rutas relativas y no debe mostrar rutas absolutas del equipo. Las salidas visibles deben usar referencias como:

```text
data/owid-energy-data.csv
data/owid-energy-codebook.csv
outputs/tablas/
outputs/graficas/
```

No deben aparecer rutas personales como `C:\...` o `D:\...`.

---

## Limitaciones

- El análisis es transversal y corresponde únicamente a 2022.
- La solución K-means tiene separación moderada.
- PCA no genera clústeres ni valida K-means.
- CP1 y CP2 omiten 39.28 % de la varianza.
- Los resultados dependen de las variables seleccionadas, la estandarización y la exclusión de casos incompletos.
- No se establecen relaciones causales.
