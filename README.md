# Análisis de perfiles energéticos mediante K-means y PCA sobre OWID Energy

**Repositorio GitHub:**
https://github.com/antoniot73/ml_no_supervisado_owid_energy

**URL Binder:**
https://mybinder.org/v2/gh/antoniot73/ml_no_supervisado_owid_energy/main?filepath=notebooks/practica_no_supervisado_owid_energy.ipynb

**GitHub Page:**
https://antoniot73.github.io/ml_no_supervisado_owid_energy/notebooks/practica_no_supervisado_owid_energy.html

**Dataset utilizado:**
https://github.com/owid/energy-data

---

# Instituto Internacional de Aguascalientes

**Maestría en Inteligencia Artificial para la Transformación Digital**  
**Programa:** Aprendizaje Inteligente  
**Alumno:** Antonio Nicolás Toro González  
**Tutor:** Dr. Francisco Javier Luna Rosas  

---

# Descripción

Este repositorio contiene la práctica **“Análisis, implementación y evaluación de modelos de aprendizaje no supervisado”**, desarrollada mediante dos técnicas complementarias:

- **K-means**, para identificar grupos de países con perfiles energéticos semejantes.
- **PCA**, para reducir dimensionalidad e interpretar las combinaciones de variables que explican las principales diferencias entre países.

Los modelos se aplican al dataset **OWID Energy** de **Our World in Data**, utilizando información correspondiente al año **2022**.

La práctica integra:

- validación de archivos;
- comprensión del dataset y del diccionario de variables;
- selección justificada de indicadores;
- construcción de una muestra analítica común;
- estandarización independiente para cada modelo;
- evaluación de K-means;
- análisis de varianza y cargas de PCA;
- visualización;
- interpretación energética;
- comparación de modelos;
- exportación de resultados;
- pruebas automáticas;
- publicación mediante GitHub Pages.

---

# Objetivo general

Identificar perfiles energéticos entre países y determinar qué variables explican las principales diferencias de sus sistemas energéticos mediante K-means y Análisis de Componentes Principales.

---

# Objetivos específicos

- Validar automáticamente los archivos de observaciones y el codebook.
- Construir una muestra transversal comparable para 2022.
- Seleccionar variables representativas de consumo, intensidad y composición energética.
- Estandarizar las variables antes de cada modelo.
- Evaluar distintos valores de `k` para K-means.
- Seleccionar el número de clústeres mediante el coeficiente de silueta.
- Interpretar los centroides de los grupos.
- Reducir dimensionalidad mediante PCA.
- Analizar la varianza explicada individual y acumulada.
- Interpretar las cargas de CP1 y CP2.
- Comparar los hallazgos de K-means y PCA.
- Generar tablas y gráficas reproducibles.
- Validar el pipeline mediante pruebas automáticas.

---

# Pregunta de investigación

> ¿Qué perfiles energéticos pueden identificarse entre los países y qué variables explican las principales diferencias de sus sistemas energéticos?

La pregunta se divide en dos partes:

1. **Identificación de perfiles:** determinar si los países pueden agruparse según su consumo de energía por habitante, intensidad energética, demanda eléctrica y composición de sus fuentes.
2. **Explicación de diferencias:** identificar qué combinaciones de variables explican la mayor parte de la variación observada entre los sistemas energéticos nacionales.

---

# Dataset

**Nombre:** OWID Energy Dataset  
**Fuente:** Our World in Data  
**Año analizado:** 2022

Archivos utilizados:

```text
data/owid-energy-data.csv
data/owid-energy-codebook.csv
```

Dimensiones de las fuentes:

```text
Dataset original: 23,377 filas × 130 columnas
Codebook: 130 variables documentadas
```

Muestra analítica final:

```text
79 países
13 columnas
10 variables de modelado
Año: 2022
Valores faltantes: 0
```

Los archivos se vinculan mediante:

```python
codebook["column"]
```

y los nombres de las columnas de `owid-energy-data.csv`.

---

# Variables utilizadas

Las diez variables se seleccionaron por:

1. relevancia sustantiva;
2. variabilidad entre países;
3. interpretabilidad;
4. cobertura suficiente en 2022;
5. compatibilidad con K-means y PCA.

## Nivel de consumo

- `energy_per_capita`
- `electricity_demand_per_capita`

## Intensidad energética

- `energy_per_gdp`

## Composición general

- `fossil_share_energy`
- `renewables_share_energy`
- `nuclear_share_energy`
- `low_carbon_share_energy`

## Desagregación fósil

- `coal_share_energy`
- `oil_share_energy`
- `gas_share_energy`

Todas las variables se estandarizan antes del modelado.

---

# Tecnologías utilizadas

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- pathlib
- logging
- unittest
- nbconvert
- GitHub Actions
- GitHub Pages
- Binder

---

# Metodología

## K-means

Se evaluaron valores de `k` entre 2 y 8.

La selección se realizó mediante:

- inercia;
- coeficiente de silueta;
- reproducibilidad con `random_state = 42`.

## PCA

PCA se aplicó sobre una copia independiente del dataset analítico y con escalamiento propio.

Se analizaron:

- varianza explicada individual;
- varianza acumulada;
- cargas de las variables;
- proyección de países en CP1 y CP2.

---

# Resultados principales

## K-means

La mayor silueta promedio se obtuvo con:

```text
k = 2
silueta ≈ 0.328
```

La solución debe interpretarse como descriptiva, porque la separación es moderada.

### Perfil fósil dominante

- consumo medio aproximado: **39,425 kWh por habitante**;
- participación fósil media: **86.1 %**;
- participación baja en carbono: **13.9 %**;
- demanda eléctrica media: **5,556 kWh por habitante**.

### Perfil de mayor consumo con más fuentes bajas en carbono

- consumo medio aproximado: **60,362 kWh por habitante**;
- participación fósil media: **48.8 %**;
- participación baja en carbono: **51.2 %**;
- demanda eléctrica media: **14,047 kWh por habitante**.

La intensidad energética media es prácticamente igual entre los clústeres:

```text
Clúster 0: 1.366 kWh por dólar internacional
Clúster 1: 1.363 kWh por dólar internacional
```

Por tanto, la separación está asociada principalmente con:

- composición de la matriz;
- consumo por habitante;
- demanda eléctrica.

## PCA

```text
CP1: 37.11 %
CP2: 23.61 %
CP1 + CP2: 60.72 %
CP1 + CP2 + CP3 + CP4: 86.05 %
```

Interpretación:

- **CP1** contrapone participación fósil frente a participación renovable y baja en carbono.
- **CP2** representa principalmente nivel e intensidad del consumo: energía por habitante, intensidad energética, demanda eléctrica y participación del gas.

La proyección bidimensional omite **39.28 %** de la varianza, por lo que debe interpretarse como un resumen parcial.

---

# Casos representativos

- **Qatar**: consumo muy alto y participación fósil cercana a 99.81 %.
- **Islandia**: consumo muy alto, pero participación baja en carbono cercana a 82.08 %.
- **Bangladesh**: consumo bajo y participación fósil cercana a 99.31 %.

Estos casos muestran que el nivel de consumo no determina por sí solo la composición de la matriz energética.

---

# Interpretación general

K-means y PCA responden preguntas distintas pero complementarias:

- K-means identifica **qué países presentan perfiles semejantes**.
- PCA explica **qué variables generan las principales diferencias**.

La conclusión principal es que el consumo energético por habitante no puede analizarse de forma aislada. Dos países pueden consumir cantidades similares de energía y, aun así, tener matrices energéticas completamente diferentes.

La composición fósil frente a baja en carbono, el consumo por habitante, la intensidad energética y la demanda eléctrica son las dimensiones más relevantes para interpretar los perfiles energéticos observados.

---

# Resultados generados

La ejecución automática produce:

- notebook ejecutable;
- reporte HTML;
- dataset analítico completo;
- dataset estandarizado;
- métricas de K-means;
- asignación de países a clústeres;
- perfiles de centroides;
- varianza explicada de PCA;
- cargas de PCA;
- puntuaciones PCA;
- tabla comparativa de modelos;
- gráficas de exploración, clústeres y componentes.

---

# Archivos generados

```text
outputs/
├── graficas/
└── tablas/
```

Tablas principales:

```text
outputs/tablas/dataset_analitico_2022_completo.csv
outputs/tablas/dataset_analitico_2022_estandarizado.csv
outputs/tablas/diccionario_10_variables_modelado.csv
outputs/tablas/metricas_kmeans.csv
outputs/tablas/asignacion_paises_cluster.csv
outputs/tablas/perfiles_centroides.csv
outputs/tablas/varianza_explicada_pca.csv
outputs/tablas/cargas_pca.csv
outputs/tablas/puntuaciones_pca.csv
outputs/tablas/comparacion_modelos.csv
```

Gráficas principales:

```text
outputs/graficas/seleccion_k_kmeans.png
outputs/graficas/clusters_kmeans.png
outputs/graficas/varianza_explicada_pca.png
outputs/graficas/cargas_pca.png
outputs/graficas/proyeccion_pca.png
```

---

# Estructura del repositorio

```text
ml_no_supervisado_owid_energy/
│
├── .github/
│   └── workflows/
├── data/
├── notebooks/
├── outputs/
│   ├── graficas/
│   └── tablas/
├── src/
├── tests/
├── README.md
├── requirements.txt
├── run_tests.ps1
├── run_tests.sh
└── .gitignore
```

---

# Instalación

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Linux o macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Ejecución local

Desde la raíz del proyecto:

```powershell
python -m src.main
```

Notebook:

```powershell
python -m jupyter notebook notebooks\practica_no_supervisado_owid_energy.ipynb
```

HTML:

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

# Ejecución en Binder

https://mybinder.org/v2/gh/antoniot73/ml_no_supervisado_owid_energy/main?filepath=notebooks/practica_no_supervisado_owid_energy.ipynb

---

# GitHub Pages

Página principal:

https://antoniot73.github.io/ml_no_supervisado_owid_energy/notebooks/practica_no_supervisado_owid_energy.html


---

# Pruebas automáticas

## Windows

```powershell
.\run_tests.ps1
```

## Linux o macOS

```bash
./run_tests.sh
```

## Ejecución manual

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

# Reproducibilidad

El proyecto utiliza:

- `random_state = 42`;
- rutas relativas;
- validación automática de archivos;
- verificación de valores faltantes;
- copias independientes del dataset para K-means y PCA;
- escalamiento independiente para cada modelo;
- exportación automática de tablas y gráficas;
- pruebas automáticas;
- notebook ejecutable;
- integración continua mediante GitHub Actions.

---

# Privacidad y rutas locales

El notebook utiliza rutas relativas y no debe mostrar rutas absolutas del equipo.

Ejemplos permitidos:

```text
data/owid-energy-data.csv
data/owid-energy-codebook.csv
outputs/tablas/
outputs/graficas/
```

No deben aparecer rutas personales como `C:\...` o `D:\...`.

---

# Limitaciones

- El análisis es transversal y corresponde únicamente a 2022.
- La solución K-means tiene separación moderada.
- PCA no genera clústeres ni valida K-means.
- CP1 y CP2 omiten 39.28 % de la varianza.
- Los resultados dependen de las variables seleccionadas, la estandarización y la exclusión de casos incompletos.
- No se establecen relaciones causales.

---

# Referencias

OGéron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow (2.ª ed.). O’Reilly Media.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning: With applications in R. Springer.

Our World in Data. (2026). Energy. https://ourworldindata.org/energy

Our World in Data. (2026). OWID energy data [Conjunto de datos]. GitHub. https://github.com/owid/energy-data

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.