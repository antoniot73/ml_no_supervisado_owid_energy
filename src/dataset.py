
"""Carga, validación y preparación del dataset energético de OWID."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_file_path(path: Path) -> None:
    """Valida que una ruta exista, sea archivo y tenga extensión CSV.

    Args:
        path: Ruta del archivo que se desea validar.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la ruta no es un archivo CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    if not path.is_file():
        raise ValueError(f"La ruta no corresponde a un archivo: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Se esperaba un archivo CSV: {path}")


def load_energy_data(path: Path) -> pd.DataFrame:
    """Carga el dataset energético desde un CSV validado.

    Args:
        path: Ruta al archivo `owid-energy-data.csv`.

    Returns:
        DataFrame con el dataset completo.

    Raises:
        ValueError: Si el archivo está vacío o no contiene columnas.
    """
    validate_file_path(path)
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"No fue posible leer el dataset: {exc}") from exc

    if data.empty or data.shape[1] == 0:
        raise ValueError("El dataset está vacío o no contiene columnas.")
    LOGGER.info("Dataset cargado: %s filas y %s columnas.", *data.shape)
    return data


def load_energy_codebook(path: Path) -> pd.DataFrame:
    """Carga y valida el diccionario de variables de OWID Energy.

    Args:
        path: Ruta al archivo `owid-energy-codebook.csv`.

    Returns:
        DataFrame con nombre, título, descripción, unidad y fuente de cada variable.

    Raises:
        ValueError: Si el archivo está vacío o no contiene las columnas obligatorias.
    """
    validate_file_path(path)
    try:
        codebook = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"No fue posible leer el codebook: {exc}") from exc

    required = ["column", "title", "description", "unit", "source"]
    if codebook.empty:
        raise ValueError("El codebook está vacío.")
    validate_columns(codebook, required)
    LOGGER.info("Codebook cargado: %s variables documentadas.", len(codebook))
    return codebook


def select_feature_metadata(
    codebook: pd.DataFrame,
    data: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Vincula las variables analíticas con su definición, unidad y fuente.

    La relación se establece entre `codebook["column"]` y los nombres de
    columnas de `owid-energy-data.csv`.

    Args:
        codebook: Diccionario de variables de OWID.
        data: Dataset energético principal.
        features: Variables utilizadas por K-means y PCA.

    Returns:
        Tabla documental de las variables seleccionadas.

    Raises:
        ValueError: Si una variable no existe en los datos o no está documentada.
    """
    validate_columns(data, features)
    validate_columns(
        codebook,
        ["column", "title", "description", "unit", "source"],
    )

    metadata = codebook.loc[
        codebook["column"].isin(features),
        ["column", "title", "description", "unit", "source"],
    ].copy()

    documented = set(metadata["column"])
    missing_metadata = [feature for feature in features if feature not in documented]
    if missing_metadata:
        raise ValueError(
            "Variables sin documentación en el codebook: "
            f"{missing_metadata}"
        )

    order = {feature: index for index, feature in enumerate(features)}
    metadata["_order"] = metadata["column"].map(order)
    metadata = (
        metadata.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Vinculación data-codebook validada para %s variables.",
        len(metadata),
    )
    return metadata


def validate_columns(data: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Comprueba que todas las columnas requeridas estén disponibles.

    Args:
        data: DataFrame a validar.
        required_columns: Nombres de columnas obligatorias.

    Raises:
        ValueError: Si faltan columnas.
    """
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def select_country_year(
    data: pd.DataFrame,
    year: int,
    features: Sequence[str],
) -> pd.DataFrame:
    """Selecciona países ISO para un año y conserva casos completos.

    Args:
        data: Dataset energético completo.
        year: Año transversal de análisis.
        features: Indicadores numéricos seleccionados.

    Returns:
        DataFrame con país, código ISO, año e indicadores completos.

    Raises:
        ValueError: Si el año es inválido o la muestra resultante es insuficiente.
    """
    if not isinstance(year, int) or year < 1900:
        raise ValueError("El año debe ser un entero válido mayor o igual a 1900.")

    required = ["country", "iso_code", "year", *features]
    validate_columns(data, required)

    selected = data.loc[
        (data["year"] == year) & data["iso_code"].notna(),
        required,
    ].copy()

    for feature in features:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
        selected.loc[~np.isfinite(selected[feature]), feature] = np.nan

    initial_rows = len(selected)
    selected = selected.dropna(subset=list(features)).reset_index(drop=True)
    removed_rows = initial_rows - len(selected)

    if len(selected) < 20:
        raise ValueError(
            f"La muestra completa para {year} es insuficiente: {len(selected)} países."
        )

    LOGGER.info(
        "Muestra transversal %s: %s países; %s filas incompletas excluidas.",
        year,
        len(selected),
        removed_rows,
    )
    return selected


def dataset_summary(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Genera un resumen descriptivo reproducible de las variables.

    Args:
        data: Muestra analítica.
        features: Variables numéricas que se resumirán.

    Returns:
        Tabla transpuesta con estadísticos descriptivos.
    """
    validate_columns(data, features)
    summary = data.loc[:, list(features)].describe().T.reset_index()
    return summary.rename(columns={"index": "variable"})
