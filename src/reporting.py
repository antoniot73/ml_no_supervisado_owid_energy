
"""Tablas e interpretación reproducible de resultados."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


def save_table(table: pd.DataFrame, output_path: Path) -> None:
    """Guarda una tabla CSV sin sobrescritura silenciosa de directorios."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8")
    LOGGER.info("Tabla guardada: %s", output_path)


def build_cluster_profiles(
    data: pd.DataFrame,
    labels: Sequence[int],
    features: Sequence[str],
) -> pd.DataFrame:
    """Calcula centroides interpretables en las unidades originales."""
    if len(data) != len(labels):
        raise ValueError("La cantidad de etiquetas no coincide con las observaciones.")
    profiled = data.loc[:, ["country", *features]].copy()
    profiled["cluster"] = list(labels)
    return profiled.groupby("cluster")[list(features)].mean().reset_index()


def build_country_clusters(
    data: pd.DataFrame,
    labels: Sequence[int],
) -> pd.DataFrame:
    """Construye la tabla país-clúster ordenada."""
    if len(data) != len(labels):
        raise ValueError("La cantidad de etiquetas no coincide con las observaciones.")
    result = data.loc[:, ["country", "iso_code", "year"]].copy()
    result["cluster"] = list(labels)
    return result.sort_values(["cluster", "country"]).reset_index(drop=True)


def summarize_results(
    selected_k: int,
    silhouette: float,
    explained_two: float,
    explained_needed: int,
) -> str:
    """Genera una interpretación breve basada en resultados calculados."""
    return (
        f"K-means seleccionó {selected_k} clústeres porque esta configuración "
        f"obtuvo la mayor silueta promedio ({silhouette:.3f}) entre k=2 y k=8. "
        f"Las dos primeras componentes principales explican {explained_two:.1%} "
        f"de la varianza total; se requieren {explained_needed} componentes para "
        "alcanzar al menos 80% de varianza acumulada. Los resultados describen "
        "similitudes multivariadas entre perfiles energéticos y no constituyen "
        "una clasificación causal ni normativa de los países."
    )


def build_model_comparison(
    selected_k: int,
    silhouette: float,
    explained_two: float,
    components_80: int,
) -> pd.DataFrame:
    """Construye un resumen comparativo entre K-means y PCA.

    Args:
        selected_k: Número de clústeres seleccionado.
        silhouette: Silueta del modelo K-means final.
        explained_two: Varianza acumulada por CP1 y CP2.
        components_80: Componentes requeridas para llegar a 80%.

    Returns:
        Tabla comparativa de objetivos, salidas y resultados.
    """
    return pd.DataFrame(
        [
            {
                "modelo": "K-means",
                "objetivo": "Agrupar países por similitud energética",
                "salida_principal": f"{selected_k} clústeres",
                "criterio_prueba": "Inercia y silueta",
                "resultado_clave": f"Silueta = {silhouette:.3f}",
            },
            {
                "modelo": "PCA",
                "objetivo": "Reducir dimensionalidad y explicar variación",
                "salida_principal": "Componentes principales",
                "criterio_prueba": "Varianza explicada y cargas",
                "resultado_clave": (
                    f"CP1+CP2 = {explained_two:.1%}; "
                    f"{components_80} componentes alcanzan 80%"
                ),
            },
        ]
    )
