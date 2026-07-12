"""Modelos no supervisados K-means y PCA para perfiles energéticos."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KMeansResults:
    """Resultados inmutables del análisis K-means."""

    scaled_data: np.ndarray
    labels: np.ndarray
    centroids_scaled: np.ndarray
    metrics: pd.DataFrame
    selected_k: int


@dataclass(frozen=True)
class PCAResults:
    """Resultados inmutables del análisis de componentes principales."""

    scaled_data: np.ndarray
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame


def scale_features(
    data: pd.DataFrame,
    features: Sequence[str],
) -> tuple[StandardScaler, np.ndarray]:
    """Estandariza las variables seleccionadas.

    Args:
        data: Muestra analítica.
        features: Variables numéricas que se transformarán.

    Returns:
        Escalador ajustado y matriz estandarizada.

    Raises:
        ValueError: Si la matriz no es válida o contiene valores no finitos.
    """
    if data.empty:
        raise ValueError("La muestra analítica está vacía.")
    matrix = data.loc[:, list(features)].to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(features):
        raise ValueError("La matriz de características no tiene la forma esperada.")
    if not np.isfinite(matrix).all():
        raise ValueError("La matriz contiene valores faltantes o no finitos.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    LOGGER.info("Estandarización completada para %s variables.", len(features))
    return scaler, scaled


def evaluate_kmeans_range(
    scaled_data: np.ndarray,
    k_values: Sequence[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Evalúa configuraciones de K-means mediante inercia y silueta."""
    if scaled_data.ndim != 2 or scaled_data.shape[0] < 3:
        raise ValueError("La matriz debe ser bidimensional y contener al menos tres filas.")

    rows: list[dict[str, float | int]] = []
    for k in k_values:
        if k < 2 or k >= scaled_data.shape[0]:
            raise ValueError(f"Valor de k inválido: {k}")
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=30,
            max_iter=500,
            random_state=random_state,
        )
        with threadpool_limits(limits=1):
            labels = model.fit_predict(scaled_data)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_data, labels)),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    LOGGER.info("K-means evaluado para k=%s.", list(k_values))
    return metrics


def run_kmeans_analysis(
    data: pd.DataFrame,
    features: Sequence[str],
    selected_k: int | None = None,
    random_state: int = 42,
) -> KMeansResults:
    """Ejecuta K-means con su propia copia del dataset y preprocesamiento.

    Args:
        data: Dataset dedicado al modelo K-means.
        features: Variables analíticas.
        selected_k: Número final de grupos; si es None, maximiza silueta.
        random_state: Semilla reproducible.

    Returns:
        Resultados de evaluación y ajuste del modelo.
    """
    kmeans_data = data.copy(deep=True)
    _, scaled = scale_features(kmeans_data, features)
    metrics = evaluate_kmeans_range(scaled, range(2, 9), random_state)

    if selected_k is None:
        selected_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    elif selected_k not in metrics["k"].tolist():
        raise ValueError("selected_k debe pertenecer al intervalo 2..8.")

    model = KMeans(
        n_clusters=selected_k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=random_state,
    )
    with threadpool_limits(limits=1):
        labels = model.fit_predict(scaled)

    LOGGER.info("K-means final ajustado con k=%s.", selected_k)
    return KMeansResults(
        scaled_data=scaled,
        labels=labels,
        centroids_scaled=model.cluster_centers_,
        metrics=metrics,
        selected_k=selected_k,
    )


def run_pca_analysis(
    data: pd.DataFrame,
    features: Sequence[str],
) -> PCAResults:
    """Ejecuta PCA con su propia copia del dataset y preprocesamiento.

    Args:
        data: Dataset dedicado al modelo PCA.
        features: Variables analíticas.

    Returns:
        Puntuaciones, cargas y varianza explicada.
    """
    pca_data = data.copy(deep=True)
    _, scaled = scale_features(pca_data, features)

    model = PCA()
    scores_array = model.fit_transform(scaled)
    component_names = [f"CP{i}" for i in range(1, len(features) + 1)]

    scores = pd.DataFrame(scores_array, columns=component_names)
    scores.insert(0, "country", pca_data["country"].tolist())

    loadings = (
        pd.DataFrame(
            model.components_.T,
            index=list(features),
            columns=component_names,
        )
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    explained = pd.DataFrame(
        {
            "componente": component_names,
            "varianza_explicada": model.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(model.explained_variance_ratio_),
        }
    )
    LOGGER.info(
        "PCA ajustado: CP1 y CP2 explican %.2f%% de la varianza.",
        100 * float(explained.loc[1, "varianza_acumulada"]),
    )
    return PCAResults(
        scaled_data=scaled,
        scores=scores,
        loadings=loadings,
        explained_variance=explained,
    )
