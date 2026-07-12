"""Visualizaciones del análisis no supervisado."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)

FIGURE_SCALE = 0.85


def ensure_output_dir(path: Path) -> None:
    """Crea de forma segura un directorio de salida."""
    path.mkdir(parents=True, exist_ok=True)



def plot_kmeans_selection(metrics: pd.DataFrame, output_path: Path) -> None:
    """Grafica inercia y silueta para los valores candidatos de k."""
    required = {"k", "inertia", "silhouette"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"La tabla debe contener: {sorted(required)}")

    ensure_output_dir(output_path.parent)
    selected_row = metrics.loc[metrics["silhouette"].idxmax()]
    selected_k = int(selected_row["k"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13 * FIGURE_SCALE, 5.2 * FIGURE_SCALE),
    )

    axes[0].plot(metrics["k"], metrics["inertia"], marker="o", linewidth=2)
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("Número de clústeres (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_xticks(metrics["k"])
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        metrics["k"],
        metrics["silhouette"],
        marker="o",
        linewidth=2,
    )
    axes[1].axvline(selected_k, linestyle="--", linewidth=1)
    axes[1].annotate(
        f"k = {selected_k}\nSilueta = {selected_row['silhouette']:.3f}",
        xy=(selected_k, selected_row["silhouette"]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.85},
    )
    axes[1].set_title("Coeficiente de silueta")
    axes[1].set_xlabel("Número de clústeres (k)")
    axes[1].set_ylabel("Silueta promedio")
    axes[1].set_xticks(metrics["k"])
    axes[1].grid(alpha=0.3)

    fig.tight_layout(pad=1.6)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)




def plot_kmeans_clusters(
    data: pd.DataFrame,
    labels: Sequence[int],
    output_path: Path,
) -> None:
    """Visualiza los perfiles K-means con una escala legible."""
    required = {"energy_per_capita", "fossil_share_energy"}
    if not required.issubset(data.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(data.columns))}")
    if len(data) != len(labels):
        raise ValueError("Las etiquetas no coinciden con las observaciones.")

    ensure_output_dir(output_path.parent)
    plot_data = data.loc[:, ["energy_per_capita", "fossil_share_energy"]].copy()
    plot_data["cluster"] = list(labels)

    fig, ax = plt.subplots(figsize=(12.0, 7.2))

    for cluster_id, group in plot_data.groupby("cluster", sort=True):
        ax.scatter(
            group["energy_per_capita"],
            group["fossil_share_energy"],
            s=58,
            alpha=0.72,
            label=f"Clúster {cluster_id}",
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xscale("log")
    ax.set_xlim(2500, 230000)
    ax.set_xticks([3000, 5000, 10000, 25000, 50000, 100000, 200000])
    ax.set_xticklabels(["3 mil", "5 mil", "10 mil", "25 mil", "50 mil", "100 mil", "200 mil"])
    ax.set_title("Perfiles energéticos identificados por K-means", pad=14)
    ax.set_xlabel("Consumo de energía por habitante (kWh/persona, escala logarítmica)")
    ax.set_ylabel("Participación de combustibles fósiles (%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(
        title="Grupo energético",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout(pad=1.8)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)

def plot_pca_variance(explained: pd.DataFrame, output_path: Path) -> None:
    """Grafica varianza individual y acumulada de PCA."""
    ensure_output_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10 * FIGURE_SCALE, 5.8 * FIGURE_SCALE))
    positions = range(len(explained))

    ax.bar(
        positions,
        explained["varianza_explicada"],
        alpha=0.78,
        label="Varianza individual",
    )
    ax.plot(
        positions,
        explained["varianza_acumulada"],
        marker="o",
        linewidth=2,
        label="Varianza acumulada",
    )
    ax.axhline(0.80, linestyle="--", linewidth=1.2, label="Umbral 80 %")
    ax.set_title("Varianza explicada por PCA")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Proporción de varianza")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(explained["componente"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout(pad=1.4)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)



def plot_pca_scores(scores: pd.DataFrame, output_path: Path) -> None:
    """Representa países en CP1 y CP2 con etiquetas selectivas no solapadas."""
    required = {"country", "CP1", "CP2"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(scores.columns))}")

    ensure_output_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(12.0, 8.0))

    ax.scatter(
        scores["CP1"],
        scores["CP2"],
        s=46,
        alpha=0.62,
        edgecolors="white",
        linewidths=0.5,
    )

    distance = (scores["CP1"] ** 2 + scores["CP2"] ** 2) ** 0.5
    label_indices = distance.nlargest(min(9, len(scores))).index
    offsets = [
        (8, 10), (8, -14), (-58, 10), (-62, -14), (10, 18),
        (10, -20), (-68, 18), (-68, -20), (12, 4),
    ]

    for offset, index in zip(offsets, label_indices):
        row = scores.loc[index]
        ax.annotate(
            row["country"],
            xy=(row["CP1"], row["CP2"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.6, "alpha": 0.65},
            bbox={"boxstyle": "round,pad=0.18", "alpha": 0.78},
        )

    ax.axhline(0, linewidth=0.8, alpha=0.55)
    ax.axvline(0, linewidth=0.8, alpha=0.55)
    ax.set_title("Países proyectados en las dos primeras componentes", pad=14)
    ax.set_xlabel("CP1: transición entre perfil fósil y bajo en carbono")
    ax.set_ylabel("CP2: nivel de consumo e intensidad energética")
    ax.grid(alpha=0.22)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout(pad=1.8)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)



def plot_pca_loadings(loadings: pd.DataFrame, output_path: Path) -> None:
    """Grafica cargas de CP1 y CP2 sin superponer leyenda ni etiquetas."""
    ensure_output_dir(output_path.parent)

    labels = {
        "energy_per_capita": "Energía por habitante",
        "energy_per_gdp": "Intensidad energética",
        "fossil_share_energy": "Participación fósil",
        "renewables_share_energy": "Participación renovable",
        "nuclear_share_energy": "Participación nuclear",
        "coal_share_energy": "Participación de carbón",
        "oil_share_energy": "Participación de petróleo",
        "gas_share_energy": "Participación de gas",
        "low_carbon_share_energy": "Participación baja en carbono",
        "electricity_demand_per_capita": "Demanda eléctrica por habitante",
    }

    table = loadings.set_index("variable")[["CP1", "CP2"]].copy()
    table.index = [labels.get(name, name) for name in table.index]
    table = table.sort_values("CP1")

    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    table.plot(kind="barh", ax=ax, width=0.72)

    ax.axvline(0, linewidth=0.9)
    ax.set_title("Contribución de las variables a CP1 y CP2", pad=14)
    ax.set_xlabel("Carga factorial")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.28)
    ax.legend(
        title="Componente",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(bottom=0.16, left=0.30)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)

def plot_dataset_coverage(data: pd.DataFrame, output_path: Path) -> None:
    """Visualiza cobertura temporal y completitud global del dataset OWID.

    Args:
        data: Dataset completo de OWID Energy.
        output_path: Ruta de salida de la imagen.

    Raises:
        ValueError: Si faltan las columnas básicas `year` o `country`.
    """
    required = {"year", "country"}
    if not required.issubset(data.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(data.columns))}")

    ensure_output_dir(output_path.parent)

    yearly_entities = (
        data.groupby("year", dropna=True)["country"]
        .nunique()
        .sort_index()
    )
    completeness = (
        data.notna()
        .mean()
        .mul(100)
        .sort_values(ascending=True)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14 * FIGURE_SCALE, 5.5 * FIGURE_SCALE),
    )

    axes[0].plot(yearly_entities.index, yearly_entities.values)
    axes[0].set_title("Cobertura temporal del dataset completo")
    axes[0].set_xlabel("Año")
    axes[0].set_ylabel("Entidades con registros")
    axes[0].grid(alpha=0.3)

    axes[1].barh(
        completeness.index,
        completeness.values,
    )
    axes[1].set_title("Completitud de las 130 variables")
    axes[1].set_xlabel("Datos disponibles (%)")
    axes[1].tick_params(axis="y", labelsize=3.2)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)


def plot_feature_distributions(
    data: pd.DataFrame,
    features: Sequence[str],
    output_path: Path,
) -> None:
    """Visualiza la distribución de todas las variables analíticas.

    Args:
        data: Muestra analítica.
        features: Variables numéricas seleccionadas.
        output_path: Ruta de salida de la imagen.
    """
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(f"Faltan variables para graficar: {missing}")

    ensure_output_dir(output_path.parent)
    rows = 5
    columns = 2
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12 * FIGURE_SCALE, 16 * FIGURE_SCALE),
    )

    for ax, feature in zip(axes.flatten(), features):
        values = data[feature].dropna()
        ax.hist(values, bins=15, alpha=0.8)
        ax.set_title(feature, fontsize=8)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Distribución de las variables usadas por K-means y PCA",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)


def plot_feature_correlation(
    data: pd.DataFrame,
    features: Sequence[str],
    output_path: Path,
) -> None:
    """Genera una matriz visual de correlaciones entre variables analíticas.

    Args:
        data: Muestra analítica.
        features: Variables numéricas seleccionadas.
        output_path: Ruta de salida de la imagen.
    """
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(f"Faltan variables para correlación: {missing}")

    ensure_output_dir(output_path.parent)
    correlation = data[list(features)].corr()

    fig, ax = plt.subplots(
        figsize=(10 * FIGURE_SCALE, 8 * FIGURE_SCALE),
    )
    image = ax.imshow(correlation.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_yticklabels(features, fontsize=6)
    ax.set_title("Correlación entre variables analíticas")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gráfica guardada: %s", output_path)
