"""Ejecución completa de K-means y PCA con datasets independientes."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .dataset import (
    dataset_summary,
    load_energy_codebook,
    load_energy_data,
    select_country_year,
    select_feature_metadata,
)
from .modeling import run_kmeans_analysis, run_pca_analysis
from .reporting import (
    build_cluster_profiles,
    build_country_clusters,
    build_model_comparison,
    save_table,
    summarize_results,
)
from .visualization import (
    plot_dataset_coverage,
    plot_feature_correlation,
    plot_feature_distributions,
    plot_kmeans_clusters,
    plot_kmeans_selection,
    plot_pca_loadings,
    plot_pca_scores,
    plot_pca_variance,
)

FEATURES = [
    "energy_per_capita",
    "energy_per_gdp",
    "fossil_share_energy",
    "renewables_share_energy",
    "nuclear_share_energy",
    "coal_share_energy",
    "oil_share_energy",
    "gas_share_energy",
    "low_carbon_share_energy",
    "electricity_demand_per_capita",
]
YEAR = 2022
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configura una bitácora uniforme para la ejecución."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Ejecuta carga, validación, ambos modelos y persistencia."""
    configure_logging()
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "owid-energy-data.csv"
    codebook_path = project_root / "data" / "owid-energy-codebook.csv"
    graph_dir = project_root / "outputs" / "graficas"
    table_dir = project_root / "outputs" / "tablas"

    try:
        raw = load_energy_data(data_path)
        codebook = load_energy_codebook(codebook_path)
        feature_metadata = select_feature_metadata(codebook, raw, FEATURES)
        analytical = select_country_year(raw, YEAR, FEATURES)

        kmeans_data = analytical.copy(deep=True)
        pca_data = analytical.copy(deep=True)

        kmeans = run_kmeans_analysis(
            kmeans_data,
            FEATURES,
            selected_k=None,
            random_state=RANDOM_STATE,
        )
        pca = run_pca_analysis(pca_data, FEATURES)

        selected_row = kmeans.metrics.loc[
            kmeans.metrics["k"] == kmeans.selected_k
        ].iloc[0]
        explained_two = float(pca.explained_variance.loc[1, "varianza_acumulada"])
        components_80 = int(
            (pca.explained_variance["varianza_acumulada"] < 0.80).sum() + 1
        )

        profiles = build_cluster_profiles(kmeans_data, kmeans.labels, FEATURES)
        assignments = build_country_clusters(kmeans_data, kmeans.labels)
        comparison = build_model_comparison(
            kmeans.selected_k,
            float(selected_row["silhouette"]),
            explained_two,
            components_80,
        )

        completeness = (
            raw.notna()
            .mean()
            .mul(100)
            .rename("porcentaje_disponible")
            .reset_index()
            .rename(columns={"index": "variable"})
            .sort_values("porcentaje_disponible", ascending=False)
        )
        analytical_scaled = analytical[["country", "iso_code", "year"]].copy()
        analytical_scaled.loc[:, FEATURES] = kmeans.scaled_data

        save_table(completeness, table_dir / "completitud_dataset_completo.csv")
        save_table(
            analytical,
            table_dir / "dataset_analitico_2022_completo.csv",
        )
        save_table(
            analytical_scaled,
            table_dir / "dataset_analitico_2022_estandarizado.csv",
        )
        save_table(
            feature_metadata,
            table_dir / "diccionario_10_variables_modelado.csv",
        )
        save_table(dataset_summary(analytical, FEATURES), table_dir / "resumen_dataset.csv")
        save_table(feature_metadata, table_dir / "diccionario_variables_seleccionadas.csv")
        save_table(kmeans.metrics, table_dir / "metricas_kmeans.csv")
        save_table(assignments, table_dir / "asignacion_paises_cluster.csv")
        save_table(profiles, table_dir / "perfiles_centroides.csv")
        save_table(pca.explained_variance, table_dir / "varianza_explicada_pca.csv")
        save_table(pca.loadings, table_dir / "cargas_pca.csv")
        save_table(pca.scores, table_dir / "puntuaciones_pca.csv")
        save_table(comparison, table_dir / "comparacion_modelos.csv")

        plot_dataset_coverage(raw, graph_dir / "exploracion_dataset_completo.png")
        plot_feature_distributions(
            analytical,
            FEATURES,
            graph_dir / "distribuciones_variables_analiticas.png",
        )
        plot_feature_correlation(
            analytical,
            FEATURES,
            graph_dir / "correlacion_variables_analiticas.png",
        )
        plot_kmeans_selection(kmeans.metrics, graph_dir / "seleccion_k_kmeans.png")
        plot_kmeans_clusters(kmeans_data, kmeans.labels, graph_dir / "clusters_kmeans.png")
        plot_pca_variance(pca.explained_variance, graph_dir / "varianza_explicada_pca.png")
        plot_pca_loadings(pca.loadings, graph_dir / "cargas_pca.png")
        plot_pca_scores(pca.scores, graph_dir / "proyeccion_pca.png")

        print("\nPIPELINE FINALIZADO")
        print(f"Países analizados: {len(analytical)}")
        print(f"Variables documentadas con codebook: {len(feature_metadata)}")
        print(
            summarize_results(
                kmeans.selected_k,
                float(selected_row["silhouette"]),
                explained_two,
                components_80,
            )
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.exception("La ejecución finalizó con error controlado: %s", exc)
        raise


if __name__ == "__main__":
    main()
