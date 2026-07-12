
"""Pruebas de aceptación del pipeline K-means y PCA."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import (
    load_energy_codebook,
    load_energy_data,
    select_country_year,
    select_feature_metadata,
)
from src.main import FEATURES, RANDOM_STATE, YEAR
from src.modeling import run_kmeans_analysis, run_pca_analysis


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "owid-energy-data.csv"
CODEBOOK_PATH = ROOT / "data" / "owid-energy-codebook.csv"
TABLE_DIR = ROOT / "outputs" / "tablas"
GRAPH_DIR = ROOT / "outputs" / "graficas"


class TestDataset(unittest.TestCase):
    """Valida fuentes, selección transversal y trazabilidad."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_energy_data(DATA_PATH)
        cls.codebook = load_energy_codebook(CODEBOOK_PATH)
        cls.analytical = select_country_year(cls.raw, YEAR, FEATURES)
        cls.metadata = select_feature_metadata(
            cls.codebook,
            cls.raw,
            FEATURES,
        )

    def test_archivos_fuente(self) -> None:
        self.assertTrue(DATA_PATH.is_file())
        self.assertTrue(CODEBOOK_PATH.is_file())
        self.assertGreater(DATA_PATH.stat().st_size, 0)
        self.assertGreater(CODEBOOK_PATH.stat().st_size, 0)

    def test_dimensiones_fuentes(self) -> None:
        self.assertEqual(self.raw.shape, (23377, 130))
        self.assertEqual(len(self.codebook), 130)

    def test_dataset_analitico_completo(self) -> None:
        self.assertEqual(self.analytical.shape, (79, 13))
        self.assertEqual(self.analytical["country"].nunique(), 79)
        self.assertTrue(self.analytical["iso_code"].notna().all())
        self.assertTrue((self.analytical["year"] == 2022).all())
        self.assertEqual(int(self.analytical[FEATURES].isna().sum().sum()), 0)

    def test_diccionario_variables(self) -> None:
        self.assertEqual(len(self.metadata), 10)
        self.assertEqual(self.metadata["column"].tolist(), FEATURES)
        self.assertTrue(
            self.metadata[["title", "description", "unit", "source"]]
            .notna()
            .all()
            .all()
        )


class TestModels(unittest.TestCase):
    """Valida K-means, PCA y reproducibilidad."""

    @classmethod
    def setUpClass(cls) -> None:
        raw = load_energy_data(DATA_PATH)
        cls.data = select_country_year(raw, YEAR, FEATURES)
        cls.kmeans = run_kmeans_analysis(
            cls.data,
            FEATURES,
            selected_k=None,
            random_state=RANDOM_STATE,
        )
        cls.pca = run_pca_analysis(cls.data, FEATURES)

    def test_estandarizacion(self) -> None:
        means = self.kmeans.scaled_data.mean(axis=0)
        stds = self.kmeans.scaled_data.std(axis=0)
        self.assertTrue(np.allclose(means, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(stds, 1.0, atol=1e-10))

    def test_kmeans(self) -> None:
        self.assertEqual(self.kmeans.selected_k, 2)
        self.assertEqual(len(self.kmeans.labels), 79)
        self.assertEqual(len(np.unique(self.kmeans.labels)), 2)
        selected = self.kmeans.metrics.loc[
            self.kmeans.metrics["k"] == self.kmeans.selected_k
        ].iloc[0]
        self.assertTrue(math.isclose(
            float(selected["silhouette"]),
            0.328,
            abs_tol=0.002,
        ))

    def test_kmeans_reproducible(self) -> None:
        repeated = run_kmeans_analysis(
            self.data,
            FEATURES,
            selected_k=None,
            random_state=RANDOM_STATE,
        )
        self.assertTrue(np.array_equal(self.kmeans.labels, repeated.labels))
        self.assertTrue(np.allclose(
            self.kmeans.metrics["silhouette"],
            repeated.metrics["silhouette"],
        ))

    def test_pca(self) -> None:
        self.assertEqual(self.pca.scores.shape, (79, 11))
        self.assertEqual(self.pca.loadings.shape, (10, 11))
        total_variance = float(
            self.pca.explained_variance["varianza_explicada"].sum()
        )
        first_two = float(
            self.pca.explained_variance.loc[1, "varianza_acumulada"]
        )
        components_80 = int(
            (
                self.pca.explained_variance["varianza_acumulada"] < 0.80
            ).sum()
            + 1
        )
        self.assertTrue(math.isclose(total_variance, 1.0, abs_tol=1e-10))
        self.assertTrue(math.isclose(first_two, 0.6072, abs_tol=0.001))
        self.assertEqual(components_80, 4)


class TestPipelineOutputs(unittest.TestCase):
    """Ejecuta el pipeline y comprueba todos los artefactos."""

    @classmethod
    def setUpClass(cls) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "src.main"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        cls.returncode = completed.returncode
        cls.stdout = completed.stdout
        cls.stderr = completed.stderr

    def test_ejecucion_exitosa(self) -> None:
        self.assertEqual(
            self.returncode,
            0,
            msg=f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}",
        )
        self.assertIn("PIPELINE FINALIZADO", self.stdout)

    def test_tablas_generadas(self) -> None:
        expected = {
            "completitud_dataset_completo.csv",
            "dataset_analitico_2022_completo.csv",
            "dataset_analitico_2022_estandarizado.csv",
            "diccionario_10_variables_modelado.csv",
            "resumen_dataset.csv",
            "diccionario_variables_seleccionadas.csv",
            "metricas_kmeans.csv",
            "asignacion_paises_cluster.csv",
            "perfiles_centroides.csv",
            "varianza_explicada_pca.csv",
            "cargas_pca.csv",
            "puntuaciones_pca.csv",
            "comparacion_modelos.csv",
        }
        existing = {path.name for path in TABLE_DIR.glob("*.csv")}
        self.assertTrue(expected.issubset(existing))
        for filename in expected:
            self.assertGreater((TABLE_DIR / filename).stat().st_size, 0)

    def test_matriz_exportada(self) -> None:
        original = pd.read_csv(
            TABLE_DIR / "dataset_analitico_2022_completo.csv"
        )
        scaled = pd.read_csv(
            TABLE_DIR / "dataset_analitico_2022_estandarizado.csv"
        )
        self.assertEqual(original.shape, (79, 13))
        self.assertEqual(scaled.shape, (79, 13))
        self.assertEqual(original["country"].tolist(), scaled["country"].tolist())
        self.assertTrue(np.allclose(
            scaled[FEATURES].mean(axis=0).to_numpy(),
            0.0,
            atol=1e-10,
        ))

    def test_graficas_generadas(self) -> None:
        expected = {
            "exploracion_dataset_completo.png",
            "distribuciones_variables_analiticas.png",
            "correlacion_variables_analiticas.png",
            "seleccion_k_kmeans.png",
            "clusters_kmeans.png",
            "varianza_explicada_pca.png",
            "cargas_pca.png",
            "proyeccion_pca.png",
        }
        existing = {path.name for path in GRAPH_DIR.glob("*.png")}
        self.assertTrue(expected.issubset(existing))
        for filename in expected:
            self.assertGreater((GRAPH_DIR / filename).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
