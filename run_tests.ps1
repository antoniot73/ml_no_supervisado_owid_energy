$ErrorActionPreference = "Stop"

Write-Host "1/4 Verificando estructura..."
if (-not (Test-Path ".\src\__init__.py")) {
    throw "No se encontró src\__init__.py"
}
if (-not (Test-Path ".\tests\test_pipeline.py")) {
    throw "No se encontró tests\test_pipeline.py"
}

Write-Host "2/4 Compilando..."
python -m compileall src tests

Write-Host "3/4 Ejecutando pruebas..."
python -m unittest discover -s tests -p "test_*.py" -v

Write-Host "4/4 Ejecutando pipeline..."
python -m src.main

Write-Host ""
Write-Host "VALIDACIÓN COMPLETA: OK"
