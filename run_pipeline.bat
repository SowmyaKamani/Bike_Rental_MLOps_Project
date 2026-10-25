@echo off
cd /d "%~dp0"
echo Running Bike Rental MLOps pipeline...
python src\mlflow_pipeline.py
if errorlevel 1 (
  echo Pipeline failed. Check the error above.
  pause
  exit /b 1
)
echo Pipeline finished successfully. Starting MLflow UI...
call start_mlflow.bat
