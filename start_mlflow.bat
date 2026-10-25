@echo off
cd /d "%~dp0"
echo Starting MLflow UI at http://127.0.0.1:5000
start "MLflow Browser" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"
python -m mlflow ui --backend-store-uri "sqlite:///./mlflow.db" --default-artifact-root "./mlartifacts" --host 127.0.0.1 --port 5000
