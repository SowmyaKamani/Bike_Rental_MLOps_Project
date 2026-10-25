from prefect import flow, task
import subprocess
import sys

@task(name="Run Bike Rental MLflow Pipeline")
def run_mlflow_pipeline():
    result = subprocess.run(
        [sys.executable, "src/mlflow_pipeline.py"],
        check=True
    )
    return "MLflow pipeline completed successfully"

@flow(name="Bike Rental Demand Prediction")
def bike_rental_flow():
    run_mlflow_pipeline()

if __name__ == "__main__":
    bike_rental_flow()