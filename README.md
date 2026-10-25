# Bike Rental Demand Prediction — MLOps / MLflow

This project follows the structure of the supplied Restaurant Sales Prediction MLOps report, adapted to the supplied Bike Rental Demand dataset.

## Dataset
- File: `data/raw/Bike_Rental_Demand_Data.csv`
- Records: 4,385
- Original columns: 14
- Target: `RentalCount`
- Train/test split: 80/20, `random_state=42`

## Workflow
Raw data → preprocessing → feature engineering → V1/V2/V3 dataset files → six regression models → MAE/MSE/RMSE/R² → MLflow tracking → model comparison → model registry.

## Models
1. Linear Regression
2. Ridge Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. SVR

## MLflow
The full MLflow workflow is `src/mlflow_pipeline.py`. It creates the experiment `Bike_Rental_Demand_Prediction`, logs parameters/metrics/model artifacts, and registers the best model as `Bike_Rental_Demand_Model`.

### Windows setup
Open Command Prompt in this project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\mlflow_pipeline.py
start_mlflow.bat
```

Then open `http://127.0.0.1:5000` in Chrome. The supplied report uses the same localhost MLflow pattern. The batch file keeps the MLflow server process running in the terminal.

## Important limitation of this execution
The current ChatGPT execution environment does not have MLflow installed and cannot download packages from the internet, so the six-model scikit-learn evaluation was executed here, but the MLflow server/UI and registry cannot honestly be claimed as executed here. The supplied `run_pipeline.bat` and `start_mlflow.bat` are prepared for the user's Windows laptop.
