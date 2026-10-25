import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW = os.path.join(
    ROOT,
    'data',
    'raw',
    'Bike_Rental_Demand_Data.csv'
)

PROCESSED = os.path.join(
    ROOT,
    'data',
    'processed',
    'bike_rental_v2_processed.csv'
)

FEATURES = os.path.join(
    ROOT,
    'data',
    'features',
    'bike_rental_v3_features.csv'
)

RESULTS = os.path.join(
    ROOT,
    'results.csv'
)

MLRUNS = os.path.join(
    ROOT,
    'mlruns'
)

DB = os.path.join(
    ROOT,
    'mlflow.db'
)


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    os.path.dirname(PROCESSED),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(FEATURES),
    exist_ok=True
)

os.makedirs(
    os.path.join(ROOT, 'models'),
    exist_ok=True
)


# ============================================================
# 3. LOAD RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("BIKE RENTAL DEMAND - MLOPS PIPELINE")
print("=" * 70)

print("\n[1/8] Loading raw dataset...")

if not os.path.exists(RAW):
    raise FileNotFoundError(
        f"\nDataset not found:\n{RAW}\n"
        "\nPlease make sure Bike_Rental_Demand_Data.csv "
        "is inside data\\raw."
    )

raw = pd.read_csv(RAW)

print("Dataset loaded successfully.")
print(f"Rows    : {raw.shape[0]}")
print(f"Columns : {raw.shape[1]}")

print("\nColumns:")
print(list(raw.columns))


# ============================================================
# 4. SAVE V1 RAW DATASET
# ============================================================

print("\n[2/8] Saving V1 raw dataset...")

RAW_VERSION = os.path.join(
    ROOT,
    'data',
    'raw',
    'bike_rental_v1_raw.csv'
)

raw.to_csv(
    RAW_VERSION,
    index=False
)

print(f"V1 saved: {RAW_VERSION}")


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

print("\n[3/8] Preprocessing dataset...")

p = raw.drop_duplicates().copy()

print(
    f"Duplicate rows removed: "
    f"{len(raw) - len(p)}"
)


# Convert Date
if 'Date' in p.columns:
    p['Date'] = pd.to_datetime(
        p['Date'],
        dayfirst=True,
        errors='coerce'
    )


# Convert Time
if 'Time' in p.columns:
    p['Time'] = pd.to_datetime(
        p['Time'],
        format='%H:%M',
        errors='coerce'
    )


# Handle numerical missing values
num_cols = p.select_dtypes(
    include=np.number
).columns

for c in num_cols:
    p[c] = p[c].fillna(
        p[c].median()
    )


# Handle categorical missing values
cat_cols = p.select_dtypes(
    include='object'
).columns

for c in cat_cols:
    p[c] = p[c].fillna(
        'Unknown'
    )


# Save V2
p.to_csv(
    PROCESSED,
    index=False
)

print(f"V2 processed dataset saved: {PROCESSED}")


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\n[4/8] Performing feature engineering...")

f = p.copy()


# Date features
f['Year'] = f['Date'].dt.year

f['Month'] = f['Date'].dt.month

f['Day'] = f['Date'].dt.day

f['DayOfWeek_Num'] = (
    f['Date'].dt.dayofweek
)


# Time feature
f['Hour'] = (
    f['Time']
    .dt.hour
    .fillna(0)
    .astype(int)
)


# Weekend feature
f['Is_Weekend'] = (
    f['DayOfWeek_Num'] >= 5
).astype(int)


# Cyclical month features
f['Month_Sin'] = np.sin(
    2 * np.pi * f['Month'] / 12
)

f['Month_Cos'] = np.cos(
    2 * np.pi * f['Month'] / 12
)


# Cyclical hour features
f['Hour_Sin'] = np.sin(
    2 * np.pi * f['Hour'] / 24
)

f['Hour_Cos'] = np.cos(
    2 * np.pi * f['Hour'] / 24
)


# Save V3
f.to_csv(
    FEATURES,
    index=False
)

print(f"V3 feature dataset saved: {FEATURES}")


# ============================================================
# 7. SELECT FEATURES AND TARGET
# ============================================================

print("\n[5/8] Preparing training data...")

TARGET = 'RentalCount'

FEATURE_LIST = [
    'DayOfWeek',
    'Season',
    'Weather',
    'Temperature',
    'Humidity',
    'WindSpeed',
    'RentalDuration',
    'UserType',
    'PaymentMethod',
    'Cancelled',
    'Year',
    'Month',
    'Day',
    'DayOfWeek_Num',
    'Hour',
    'Is_Weekend',
    'Month_Sin',
    'Month_Cos',
    'Hour_Sin',
    'Hour_Cos'
]


# Verify required columns
missing_features = [
    c for c in FEATURE_LIST
    if c not in f.columns
]

if missing_features:
    raise ValueError(
        "\nThe following required feature columns "
        "are missing from the dataset:\n"
        + "\n".join(missing_features)
    )


if TARGET not in f.columns:
    raise ValueError(
        f"\nTarget column '{TARGET}' "
        "was not found in the dataset."
    )


X = f[FEATURE_LIST].copy()

y = f[TARGET].copy()


print(f"Target variable : {TARGET}")
print(f"Number features : {len(FEATURE_LIST)}")
print(f"Rows            : {len(X)}")


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training rows   : {len(X_train)}")
print(f"Testing rows    : {len(X_test)}")


# ============================================================
# 9. PREPROCESSING PIPELINE
# ============================================================

cat = X.select_dtypes(
    include='object'
).columns.tolist()

num = [
    c for c in FEATURE_LIST
    if c not in cat
]


pre = ColumnTransformer(
    transformers=[
        (
            'num',
            SimpleImputer(
                strategy='median'
            ),
            num
        ),

        (
            'cat',
            Pipeline(
                steps=[
                    (
                        'imp',
                        SimpleImputer(
                            strategy='most_frequent'
                        )
                    ),

                    (
                        'onehot',
                        OneHotEncoder(
                            handle_unknown='ignore'
                        )
                    )
                ]
            ),
            cat
        )
    ]
)


# ============================================================
# 10. MACHINE LEARNING MODELS
# ============================================================

models = {

    'Linear Regression':
        LinearRegression(),

    'Ridge Regression':
        Ridge(
            alpha=1.0
        ),

    'Decision Tree':
        DecisionTreeRegressor(
            random_state=42
        ),

    'Random Forest':
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    'Gradient Boosting':
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ),

    'SVR':
        SVR()
}


# ============================================================
# 11. MLflow CONFIGURATION
# ============================================================

print("\n[6/8] Configuring MLflow...")

tracking_uri = (
    'sqlite:///'
    + DB.replace('\\', '/')
)

mlflow.set_tracking_uri(
    tracking_uri
)

mlflow.set_experiment(
    'Bike_Rental_Demand_Prediction'
)

print(
    f"Tracking URI: {tracking_uri}"
)

print(
    "Experiment: Bike_Rental_Demand_Prediction"
)


# ============================================================
# 12. TRAIN AND LOG SIX MODELS
# ============================================================

print("\n[7/8] Training six regression models...")
print("=" * 70)


rows = []


for name, model in models.items():

    print(
        f"\nTraining: {name}"
    )

    # Build pipeline
    steps = [
        (
            'preprocessor',
            pre
        )
    ]


    # SVR requires scaling
    if name == 'SVR':

        steps.append(
            (
                'scaler',
                StandardScaler(
                    with_mean=False
                )
            )
        )


    steps.append(
        (
            'model',
            model
        )
    )


    pipe = Pipeline(
        steps
    )


    # MLflow run
    with mlflow.start_run(
        run_name=name
    ) as run:

        # Train
        pipe.fit(
            X_train,
            y_train
        )


        # Predict
        pred = pipe.predict(
            X_test
        )


        # Metrics
        mae = mean_absolute_error(
            y_test,
            pred
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = float(
            np.sqrt(mse)
        )

        r2 = r2_score(
            y_test,
            pred
        )


        # Log parameters
        mlflow.log_param(
            'Model',
            name
        )

        mlflow.log_param(
            'test_size',
            0.20
        )

        mlflow.log_param(
            'random_state',
            42
        )

        mlflow.log_param(
            'feature_count',
            len(FEATURE_LIST)
        )

        mlflow.log_param(
            'dataset_version',
            'V3.0'
        )


        # Log metrics
        mlflow.log_metric(
            'MAE',
            float(mae)
        )

        mlflow.log_metric(
            'MSE',
            float(mse)
        )

        mlflow.log_metric(
            'RMSE',
            rmse
        )

        mlflow.log_metric(
            'R2',
            float(r2)
        )


        # ====================================================
        # IMPORTANT MLflow FIX
        #
        # MLflow 3.16 + skops can reject numpy.dtype
        # as an untrusted type.
        #
        # cloudpickle avoids this specific serialization
        # problem.
        # ====================================================

        signature = infer_signature(
            X_train,
            pipe.predict(
                X_train.iloc[:5]
            )
        )


        mlflow.sklearn.log_model(
            pipe,
            'model',
            signature=signature,
            serialization_format='cloudpickle'
        )


        # Store results
        rows.append(
            {
                'Model': name,
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': rmse,
                'R2': float(r2),
                'Run_ID': run.info.run_id
            }
        )


        print(
            f"Completed: {name}"
        )

        print(
            f"MAE  : {mae:.6f}"
        )

        print(
            f"MSE  : {mse:.6f}"
        )

        print(
            f"RMSE : {rmse:.6f}"
        )

        print(
            f"R2   : {r2:.6f}"
        )

        print(
            f"Run ID: {run.info.run_id}"
        )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)

results = results.sort_values(
    'R2',
    ascending=False
).reset_index(
    drop=True
)


results.to_csv(
    RESULTS,
    index=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best = results.iloc[0]

best_model_name = best[
    'Model'
]

best_run_id = best[
    'Run_ID'
]

best_r2 = best[
    'R2'
]

best_rmse = best[
    'RMSE'
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best Model : {best_model_name}"
)

print(
    f"R2         : {best_r2:.6f}"
)

print(
    f"RMSE       : {best_rmse:.6f}"
)

print(
    f"Run ID     : {best_run_id}"
)


# ============================================================
# 15. REGISTER BEST MODEL
# ============================================================

print("\nRegistering best model...")

model_uri = (
    f"runs:/{best_run_id}/model"
)

try:

    mv = mlflow.register_model(
        model_uri=model_uri,
        name='Bike_Rental_Demand_Model'
    )

    print(
        "Registered best model successfully."
    )

    print(
        "Model name: "
        "Bike_Rental_Demand_Model"
    )

    print(
        f"Version: {mv.version}"
    )

except Exception as exc:

    print(
        "Model registration warning:"
    )

    print(exc)


# ============================================================
# 16. CREATE ADDITIONAL MODEL VERSIONS
# ============================================================

print("\nCreating additional Gradient Boosting versions...")


for n_estimators in [
    150,
    200
]:

    print(
        f"\nTraining Gradient Boosting "
        f"with n_estimators={n_estimators}"
    )


    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        random_state=42
    )


    pipe = Pipeline(
        [
            (
                'preprocessor',
                pre
            ),

            (
                'model',
                model
            )
        ]
    )


    with mlflow.start_run(
        run_name=(
            f'Gradient Boosting v'
            f'{n_estimators}'
        )
    ) as run:

        pipe.fit(
            X_train,
            y_train
        )


        pred = pipe.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            pred
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = float(
            np.sqrt(mse)
        )

        r2 = r2_score(
            y_test,
            pred
        )


        # Log parameters
        mlflow.log_param(
            'Model',
            'Gradient Boosting'
        )

        mlflow.log_param(
            'n_estimators',
            n_estimators
        )

        mlflow.log_param(
            'random_state',
            42
        )


        # Log metrics
        mlflow.log_metric(
            'MAE',
            float(mae)
        )

        mlflow.log_metric(
            'MSE',
            float(mse)
        )

        mlflow.log_metric(
            'RMSE',
            rmse
        )

        mlflow.log_metric(
            'R2',
            float(r2)
        )


        # ====================================================
        # IMPORTANT:
        # Use cloudpickle here as well.
        # ====================================================

        mlflow.sklearn.log_model(
            pipe,
            'model',
            serialization_format='cloudpickle'
        )


        print(
            f"Gradient Boosting "
            f"{n_estimators} completed."
        )

        print(
            f"R2: {r2:.6f}"
        )


        # Register additional version
        try:

            registered = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                'Bike_Rental_Demand_Model'
            )

            print(
                "Registered model version: "
                f"{registered.version}"
            )

        except Exception as exc:

            print(
                "Additional registration warning:"
            )

            print(exc)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    "\nFinal model comparison:"
)

print(
    results.to_string(
        index=False
    )
)

print(
    f"\nBest model: "
    f"{best_model_name}"
    f" | R2={best_r2:.6f}"
)

print(
    f"Best model RMSE: "
    f"{best_rmse:.6f}"
)

print(
    "\nMLflow experiment:"
)

print(
    "Bike_Rental_Demand_Prediction"
)

print(
    "\nRegistered model:"
)

print(
    "Bike_Rental_Demand_Model"
)

print(
    "\nResults saved to:"
)

print(
    RESULTS
)

print(
    "\nYou can now start the MLflow UI."
)

print("=" * 70)