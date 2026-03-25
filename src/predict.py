import joblib
import pandas as pd
from pathlib import Path
from preprocess import prepare_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_fatigue.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
INPUT_CSV = BASE_DIR / "data" / "raw" / "predict_input.csv"

if __name__ == "__main__":
    # Load model and feature columns
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # Load new employee data
    new_data = pd.read_csv(INPUT_CSV)

    # Prepare features
    X = prepare_features(new_data)

    # Align columns with training
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    # Make predictions
    predictions = model.predict(X)

    # Add predictions to dataframe and display
    new_data['predicted_fatigue_level'] = predictions
    print(new_data[['employee_id', 'predicted_fatigue_level']])
