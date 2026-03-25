import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from preprocess import load_and_prepare_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fatigue_10k.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_fatigue.pkl"

X, y = load_and_prepare_data(DATA_PATH)

FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)
print("Saved training feature columns at:", FEATURE_COLUMNS_PATH)

X_train, X_test, y_train, y_test = train_test_split (   # train_test split
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 12,
    random_state = 42,
    class_weight = 'balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test) # Evaluate baseline performance

f1 = f1_score(y_test, y_pred, average = 'weighted')
print("Weighted F1 Score:", round(f1, 4))

joblib.dump(model, MODEL_PATH)


