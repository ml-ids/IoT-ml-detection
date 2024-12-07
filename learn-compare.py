from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Tuple
from utils import models, FSELECTION_PATH, MODELS_COMPARISON_PATH

# Constants
DATASET_NAME = "binary_10best_features.csv"
RANDOM_STATE = 42
LR_MAX_ITER = 300
TEST_SIZE = 0.3


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def calculate_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    method: str,
    results: List[Tuple[str, str, float, float, float]],
):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    results.append((model_name, method, accuracy, precision, f1))
    print(
        f"{model_name} (Method {method}): Accuracy = {accuracy:.8f}, Precision = {precision:.8f}, F1-Score = {f1:.8f}"
    )


def train_and_evaluate(
    models: List[Tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method: str,
    results: List[Tuple[str, str, float, float, float]],
):
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        calculate_metrics(y_test, y_pred, name, method, results)


print("# Loading dataset...")
data_cleaned = pd.read_csv(f"{FSELECTION_PATH}{DATASET_NAME}")
X = data_cleaned.drop("label", axis=1)
y = data_cleaned["label"]

print("# Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

results = []

for method in ["None", "SMOTE"]:
    if method == "SMOTE":
        print("# Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)
    print(f"# Training and evaluating models with method: {method}...")
    train_and_evaluate(models, X_train, y_train, X_test, y_test, method, results)

print("# Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_scaling_needed = [
    ("Logistic Regression", LogisticRegression(max_iter=LR_MAX_ITER))
]

for method in ["None", "SMOTE"]:
    if method == "SMOTE":
        print("# Applying SMOTE to scaled features...")
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train, RANDOM_STATE)
    print(f"# Training and evaluating scaled models with method: {method}...")
    train_and_evaluate(
        models_scaling_needed,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        method,
        results,
    )

print("# Saving results to CSV...")
results_df = pd.DataFrame(
    results, columns=["Model", "Method", "Accuracy", "Precision", "F1-Score"]
)
results_df.to_csv(
    f"{MODELS_COMPARISON_PATH}{DATASET_NAME.replace('.csv', '')}_compare.csv",
    index=False,
)
