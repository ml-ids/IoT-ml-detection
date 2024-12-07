from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd


def calculate_metrics(y_test, y_pred, model_name, method, results):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append((model_name, method, accuracy, precision, f1))
    print(f"{model_name} (Method {method}): " f"Accuracy = {accuracy:.8f}, Precision = {precision:.8f}, F1-Score = {f1:.8f}")  # print with 8 digit precision


data_cleaned = pd.read_csv("csvs/features-clean.csv")
data_cleaned = pd.DataFrame(data_cleaned)

RANDOM_STATE = 42
LR_MAX_ITER = 300
MLP_MAX_ITER = 500
PR_MAX_ITER = 300
TEST_SIZE = 0.3

X = data_cleaned.drop("label", axis=1)  # features
y = data_cleaned["label"]  # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)  # split into 0.7 data, 0.3 test data ratio

results = []
models = [  # models to be used
    ("Random Forest", RandomForestClassifier()), #https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    ("Decision Tree", DecisionTreeClassifier()), #https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    ("KNN", KNeighborsClassifier()), #https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=MLP_MAX_ITER)), #https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html
    ("Perceptron", Perceptron(max_iter=PR_MAX_ITER)) #https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Perceptron.html
    ("XGBoost", xgb.XGBClassifier()), # https://xgboost.readthedocs.io/en/latest/python/python_api.html
]

for i in ["None", "SMOTE"]:  # train models without and then with SMOTE applied
    if i == "SMOTE":
        # Documentation: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

        # print("Before SMOTE: ", X_train.shape, y_train.shape)
        # print(y_train.value_counts().reset_index())
        # dfboth=pd.concat([X_train,y_train],axis=1)
        # dfboth.to_csv("before-smoth.csv", index=False)

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # print("After SMOTE: ", X_train.shape, y_train.shape)
        # dfboth=pd.concat([X_train,y_train],axis=1)
        # dfboth.to_csv("after-smoth.csv", index=False)
        # print(y_train.value_counts().reset_index())


    for name, model in models:
        model.fit(X_train, y_train)  # train the model
        y_pred = model.predict(X_test)  # predictions

        calculate_metrics(y_test, y_pred, name, i, results)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models_scaling_needed = [
    ("Logistic Regression", LogisticRegression(max_iter=LR_MAX_ITER)) #https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
    ]  # models that need scaling, StandardScaler in this case


for i in ["None", "SMOTE"]:
    if i == "SMOTE":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_scaled, y_train_scaled = smote.fit_resample(X_train_scaled, y_train)

    for name, model in models_scaling_needed:
        model.fit(X_train_scaled, y_train)  # train the model
        y_pred = model.predict(X_test_scaled)  # predictions

        calculate_metrics(y_test, y_pred, name, i, results)


# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results, columns=["Model", "Method", "Accuracy", "Precision", "F1-Score"])

results_df.to_csv("csvs/learn-compare.csv", index=False)
