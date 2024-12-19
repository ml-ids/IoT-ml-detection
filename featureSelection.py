import time
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from utils import *

name, model = models[0]
timings = []


def scale(x):
    print("# Scaling numerical features...")
    numerical_features = int_features + float_features
    scaler = StandardScaler()
    x[numerical_features] = scaler.fit_transform(x[numerical_features])
    return x


def encode(x):
    print("# Encoding categorical features...")
    categorical_features = string_features + boolean_features
    for col in categorical_features:
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(x[col])
    return x


def fillMissingValues(data):
    for col in int_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    for col in float_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    for col in string_features:
        data[col] = data[col].fillna("")
    for col in boolean_features:
        data[col] = data[col].fillna(False)
    return data


def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return (accuracy, precision, recall, f1)


def getVariances(threshold, data):
    var_threshold = VarianceThreshold(threshold=threshold)
    var_threshold.fit(data)
    variances = var_threshold.variances_
    # Create a DataFrame for variances
    variances_df = pd.DataFrame(variances, index=data.columns, columns=["Variance"])
    # Sort variances in descending order
    variances_df = variances_df.sort_values(by="Variance", ascending=False)
    # Features with variance >= threshold
    features_high_variance = variances_df[variances_df["Variance"] >= threshold]
    # Features with variance < threshold
    features_low_variance = variances_df[variances_df["Variance"] < threshold]
    return variances_df, features_high_variance, features_low_variance


def getKBest(k, data, target):
    # selection
    startTime = time.time()
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    x_kbest = selector.fit_transform(data, target)
    endTime = time.time()
    timings.append((f"KBest{target.name}_{k}", endTime - startTime))

    # Create a DataFrame with the selected features
    kbest_features = data.columns[selector.get_support()]
    x_kbest_df = pd.DataFrame(x_kbest, columns=kbest_features)

    return x_kbest_df


def process_kbest(data, target, k, file_path):
    try:
        x_kbest_df = pd.read_csv(file_path)
        print(f"# {file_path} already exists. Skipping KBest computation for k={k}.")
    except FileNotFoundError:
        print(f"# Applying KBest for {target.name} target with k={k}...")
        x_kbest_df = getKBest(k, data, target)
        x_kbest_df[target.name] = target.values
        x_kbest_df.to_csv(file_path, index=False)
    x_kbest_df = x_kbest_df.drop(columns=[target.name], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_kbest_df, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"# Training...")
    startTime = time.time()
    model.fit(x_train, y_train)
    endTime = time.time()
    timings.append((f"KBestTraining{target.name}_{k}", endTime - startTime))
    print(f"# Predicting...")
    y_pred = model.predict(x_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    kbest_features_names = x_kbest_df.columns.tolist()
    return (k, accuracy, precision, recall, f1, kbest_features_names)


def save_results(results, file_path, columns):
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(file_path, index=False)


def getRFE(k, data, target):
    # selection
    startTime = time.time()
    selector = RFE(estimator=model, n_features_to_select=k)
    x_rfe = selector.fit_transform(data, target)
    endTime = time.time()
    timings.append((f"RFE{target.name}_{k}", endTime - startTime))

    # Create a DataFrame with the selected features
    rfe_features = data.columns[selector.get_support()]
    x_rfe_df = pd.DataFrame(x_rfe, columns=rfe_features)

    return x_rfe_df


def process_rfe(data, target, k, file_path):
    try:
        x_rfe_df = pd.read_csv(file_path)
        print(f"# {file_path} already exists. Skipping RFE computation for k={k}.")
    except FileNotFoundError:
        print(f"# Applying RFE for {target.name} target with k={k}...")
        x_rfe_df = getRFE(k, data, target)
        x_rfe_df[target.name] = target.values
        x_rfe_df.to_csv(file_path, index=False)
    x_rfe_df = x_rfe_df.drop(columns=[target.name], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_rfe_df, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"# Training...")
    startTime = time.time()
    model.fit(x_train, y_train)
    endTime = time.time()
    timings.append((f"RFETraining{target.name}_{k}", endTime - startTime))
    print(f"# Predicting...")
    y_pred = model.predict(x_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    rfe_features_names = x_rfe_df.columns.tolist()
    return (k, accuracy, precision, recall, f1, rfe_features_names)


def apply_boruta(data_noLabel, target):
    x_train, x_test, y_train, y_test = train_test_split(
        data_noLabel, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    startTime = time.time()
    boruta = BorutaPy(
        model,
        n_estimators="auto",
        verbose=2,
        random_state=RANDOM_STATE,
    )
    boruta.fit(x_train.values, y_train.values)
    sel_x_train = boruta.transform(x_train.values)
    sel_x_test = boruta.transform(x_test.values)
    endTime = time.time()
    timings.append((f"Boruta{target.name}", endTime - startTime))
    startTime = time.time()
    model.fit(sel_x_train, y_train)
    endTime = time.time()
    timings.append((f"BorutaTraining{target.name}", endTime - startTime))
    y_pred = model.predict(sel_x_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    selected_features_mask = boruta.support_
    selected_features = x_train.columns[selected_features_mask].tolist()
    return (accuracy, precision, recall, f1, selected_features)


# Load the data
print("# Loading data...")
data = pd.read_csv(
    f"{DATASET_PATH}Ton_IoT_train_test_network.csv",
    dtype=toniot_dtype_spec,
    true_values=["T", "t", "1"],
    false_values=["F", "f", "0"],
    na_values=["-"],
)
print("# Data loaded successfully")
print(f"# Features [{data.columns.size}]: ", sorted(data.columns))

# Drop IP and port columns
data = data.drop(columns=["src_ip", "dst_ip", "src_port", "dst_port"], axis=1)
string_features.remove("src_ip")
string_features.remove("dst_ip")
int_features.remove("src_port")
int_features.remove("dst_port")
print(f"# Dropped IP and port columns")

# Value counts for classes
print("# Value counts for binary classes:")
y_label = data["label"]
counts = y_label.value_counts()
print(counts.to_string())
print("# Value counts for multi-class:")
y_type = data["type"]
counts = y_type.value_counts()
print(counts.to_string())

# Fill missing values
print("# Filling missing values...")
data = fillMissingValues(data)
print("# Filled missing values in the train dataset.")

# Scale and encode features
data_scaled = scale(data)
data_prep = encode(data_scaled)
# Split the data into features and target variables
data_binary = data_prep.drop(columns=["type"], axis=1)
data_multi = data_prep.drop(columns=["label"], axis=1)
data_noLabel = data_prep.drop(columns=["label", "type"], axis=1)
y_binary = data_prep["label"]
y_multi = data_prep["type"]

# Apply VarianceThreshold
print("# Applying VarianceThreshold...")
startTime = time.time()
variances_df, features_high_variance, features_low_variance = getVariances(
    0.3, data_noLabel
)
endTime = time.time()
timings.append(("VarianceThreshold", endTime - startTime))
variances_df.to_csv(f"{FSELECTION_PATH}features_variance.csv")
features_high_variance.to_csv(f"{FSELECTION_PATH}features_high_variance.csv")
features_low_variance.to_csv(f"{FSELECTION_PATH}features_low_variance.csv")

# Apply KBest
results_binary = []
results_multi = []
columns = ["Model", "K", "Accuracy", "Precision", "Recall", "F1-Score", "Features"]
for k in range(1, data_noLabel.shape[1] + 1):
    binary_file_path = f"{FSELECTION_PATH}binary_{k}best_features.csv"
    multi_file_path = f"{FSELECTION_PATH}multi_{k}best_features.csv"
    kbestBinaryResult = process_kbest(data_noLabel, y_binary, k, binary_file_path)
    kbestMultiResult = process_kbest(data_noLabel, y_multi, k, multi_file_path)
    results_binary.append((name,) + kbestBinaryResult)
    results_multi.append((name,) + kbestMultiResult)
save_results(
    results_binary, f"{FSELECTION_PATH}binary_kbest_{name}_results.csv", columns
)
save_results(results_multi, f"{FSELECTION_PATH}multi_kbest_{name}_results.csv", columns)

# Apply RFE
results_binary = []
results_multi = []
columns = ["Model", "K", "Accuracy", "Precision", "Recall", "F1-Score", "Features"]
for k in range(1, data_noLabel.shape[1] + 1):
    binary_file_path = f"{FSELECTION_PATH}binary_{k}rfe_features.csv"
    multi_file_path = f"{FSELECTION_PATH}multi_{k}rfe_features.csv"
    rfeBinaryResult = process_rfe(data_noLabel, y_binary, k, binary_file_path)
    rfeMultiResult = process_rfe(data_noLabel, y_multi, k, multi_file_path)
    results_binary.append((name,) + rfeBinaryResult)
    results_multi.append((name,) + rfeMultiResult)
save_results(results_binary, f"{FSELECTION_PATH}binary_rfe_{name}_results.csv", columns)
save_results(results_multi, f"{FSELECTION_PATH}multi_rfe_{name}_results.csv", columns)

# Apply Boruta
columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Features"]
print(f"# Applying Boruta for label classification...")
accuracy, precision, recall, f1, selected_features = apply_boruta(
    data_noLabel, y_binary
)
save_results(
    [(name, accuracy, precision, recall, f1, selected_features)],
    f"{FSELECTION_PATH}binary_boruta_{name}_results.csv",
    columns,
)
print(f"# Applying Boruta for type classification...")
accuracy, precision, recall, f1, selected_features = apply_boruta(data_noLabel, y_multi)
save_results(
    [(name, accuracy, precision, recall, f1, selected_features)],
    f"{FSELECTION_PATH}multi_boruta_{name}_results.csv",
    columns,
)

# Save timings
timings_df = pd.DataFrame(timings, columns=["Method", "Time"])
timings_df.to_csv(f"{FSELECTION_PATH}timings_{name}.csv", index=False)

print(f"# Results in {FSELECTION_PATH}")

# correlation_matrix = data_cleaned.corr()
# plt.figure(figsize=(20, 20))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Feature-to-Feature Correlation Heatmap')
# plt.show()

# correlations = data_cleaned.corr()['label'].sort_values()
# plt.figure(figsize=(14, 8))
# correlations.plot(kind='bar')
# plt.show()

# # Plot class distribution
# plt.figure(figsize=(10, 6))
# sns.countplot(x='label', data=data_cleaned)
# plt.title('Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

# data_cleaned.to_csv('data/features-clean.csv', index=False)
