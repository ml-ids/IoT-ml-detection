import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from utils import *


def scaleEncode(x):
    print("# Scaling numerical features...")
    numerical_features = int_features + float_features
    scaler = StandardScaler()
    x_numerical_scaled = scaler.fit_transform(x[numerical_features])
    numerical_df = pd.DataFrame(x_numerical_scaled, columns=numerical_features)
    # Encode categorical features using LabelEncoder
    print("# Encoding categorical features...")
    categorical_features = string_features + boolean_features
    label_encoded_dfs = []
    for col in categorical_features:
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(x[col])
        label_encoded_dfs.append(x[col])
    # Combine scaled numerical and label-encoded categorical features
    print("# Combining scaled and encoded features...")
    label_encoded_df = pd.concat(label_encoded_dfs, axis=1)
    x_prep = pd.concat([numerical_df, label_encoded_df], axis=1)
    return x_prep


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
    f1 = f1_score(y_test, y_pred, average="weighted")
    return (accuracy, precision, f1)


def getVariances(threshold, data):
    var_threshold = VarianceThreshold(threshold=threshold)  # Set the threshold to 0.3
    var_threshold.fit(data)
    variances = var_threshold.variances_
    # Create a DataFrame for variances
    variances_df = pd.DataFrame(variances, index=data.columns, columns=["Variance"])
    # Sort variances in descending order
    variances_df = variances_df.sort_values(by="Variance", ascending=False)
    # Print features with variance >= 0.3
    features_high_variance = variances_df[variances_df["Variance"] >= 0.3]
    # Print features with variance < 0.3
    features_low_variance = variances_df[variances_df["Variance"] < 0.3]
    return variances_df, features_high_variance, features_low_variance


def getKBest(k, data, target, targetName):
    # selection
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    x_kbest = selector.fit_transform(data.drop(columns=targetName), target)

    # Create a DataFrame with the selected features
    kbest_features = data.drop(columns=targetName).columns[selector.get_support()]
    x_kbest_df = pd.DataFrame(x_kbest, columns=kbest_features)

    return x_kbest_df


def process_kbest(data, target, target_name, k, file_path, model):
    try:
        x_kbest_df = pd.read_csv(file_path)
        print(f"# {file_path} already exists. Skipping KBest computation for k={k}.")
    except FileNotFoundError:
        print(f"# Applying KBest for {target_name} target with k={k}...")
        x_kbest_df = getKBest(k, data, target, target_name)
        x_kbest_df[target_name] = target.values
        x_kbest_df.to_csv(file_path, index=False)
    x_kbest_df = x_kbest_df.drop(columns=[target_name], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_kbest_df, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"# Training...")
    model.fit(x_train, y_train)
    print(f"# Predicting...")
    y_pred = model.predict(x_test)
    accuracy, precision, f1 = calculate_metrics(y_test, y_pred)
    kbest_features_names = x_kbest_df.columns.tolist()
    return (k, accuracy, precision, f1, kbest_features_names)


def save_results(results, file_path, columns):
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(file_path, index=False)


def getRFE(k, data, target, targetName):
    # selection
    selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=k)
    x_rfe = selector.fit_transform(data.drop(columns=targetName), target)

    # Create a DataFrame with the selected features
    rfe_features = data.drop(columns=targetName).columns[selector.get_support()]
    x_rfe_df = pd.DataFrame(x_rfe, columns=rfe_features)

    return x_rfe_df


def process_rfe(data, target, target_name, k, file_path, model):
    try:
        x_rfe_df = pd.read_csv(file_path)
        print(f"# {file_path} already exists. Skipping RFE computation for k={k}.")
    except FileNotFoundError:
        print(f"# Applying RFE for {target_name} target with k={k}...")
        x_rfe_df = getRFE(k, data, target, target_name)
        x_rfe_df[target_name] = target.values
        x_rfe_df.to_csv(file_path, index=False)
    x_rfe_df = x_rfe_df.drop(columns=[target_name], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_rfe_df, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"# Training...")
    model.fit(x_train, y_train)
    print(f"# Predicting...")
    y_pred = model.predict(x_test)
    accuracy, precision, f1 = calculate_metrics(y_test, y_pred)
    rfe_features_names = x_rfe_df.columns.tolist()
    return (k, accuracy, precision, f1, rfe_features_names)


def apply_boruta(data_noLabel, target, target_name, model):
    x_train, x_test, y_train, y_test = train_test_split(
        data_noLabel, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"# Applying Boruta for {target_name} classification...")
    boruta = BorutaPy(
        model,
        n_estimators="auto",
        verbose=2,
        random_state=RANDOM_STATE,
    )
    boruta.fit(x_train.values, y_train.values)
    sel_x_train = boruta.transform(x_train.values)
    sel_x_test = boruta.transform(x_test.values)
    model.fit(sel_x_train, y_train)
    y_pred = model.predict(sel_x_test)
    accuracy, precision, f1 = calculate_metrics(y_test, y_pred)
    selected_features_mask = boruta.support_
    selected_features = x_train.columns[selected_features_mask].tolist()
    return (accuracy, precision, f1, selected_features)


name, model = models[0]
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

# Plot class distribution
# plt.figure(figsize=(10, 6))
# ax = sns.countplot(x='label', data=data)
# plt.title('Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Count')
# for p in ax.patches:
#     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
#                 textcoords='offset points')
# plt.show()

# Fill missing values
# print("# Filling missing values...")
data = fillMissingValues(data)
print("# Filled missing values in the train dataset.")

# Scale numerical features
data_prep = scaleEncode(data)
# Split the data into features and target variables
data_binary = data_prep.drop(columns=["type"], axis=1)
data_multi = data_prep.drop(columns=["label"], axis=1)
data_noLabel = data_prep.drop(columns=["label", "type"], axis=1)
y_binary = data["label"]
y_multi = data["type"]

# Apply VarianceThreshold
print("# Applying VarianceThreshold...")
variances_df, features_high_variance, features_low_variance = getVariances(
    0.3, data_noLabel
)
variances_df.to_csv(f"{FSELECTION_PATH}features_variance.csv")
features_high_variance.to_csv(f"{FSELECTION_PATH}features_high_variance.csv")
features_low_variance.to_csv(f"{FSELECTION_PATH}features_low_variance.csv")

# Apply KBest
results_binary = []
results_multi = []
columns = ["Model", "K", "Accuracy", "Precision", "F1-Score", "Features"]
for k in range(1, data_noLabel.shape[1] + 1):
    binary_file_path = f"{FSELECTION_PATH}binary_{k}best_features.csv"
    multi_file_path = f"{FSELECTION_PATH}multi_{k}best_features.csv"
    results_binary.append(
        (name,)
        + process_kbest(data_binary, y_binary, "label", k, binary_file_path, model)
    )
    results_multi.append(
        (name,) + process_kbest(data_multi, y_multi, "type", k, multi_file_path, model)
    )

save_results(
    results_binary, f"{FSELECTION_PATH}binary_kbest_{name}_results.csv", columns
)
save_results(results_multi, f"{FSELECTION_PATH}multi_kbest_{name}_results.csv", columns)

# Apply RFE
results_binary = []
results_multi = []
columns = ["Model", "K", "Accuracy", "Precision", "F1-Score", "Features"]
for k in range(1, data_noLabel.shape[1] + 1):
    binary_file_path = f"{FSELECTION_PATH}binary_{k}rfe_features.csv"
    multi_file_path = f"{FSELECTION_PATH}multi_{k}rfe_features.csv"
    results_binary.append(
        (name,)
        + process_rfe(data_binary, y_binary, "label", k, binary_file_path, model)
    )
    results_multi.append(
        (name,) + process_rfe(data_multi, y_multi, "type", k, multi_file_path, model)
    )

save_results(results_binary, f"{FSELECTION_PATH}binary_rfe_{name}_results.csv", columns)
save_results(results_multi, f"{FSELECTION_PATH}multi_rfe_{name}_results.csv", columns)

# Apply Boruta
columns = ["Model", "Accuracy", "Precision", "F1-Score", "Features"]
accuracy, precision, f1, selected_features = apply_boruta(
    data_noLabel, y_binary, "label", model
)
save_results(
    [(name, accuracy, precision, f1, selected_features)],
    f"{FSELECTION_PATH}binary_boruta_{name}_results.csv",
    columns,
)
accuracy, precision, f1, selected_features = apply_boruta(
    data_noLabel, y_multi, "type", model
)
save_results(
    [(name, accuracy, precision, f1, selected_features)],
    f"{FSELECTION_PATH}multi_boruta_{name}_results.csv",
    columns,
)

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
