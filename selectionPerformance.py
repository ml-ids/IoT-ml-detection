import os
import time
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder


from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import *


#const
dataset = f"{DATASET_PATH}Ton_IoT_train_test_network.csv"
#fselection_binary = f"{FSELECTION_PATH}binary_5best_features.csv"
#fselection_multi = f"{FSELECTION_PATH}multi_5best_features.csv"
all_results= []
train_timings=[]
test_timings=[]

## utils.py should be changed to this
models = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ("KNN", KNeighborsClassifier(n_neighbors=KNN_K)),  # set to 5 for now
    ("NaiveBayes_GaussianNB", GaussianNB()),
    ("LogisticRegression", LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)),
    ("XGBoost", XGBClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    #("SVM", SVC(kernel="linear", random_state=RANDOM_STATE, max_iter=SVC_MAX_ITER)),    if somene love this algo, please run it, it will take all day.... trust me
]


def save_model(filename, model, name, file_path):
    model_filename = f"{file_path}{filename}_{name}_model.pkl"  ## creates empty file
    joblib.dump(model, model_filename)
    print(f"Model saved at {model_filename} at {file_path}")

def scale(x):
    print("# Scaling numerical features...")
    numerical_features = int_features + float_features
    scaler = StandardScaler()
    x_numerical_scaled = scaler.fit_transform(x[numerical_features])
    numerical_df = pd.DataFrame(x_numerical_scaled, columns=numerical_features)
    return pd.concat([numerical_df, x.drop(columns=numerical_features)], axis=1)


def encode(x):
    print("# Encoding categorical features...")
    categorical_features = string_features + boolean_features
    label_encoded_dfs = []
    for col in categorical_features:
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(x[col])
        label_encoded_dfs.append(x[col])
    label_encoded_df = pd.concat(label_encoded_dfs, axis=1)
    return pd.concat(
        [label_encoded_df, x.drop(columns=categorical_features)], axis=1
    )


def scaleEncode(x):
     scaled_df = scale(x)
     scaledEncoded_df = encode(scaled_df)
     return scaledEncoded_df

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


def calculate_metrics(featurefilename, y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    output = {
        'file': featurefilename,
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    all_results.append(output)
    

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    return (accuracy, precision, recall, f1)

def export_each_file(featurefilename,accuracy, precision, recall, f1, model_name):

    output = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    filename = f"{MODELS_COMPARISON_PATH}{featurefilename}_{model_name}_results.csv"
    dfresult = pd.DataFrame([output])
    dfresult.to_csv(filename, index=False)


def export_results():
    df = pd.DataFrame(all_results)
    filename = f"{MODELS_COMPARISON_PATH}all_metrics_combined.csv"
    df.to_csv(filename, index=False)
    print(f"#Saved to {filename}")

def load_features(feature_list):
    # the first row of the csv
    df = pd.read_csv(feature_list, nrows=1)
    columns = df.columns.drop(['label', 'type'], errors='ignore').tolist()
 
    print("features:", columns)

    return columns


def apply_preprocessing():

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
    data_prep = scaleEncode(data)
    # Split the data into features and target variables
    x_data_noLabel = data_prep.drop(columns=["label", "type"], axis=1)
    y_binary = data_prep["label"]
    y_multi = data_prep["type"]

    

    print(f"\n# Checking shape of data before split: {x_data_noLabel.shape}")


    print("# Splitting dataset into training and testing sets... binary")
    x_train, x_test, y_binary_train, y_binary_test = train_test_split(
        x_data_noLabel, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n# Checking shape of x_train: {x_train.shape}, y_train: {y_binary_train.shape}")
    print(f"\n# Checking shape of x_test: {x_test.shape}, y_test: {y_binary_test.shape}")

    print("#\n# Splitting dataset into training and testing sets... multi")
    x_train, x_test, y_multi_train, y_multi_test = train_test_split(
        x_data_noLabel, y_multi, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


    print(f"\n# Checking shape of x_train: {x_train.shape}, y_train: {y_multi_train.shape}")
    print(f"\n# Checking shape of of x_test: {x_test.shape}, y_test: {y_multi_test.shape}")

    return x_train, x_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test


def apply_models(mode, x_train, x_test, y_train, y_test):
    
    for model_name, model in models:
        print("\n\n")
        print(f"training {model_name}.. {mode}")

        train_startTime = time.time()
        model.fit(x_train, y_train)
        train_endTime = time.time()
        train_timings.append((mode, model_name, train_endTime - train_startTime))




        test_startTime = time.time()
        y_pred = model.predict(x_test)
        test_endTime = time.time()
        test_timings.append((mode, model_name, test_endTime - test_startTime))


        metrics = calculate_metrics(mode, y_test, y_pred, model_name)
        print(f"rq{model_name}, {metrics}") ## sanity check
        save_model(mode, model, model_name, MODELS_COMPARISON_PATH)
    print("FIN\n\n")


def apply_selectedfeatures():
    all_feature_files = os.listdir(FSELECTION_PATH)
    x_train, x_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test = apply_preprocessing()

    for file in all_feature_files:

        filename = os.path.splitext(file)[0]
        if file.endswith('.csv'):
            file_path = os.path.join(FSELECTION_PATH, file)
            print(f"Processing file: {file_path}")
            features = load_features(file_path)
            x_binary_train = x_train[features]
            x_binary_test = x_test[features]
            x_multi_train = x_train[features]
            x_multi_test = x_test[features]

            if file.startswith('binary_'):
                apply_models(filename, x_binary_train, x_binary_test, y_binary_train, y_binary_test)
            elif file.startswith('multi_'):
                apply_models(filename, x_multi_train, x_multi_test, y_multi_train, y_multi_test)
            else:
                print("error with file format.")  #x)

            print("next file..")

    print("all flies completed")


def saveTime(name):
    timings_df = pd.DataFrame(train_timings, columns=["File","Method", "Time"])
    timings_df.to_csv(f"{MODELS_COMPARISON_PATH}traintimings_{name}.csv", index=False)

    timings_df = pd.DataFrame(test_timings, columns=["File","Method", "Time"])
    timings_df.to_csv(f"{MODELS_COMPARISON_PATH}testtimings_{name}.csv", index=False)


def main():
    print("#1 applying selected features")
    apply_selectedfeatures()
    print("done saved...")
    export_results()
    saveTime("timecollection")
    print("# finalfinal")

if __name__ == "__main__":
    main()