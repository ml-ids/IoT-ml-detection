import os
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


## cannot remove scale and encode because time constraints x)
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
    ##filename = f"{MODELS_COMPARISON_PATH}{featurefilename}_{model_name}_results.csv"
    ##dfresult = pd.DataFrame([output])
    ##dfresult.to_csv(filename, index=False)
    

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
    #columns = df.columns.tolist()
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
    ###data_binary = data_prep.drop(columns=["type"], axis=1)
    ###data_multi = data_prep.drop(columns=["label"], axis=1)
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


## useless junk which works but w/e
# def apply_randomforest(x_train, x_test, y_train, y_test):
#     rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    

#     rf.fit(x_train, y_train)
#     y_pred = rf.predict(x_test)
    
#     acc = accuracy_score(y_test, y_pred)  # remember to fix later
#     print(f"Random Forest Accuracy: {acc:.4f}")
    
#     # Save the model
#     save_model(rf, "random_forest", FSELECTION_PATH)
#     return y_pred


## replaced version of apply_models
# def OLD_apply_models(x_binary_train, x_multi_train, x_binary_test, x_multi_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test ):

#     for model_name, model in models:

#         #binary
#         print("\n\n")
#         print(f"training {model_name}.. binary")
#         mode=("binary")
#         model.fit(x_binary_train, y_binary_train)
#         y_pred = model.predict(x_binary_test)
#         metrics = calculate_metrics(mode, y_binary_test, y_pred, model_name)
#         print(f"rq{model_name}, {metrics}") ## sanity check
#         save_model(mode, model, model_name, MODELS_COMPARISON_PATH)
#         print("\n\n")
#         #multi
#         print(f"training {model_name}.. multi")
#         mode=("multi")
#         model.fit(x_multi_train, y_multi_train)
#         y_pred = model.predict(x_multi_test)
#         metrics = calculate_metrics(mode, y_multi_test, y_pred, model_name)
#         print(f"rq{model_name}, {metrics}") ## sanity check
#         save_model(mode, model, model_name, MODELS_COMPARISON_PATH)
#         print("\n\n")

#     print("cooked")



def apply_models(mode, x_train, x_test, y_train, y_test):
    
    for model_name, model in models:
        print("\n\n")
        print(f"training {model_name}.. {mode}")

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = calculate_metrics(mode, y_test, y_pred, model_name)
        print(f"rq{model_name}, {metrics}") ## sanity check
        save_model(mode, model, model_name, MODELS_COMPARISON_PATH)
    print("FIN\n\n")


# oh god...
def apply_selectedfeatures():
    all_feature_files = os.listdir(FSELECTION_PATH)
    x_train, x_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test = apply_preprocessing()

    ###x_train, x_test, y_train, y_test = apply_preprocessing1()
    ## rip pc...
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

            #apply_models2(filename, x_train, x_test, y_train, y_test)

            #this can simplifies but im too tired
            if file.startswith('binary_'):
                apply_models(filename, x_binary_train, x_binary_test, y_binary_train, y_binary_test)
            elif file.startswith('multi_'):
                apply_models(filename, x_multi_train, x_multi_test, y_multi_train, y_multi_test)
            else:
                print("error with file format.")  #x)

            print("next file..")

    print("all flies completed")







# binary_features = load_features(fselection_binary)
# multi_features = load_features(fselection_multi)

# x_train, x_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test = apply_preprocessing()


# print(x_train.head())

# print(binary_features)
# print("\n\n")
# print(multi_features)

# x_binary_train = x_train[binary_features]
# x_binary_test = x_test[binary_features]

# x_multi_train = x_train[multi_features]
# x_multi_test = x_test[multi_features]


# ## our chad features
# OLD_apply_models(x_binary_train, x_multi_train, x_binary_test, x_multi_test, y_binary_train, y_multi_train, y_binary_test, y_multi_test)


def main():
    print("#1 applying selected features")
    apply_selectedfeatures()
    print("done saved...")
    export_results()
    print("# finalfinal")

if __name__ == "__main__":
    main()