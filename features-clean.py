import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
# Import the feature types definition
from featureTypesDef import *

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

def getVariances(threshold, data):
    var_threshold = VarianceThreshold(threshold=threshold)  # Set the threshold to 0.3
    var_threshold.fit(data)
    variances = var_threshold.variances_
    # Create a DataFrame for variances
    variances_df = pd.DataFrame(variances, index=data.columns, columns=["Variance"])
    # Sort variances in descending order
    variances_df = variances_df.sort_values(by="Variance", ascending=False)
    # Print features with variance >= 0.3
    features_high_variance = variances_df[variances_df['Variance'] >= 0.3]
    # Print features with variance < 0.3
    features_low_variance = variances_df[variances_df['Variance'] < 0.3]
    return variances_df, features_high_variance, features_low_variance

# Load the data
print('# Loading data...')
data = pd.read_csv('csvs/train_test_network.csv', 
                   dtype=toniot_dtype_spec,
                   true_values=['T', 't', '1'], false_values=['F', 'f', '0'],
                   na_values=['-'])
print('# Data loaded successfully')

#Value counts for classes
print("# Value counts for binary classes:")
y_label = data['label']
counts = y_label.value_counts()
print(counts.to_string())
print("# Value counts for multi-class:")
y_type = data['type']
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
for col in int_features:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
for col in float_features:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
for col in string_features:
    data[col] = data[col].fillna("")
for col in boolean_features:
    data[col] = data[col].fillna(False)
print("# Filled missing values in the train dataset.")

data_binary = data.drop(columns=['type'], axis=1)
data_multi = data.drop(columns=['label'], axis=1)
data_noLabel = data.drop(columns=['label', 'type'], axis=1)

# Scale numerical features
data_prep = scaleEncode(data)

# Apply VarianceThreshold
print("# Applying VarianceThreshold...")
variances_df, features_high_variance, features_low_variance = getVariances(0.3, data_prep)
variances_df.to_csv('csvs/variances.csv')
features_high_variance.to_csv('csvs/features_high_variance.csv')
features_low_variance.to_csv('csvs/features_low_variance.csv')

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

# data_cleaned.to_csv('csvs/features-clean.csv', index=False)