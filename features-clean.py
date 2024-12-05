import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# Import the feature types definition
from featureTypesDef import *

def getPreprocessor(features):
    # Identify the numerical, categorical, and boolean features
    numerical_features = features.select_dtypes(include=['Int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    # Create a preprocessor that will scale numerical, one-hot encode categorical, and pass-through boolean features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)        
        ])
    return preprocessor

# Load the data
print('# Loading data...')
data = pd.read_csv('csvs/train_test_network.csv', 
                   dtype=toniot_dtype_spec,
                   true_values=['T', 't', '1'], false_values=['F', 'f', '0'],
                   na_values=['-'])
print('# Data loaded successfully')

#Value counts for classes
print("# Value counts for classes:")
y = data['label']
counts = y.value_counts()
print(counts.to_string())

# Plot class distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='label', data=data)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                textcoords='offset points')
plt.show()

# Drop columns
columns_to_drop = ['type']
# print("# Dropping columns...")
data_cleaned = data.drop(columns=columns_to_drop, axis=1)
print(f'# Columns dropped: {columns_to_drop}')

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

# Scale numerical and one-hot encode
x = data_cleaned.drop(columns=['label'], axis=1)
preprocessor = getPreprocessor(x)
x_scaled = preprocessor.fit_transform(x)
print("# scaled and one-hot encoded features.")

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