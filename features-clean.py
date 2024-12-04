import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('train_test_network.csv')
data = pd.DataFrame(data)

columns_to_drop = ['type', 'dst_ip_bytes', 'missed_bytes', 'duration', 'dst_pkts', 
                   'dns_RA', 'dns_AA', 'dns_RD', 'ssl_subject', 'ssl_issuer', 'ssl_version',
                   'ssl_cipher', 'ssl_resumed', 'ssl_established', 'http_trans_depth', 'http_method',
                   'http_version', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
                   'weird_name', 'weird_addl', 'weird_notice']
data_cleaned = data.drop(columns=columns_to_drop, axis=1)
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
print(f"categorical columns: {categorical_columns}")

# Apply LabelEncoder to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col].astype(str))
    label_encoders[col] = le

correlation_matrix = data_cleaned.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature-to-Feature Correlation Heatmap')
plt.show()

correlations = data_cleaned.corr()['label'].sort_values()
plt.figure(figsize=(14, 8))
correlations.plot(kind='bar')
plt.show()

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=data_cleaned)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

data_cleaned.to_csv('features-clean.csv', index=False)