import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
"""
This file creates some statatistics and plots details of the features

"""

parser = argparse.ArgumentParser(description="Cleans the dataset and prints some stats")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-d", "--data-set", type=str, help="Path to the data set.", default="trainsets/Ton_IoT_train_test_network_clean.pkl")
parser.add_argument("-s", "--show", action="store_true", default=False, help="show graphics instead of saving in graphics/ folder")
args = parser.parse_args()

# df = pd.read_csv(args.data_set)
# df_definition = args.data_set.replace('.csv', '_dataset.json')

# if os.path.exists(df_definition):
#     with open(df_definition, 'r') as f:
#         dtypes = pd.read_json(df_definition, typ='series')
#     df_loaded = df.astype(df_definition)
# else:
    # print(f"{df_definition} no definition, load as plain csv.")


df = pd.read_pickle(args.data_set)

columns_to_drop = ['type', 'dst_ip_bytes', 'missed_bytes', 'duration', 'dst_pkts', 
                   'dns_RA', 'dns_AA', 'dns_RD', 'ssl_subject', 'ssl_issuer', 'ssl_version',
                   'ssl_cipher', 'ssl_resumed', 'ssl_established', 'http_trans_depth', 'http_method',
                   'http_version', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
                   'weird_name', 'weird_addl', 'weird_notice']

#data_cleaned = df.drop(columns=columns_to_drop, axis=1)

categorical_columns = df.select_dtypes(include=['object']).columns
print(f"categorical columns: {categorical_columns}")

# Apply LabelEncoder to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    #print(f"Unique values in {col}: {df[col].nunique()}")



# bytes and packets relation
df_subset = df[['src_bytes', 'src_pkts', 'src_ip_bytes', 'dst_bytes', 'dst_pkts', 'dst_ip_bytes']]

# Plot using Pearson method
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Plot using Pearson method
correlation_matrix_pearson = df_subset.corr(method='pearson')
mask = np.triu(np.ones_like(correlation_matrix_pearson, dtype=bool))
sns.heatmap(correlation_matrix_pearson, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0, mask=mask, ax=axes[0])
axes[0].set_title('Pearson')

# Plot using Kendall method
correlation_matrix_kendall = df_subset.corr(method='kendall')
sns.heatmap(correlation_matrix_kendall, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0, mask=mask, ax=axes[1])
axes[1].set_title('Kendall')

# Plot using spearman method
correlation_matrix_kendall = df_subset.corr(method='spearman')
sns.heatmap(correlation_matrix_kendall, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0, mask=mask, ax=axes[2])
axes[2].set_title('Spearman')

plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.5)
plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig("graphics/feature-correlation-bytespackets.png")


# Remove all bool datatypes except 'type' which is boolean
bool_columns = df.select_dtypes(include=['bool']).columns
bool_columns = bool_columns.drop('type', errors='ignore')
f2l = df.drop(columns=bool_columns)

correlations = f2l.corr()['type'].sort_values()
correlations = correlations.drop('type', errors='ignore')
plt.figure(figsize=(14, 8))
correlations.plot(kind='bar',color="#007698")
plt.title('Feature-to-Label Correlation Sorted (binary)')
#plt.figtext(0.5, 0.01, "Note: This graph has currently a bug with BOOL values", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
# for index, value in enumerate(correlations):
#     plt.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

if args.show:
    plt.show()
else:
    plt.savefig("graphics/feature-sorted.png")


f2f = df.drop(columns=bool_columns)

correlation_matrix = f2f.corr(method='spearman')
plt.figure(figsize=(20, 20))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0, mask=mask)
plt.title('Feature-to-Feature Correlation Heatmap')
#plt.figtext(0.5, 0.01, "Note: This graph has currently a bug with BOOL values", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig("graphics/feature-correlation.png")