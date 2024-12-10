import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    boolean_features,
    float_features,
    int_features,
    string_features,
    DATASET_PATH,
)

"""
This reads the raw CSV data, applies the correct data types, print some stats and saves the cleaned data to a new CSV file plus data type defintions

"""

parser = argparse.ArgumentParser(description="Cleans the dataset and prints some stats")
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase output verbosity. Use multiple times for more verbosity.",
)
parser.add_argument(
    "-d",
    "--data-set",
    type=str,
    help="Path to the data set CSV file.",
    default=f"{DATASET_PATH}Ton_IoT_train_test_network.csv",
)
parser.add_argument(
    "-s",
    "--show",
    action="store_true",
    default=False,
    help="show graphics instead of saving in graphics/ folder",
)
args = parser.parse_args()


df = pd.read_csv(args.data_set)
df = pd.DataFrame(df)

## Apply datatypes and set default valuse if empty
for col in int_features:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
for col in float_features:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
for col in string_features:
    df[col] = df[col].fillna("").astype(str)
for col in boolean_features:
    df[col] = df[col].astype(bool)


# Plot the data types of each feature/column
plt.figure(figsize=(12, 4))
df.dtypes.value_counts().plot(kind="bar", color="#007698")
plt.title("Data Types of Features")
plt.xlabel("Data Type")
plt.ylabel("Number of Features/Columns")
for i, count in enumerate(df.dtypes.value_counts()):
    plt.text(i, count, str(count), ha="center", va="bottom")
plt.figtext(0, 0, f"{args.data_set}", fontsize=10)
plt.subplots_adjust(bottom=0.2)

if args.show:
    plt.show()
else:
    plt.savefig("graphics/features-datatypes.png")

# Plot the number of unique items per feature/column, also drop ip and type for comparision
plt.figure(figsize=(12, 5))
unique_counts = (
    df.select_dtypes(include=["object"])
    .drop(columns=["src_ip", "dst_ip", "type"])
    .nunique()
)
unique_counts.plot(kind="bar", color="#007698")
plt.title("Number of Unique Items per Categorical Features")
plt.xlabel("Feature/Column")
plt.ylabel("Number of Unique Items")
plt.xticks(rotation=45)
plt.figtext(0, 0, f"{args.data_set}", fontsize=10)
plt.subplots_adjust(bottom=0.3)

for i, count in enumerate(unique_counts):
    plt.text(i, count, str(count), ha="center", va="bottom")

if args.show:
    plt.show()
else:
    plt.savefig("graphics/features-categorical-unique.png")


# same for integer values
plt.figure(figsize=(12, 5))
unique_counts = df.select_dtypes(include=["int64"]).nunique()
unique_counts.plot(kind="bar", color="#007698")
plt.title("Number of Unique Items per Integer Features")
plt.xlabel("Feature/Column")
plt.ylabel("Number of Unique Items")
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

for i, count in enumerate(unique_counts):
    plt.text(i, count, str(count), ha="center", va="bottom")

if args.show:
    plt.show()
else:
    plt.savefig("graphics/features-integer-unique.png")

# Print the unique count of the 'type'/'label' field
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1]})

# Multiclass Class Distribution
type_counts = df["type"].value_counts()
type_counts.plot(kind="bar", color=["#007698"], ax=ax1)
ax1.set_title("Multiclass Class Distribution")
ax1.set_xlabel("Type")
ax1.set_ylabel("Flow Count")
ax1.set_ylim(0, type_counts.max() * 1.1)
for i, count in enumerate(type_counts):
    percentage = count / type_counts.sum() * 100
    ax1.text(i, count, f"{percentage:.2f}%\n{count}", ha="center", va="bottom", fontsize=10)

# Binary Class Distribution
type_counts = df["label"].value_counts()
type_counts.index = type_counts.index.map({False: "0 - benign", True: "1 - attack"})
type_counts.plot(kind="bar", color=["#007698"], ax=ax2)
ax2.set_title("Binary Class Distribution")
ax2.set_xlabel("Type")
ax2.set_ylabel("Flow Count")
ax2.set_ylim(0, type_counts.max() * 1.1)
ax2.set_xticks(range(len(type_counts)))
ax2.set_xticklabels(type_counts.index, rotation=0)
for i, count in enumerate(type_counts):
    percentage = count / type_counts.sum() * 100
    ax2.text(i, count, f"{percentage:.2f}%\n{count}", ha="center", va="bottom", fontsize=10)

plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distributions.png")

# or pie chart?
# plt.figure(figsize=(7, 7))
# type_counts = df['label'].value_counts()
# type_counts.index = type_counts.index.map({False: '0 - benign', True: '1 - attack'})
# type_counts.plot(kind='pie', color=['#007698','#211a51'])
# plt.title("Binary Class Distribution")
# plt.figtext(0, 0, f"{args.data_set}", fontsize=8)
# plt.subplots_adjust(bottom=0.2)
# plt.pie(type_counts, autopct='%1.1f%% ')
print(df["type"].value_counts(normalize=True))
print(df["label"].value_counts(normalize=True))


# bar chart by proto, split by label
proto_label_counts = df.groupby(['proto', 'label']).size().unstack(fill_value=0)
proto_label_counts = proto_label_counts.loc[proto_label_counts.sum(axis=1).sort_values(ascending=False).index]
proto_label_counts.drop(index='icmp', errors='ignore').plot(kind='barh', stacked=True, figsize=(14, 3), color=["#007698","#ff8c00"])
plt.title("Service Distribution by Proto")
plt.xlabel("Flow Count")
plt.ylabel("Protocol")
plt.legend(title="Label", labels=["Benign", "Attack"])
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
for i, (proto, row) in enumerate(proto_label_counts.iterrows()):
    cumulative_total = 0  # To track the cumulative position for stacking
    for j, count in enumerate(row):
        if count > 0:  # Only label if there's a positive count
            cumulative_total += count
            percentage = count / proto_label_counts.values.sum() * 100
            if percentage >= 1:  # Only add labels for significant bars
                plt.text(
                    cumulative_total - count / 2,  # Place label at the center of the current bar segment
                    i, 
                    f"{percentage:.0f}%\n{count}", ha="center", va="center", fontsize=10)

if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distribution-proto.png")

# bar chart by proto, split by label
service_label_counts = df.groupby(['service', 'label']).size().unstack(fill_value=0)
service_label_counts = service_label_counts.loc[service_label_counts.sum(axis=1).sort_values(ascending=False).index]
service_label_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=["#007698","#ff8c00"])
plt.title("Service Distribution by Label")
plt.xlabel("Protocol")
plt.ylabel("Flow Count")
plt.legend(title="Label", labels=["Benign", "Attack"])
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
for i, (port, row) in enumerate(proto_label_counts.iterrows()):
    total = row.sum()
    for j, count in enumerate(row):
        percentage = count / df['dst_port'].value_counts().sum() * 100
        if percentage >= 1:
            plt.text(i, count, f"{percentage:.0f}%\n{count}", ha="center", va="bottom", fontsize=10)
if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distribution-service.png")    


# bar chart of top 10 dst_ports, splited by label
top10_dst_ports = df['dst_port'].value_counts().nlargest(10).index
dst_port_label_counts = df[df['dst_port'].isin(top10_dst_ports)].groupby(['dst_port', 'label']).size().unstack(fill_value=0)
dst_port_label_counts = dst_port_label_counts.loc[dst_port_label_counts.sum(axis=1).sort_values(ascending=False).index]
dst_port_label_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=["#007698","#ff8c00"])
plt.title("Top 10 Destination Destination Ports")
plt.xlabel("Destination Port")
plt.ylabel("Flow Count")
plt.legend(title="Label", labels=["Benign", "Attack"])
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
for i, (port, row) in enumerate(dst_port_label_counts.iterrows()):
    total = row.sum()
    for j, count in enumerate(row):
        percentage = count / df['dst_port'].value_counts().sum() * 100
        if percentage >= 1:
            plt.text(i, count, f"{percentage:.0f}%\n{count}", ha="center", va="bottom", fontsize=10)
if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distribution-top10-dstport.png")

if args.verbose > 0:
    print("Data types:\n")
    # Print all labels with their data types
    for column in df.columns:
        print(f"{column}: {df[column].dtype}")        



# bar chart of src_ip, split by label
top10_src_ips = df['src_ip'].value_counts().nlargest(10).index
src_ip_label_counts = df[df['src_ip'].isin(top10_src_ips)].groupby(['src_ip', 'label']).size().unstack(fill_value=0)
src_ip_label_counts = src_ip_label_counts.loc[src_ip_label_counts.sum(axis=1).sort_values(ascending=False).index]
src_ip_label_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=["#007698", "#ff8c00"])
plt.title("Source Top10 IP Distribution by Label")
plt.xlabel("Source IP")
plt.ylabel("Flow Count")
plt.legend(title="Label", labels=["Benign", "Attack"])
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
for i, (ip, row) in enumerate(src_ip_label_counts.iterrows()):
    total = row.sum()
    for j, count in enumerate(row):
        percentage = count / df['src_ip'].value_counts().sum() * 100
        if percentage >= 1:
            plt.text(i, count, f"{percentage:.0f}%\n{count}", ha="center", va="bottom", fontsize=10)
if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distribution-srcip.png")


# bar chart of dst_ip, split by label
top10_dst_ips = df['dst_ip'].value_counts().nlargest(10).index
dst_ip_label_counts = df[df['dst_ip'].isin(top10_dst_ips)].groupby(['dst_ip', 'label']).size().unstack(fill_value=0)
dst_ip_label_counts = dst_ip_label_counts.loc[dst_ip_label_counts.sum(axis=1).sort_values(ascending=False).index]
dst_ip_label_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=["#007698", "#ff8c00"])
plt.title("Destination Top10 IP Distribution by Label")
plt.xlabel("Source IP")
plt.ylabel("Flow Count")
plt.legend(title="Label", labels=["Benign", "Attack"])
plt.figtext(0.01, 0.01, f"{args.data_set}", fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
for i, (ip, row) in enumerate(dst_ip_label_counts.iterrows()):
    total = row.sum()
    for j, count in enumerate(row):
        percentage = count / df['dst_ip'].value_counts().sum() * 100
        if percentage >= 1:
            plt.text(i, count, f"{percentage:.0f}%\n{count}", ha="center", va="bottom", fontsize=10)
if args.show:
    plt.show()
else:
    plt.savefig("graphics/label-distribution-dstip.png")



df.to_csv(os.path.splitext(args.data_set)[0] + "_clean.csv", index=False)
df.dtypes.astype(str).to_json(
    os.path.splitext(args.data_set)[0] + "_clean_datatype.json"
)

# alternative to save as pickle with the data types
df.to_pickle(os.path.splitext(args.data_set)[0] + "_clean.pkl")
