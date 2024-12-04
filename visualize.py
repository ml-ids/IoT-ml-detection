
import pandas as pd
import matplotlib.pyplot as plt
import json

results_df = pd.read_csv('learn-compare.csv')
json.dump(results_df.to_dict(), open('learn-compare.json', 'w'))
results_df = pd.DataFrame(results_df)




# Bar plot for all metrics?
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


#print(plt.style.available)
#plt.style.use('fivethirtyeight')

#compute min max
metrics = ["Accuracy", "Precision", "F1-Score"]
y_min = results_df[metrics].min().min()-0.1  # Minimum of all metrics
y_max = results_df[metrics].max().max()+0.1  # Maximum of all metrics


# Accuracy plot
acc_plot = results_df.pivot(index="Model", columns="Method", values="Accuracy").plot(kind='bar', ax=axes[0])
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Model')
axes[0].set_ylim(y_min, y_max)
for container in acc_plot.containers:
    acc_plot.bar_label(
        container, fmt='%.6f', fontsize=8, label_type='center', padding=1, rotation=90
    )


# Precision plot
prec_plot = results_df.pivot(index="Model", columns="Method", values="Precision").plot(kind='bar', ax=axes[1])
axes[1].set_title('Precision Comparison')
axes[1].set_ylabel('Precision')
axes[1].set_xlabel('Model')
axes[1].set_ylim(y_min, y_max)
for container in prec_plot.containers:
    prec_plot.bar_label(
        container, fmt='%.6f', fontsize=8, label_type='center', padding=1, rotation=90
    )

# F1-Score plot
f1_plot = results_df.pivot(index="Model", columns="Method", values="F1-Score").plot(kind='bar', ax=axes[2])
axes[2].set_title('F1-Score Comparison')
axes[2].set_ylabel('F1-Score')
axes[2].set_xlabel('Model')
axes[2].set_ylim(y_min, y_max)
for container in f1_plot.containers:
    f1_plot.bar_label(
        container, fmt='%.6f', fontsize=8, label_type='center', padding=1, rotation=90
    )
for ax in axes:
    ax.legend(loc='lower left', ncols=3)
    ax.grid(axis='y', linestyle='--', alpha=0.2)

plt.tight_layout()
plt.show()


# or as as heatmap?
import seaborn as sns
pivot_data = results_df.pivot(index="Model", columns="Method")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


sns.heatmap(pivot_data["Accuracy"], annot=True, fmt=".6f", cmap="Blues", ax=axes[0])
axes[0].set_title('Accuracy Heatmap')
axes[0].set_xlabel('SMOTE')
axes[0].set_ylabel('Model')

sns.heatmap(pivot_data["Precision"], annot=True, fmt=".6f", cmap="Greens", ax=axes[1])
axes[1].set_title('Precision Heatmap')
axes[1].set_xlabel('SMOTE')
axes[1].set_ylabel('Model')

sns.heatmap(pivot_data["F1-Score"], annot=True, fmt=".6f", cmap="Reds", ax=axes[2])
axes[2].set_title('F1-Score Heatmap')
axes[2].set_xlabel('SMOTE')
axes[2].set_ylabel('Model')

plt.tight_layout()
plt.show()
