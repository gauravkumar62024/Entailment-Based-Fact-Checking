import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load files
with open("fixed_classified_claims1.json", "r") as f1:
    cluster_data = json.load(f1)

with open("predictions_llama_123.json", "r") as f2:
    prediction_data = json.load(f2)

with open("misclassified_llama_123.json", "r") as f3:
    misclassified_data = json.load(f3)

# Convert to DataFrames
df_cluster = pd.DataFrame(cluster_data)[["claim", "category"]]
df_predictions = pd.DataFrame(prediction_data)[["claim", "true_label", "predicted_label"]]
df_misclassified = pd.DataFrame(misclassified_data)[["claim"]]

# Merge and mark misclassified
df_merged = pd.merge(df_predictions, df_cluster, on="claim", how="left")
df_merged["is_misclassified"] = df_merged["claim"].isin(df_misclassified["claim"])

# Compute stats
group_stats = df_merged.groupby("category").agg(
    Total_Claims=("claim", "count"),
    Misclassified=("is_misclassified", "sum"),
)
group_stats["Correctly_Classified"] = group_stats["Total_Claims"] - group_stats["Misclassified"]
group_stats["Accuracy (%)"] = (group_stats["Correctly_Classified"] / group_stats["Total_Claims"] * 100).round(2)
group_stats = group_stats.sort_values(by="Accuracy (%)").reset_index()

# Save CSV
group_stats.to_csv("cluster_veracity_error_analysis.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=group_stats, x="category", y="Accuracy (%)", palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.title("Accuracy of Veracity Prediction per Category")
plt.tight_layout()
plt.savefig("cluster_veracity_accuracy_plot.png")
plt.show()

