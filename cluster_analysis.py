
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphs", exist_ok=True)

# Load original and clustered datasets
original = pd.read_csv("data/ott_users.csv")
clustered = pd.read_csv("data/ott_users_clustered.csv")

# Add cluster labels to original customer records
df = original.merge(
    clustered[["customer_id", "cluster"]],
    on="customer_id",
    how="inner"
)

# Analyze average customer behavior in each cluster
summary = df.groupby("cluster").agg(
    user_count=("customer_id", "count"),
    avg_age=("age", "mean"),
    avg_watch_hours=("watch_hours", "mean"),
    avg_last_login_days=("last_login_days", "mean"),
    avg_monthly_fee=("monthly_fee", "mean"),
    avg_daily_watch_time=("avg_watch_time_per_day", "mean"),
    churn_rate=("churned", "mean")
).round(2)

print("\nCLUSTER ANALYSIS")
print(summary.to_string())

# Find the most common genre in each cluster
print("\nMOST POPULAR GENRE IN EACH CLUSTER")
print(
    df.groupby("cluster")["favorite_genre"]
    .agg(lambda x: x.mode().iloc[0])
)

# Plot average watch hours
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="cluster", y="watch_hours", errorbar=None)
plt.title("Average Watch Hours by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Watch Hours")
plt.tight_layout()
plt.savefig("graphs/cluster_watch_hours.png")
plt.show()

# Plot churn rate
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="cluster", y="churned", errorbar=None)
plt.title("Churn Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("graphs/cluster_churn_rate.png")
plt.show()

summary.to_csv("data/cluster_summary.csv")
print("\nAnalysis saved to data/cluster_summary.csv")
