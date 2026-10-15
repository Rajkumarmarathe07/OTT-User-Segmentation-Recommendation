
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/ott_users.csv")

os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Favorite Genre Distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="favorite_genre", hue="favorite_genre", legend=False)
plt.title("OTT Users by Favorite Genre")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/favorite_genre.png")
plt.show()

# 2. Subscription Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="subscription_type",
              hue="subscription_type", legend=False)
plt.title("OTT Users by Subscription Type")
plt.tight_layout()
plt.savefig("graphs/subscription_type.png")
plt.show()

# 3. Watch Hours Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="watch_hours", bins=20, kde=True)
plt.title("OTT Watch Hours Distribution")
plt.tight_layout()
plt.savefig("graphs/watch_hours.png")
plt.show()

# 4. Average Watch Hours by Genre
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="favorite_genre", y="watch_hours")
plt.title("Average Watch Hours by Favorite Genre")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/watch_hours_by_genre.png")
plt.show()

print("EDA completed successfully!")
print("Graphs saved inside the graphs folder.")
