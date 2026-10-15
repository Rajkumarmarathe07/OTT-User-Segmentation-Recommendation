
import pandas as pd

# Load original customer data and clustering results
original_df = pd.read_csv("data/ott_users.csv")
clustered_df = pd.read_csv("data/ott_users_clustered.csv")

# Combine customer details with cluster labels
df = original_df.merge(
    clustered_df[["customer_id", "cluster"]],
    on="customer_id",
    how="inner"
)

print("Customer data loaded successfully:", df.shape)

# Genre recommendation mapping
genre_recommendations = {
    "Action": ["Sci-Fi", "Drama", "Horror"],
    "Comedy": ["Romance", "Drama", "Action"],
    "Documentary": ["Drama", "Sci-Fi", "Comedy"],
    "Drama": ["Romance", "Documentary", "Sci-Fi"],
    "Horror": ["Action", "Sci-Fi", "Drama"],
    "Romance": ["Drama", "Comedy", "Documentary"],
    "Sci-Fi": ["Action", "Documentary", "Horror"]
}


def recommend_genres(customer_id):
    user = df[df["customer_id"].astype(str) == str(customer_id)]

    if user.empty:
        print("Customer ID not found.")
        return

    user = user.iloc[0]

    genre = user["favorite_genre"]
    cluster = user["cluster"]
    recommendations = genre_recommendations.get(genre, [])

    print("\n===== OTT USER RECOMMENDATIONS =====")
    print("Customer ID:", customer_id)
    print("Customer Segment:", cluster)
    print("Favorite Genre:", genre)
    print("Recommended Genres:", ", ".join(recommendations))

    if user["churned"] == 1:
        print("Status: Churned user in the dataset")
        print("Suggestion: Consider personalized content or a retention offer.")
    else:
        print("Status: Not marked as churned in the dataset")


# Test with the first customer
sample_customer_id = df["customer_id"].iloc[0]
recommend_genres(sample_customer_id)

# Search for other customers
while True:
    customer_id = input(
        "\nEnter a customer ID (or type 'exit' to stop): "
    ).strip()

    if customer_id.lower() == "exit":
        break

    recommend_genres(customer_id)
