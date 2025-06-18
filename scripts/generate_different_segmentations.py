import os
import sqlite3
import pandas as pd
from faker import Faker
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ------------------------------------
# Data generation core (same as before)
# ------------------------------------

def generate_behavioral_segmented_db(db_name, segment_profiles, n_leads=10000, clustering="kmeans"):
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)
    email_providers = ['gmail.com', 'hotmail.com', 'yahoo.com']

    leads_list = []
    transactions = []
    used_emails = set()

    for profile in segment_profiles:
        count = int(n_leads * profile["prob"])
        emails = set()
        selected_names = []

        while len(emails) < count:
            name = fake.name()
            base_email = f"{name.lower().replace(' ', '.')}@{random.choice(email_providers)}"
            email = base_email
            suffix = 1
            while email in used_emails:
                email = base_email.replace("@", f"{suffix}@")
                suffix += 1
            used_emails.add(email)
            emails.add(email)
            selected_names.append(name)

        emails = list(emails)

        p1 = np.clip(np.random.normal(profile["p1_mean"], profile["p1_std"], count), 0, 1)
        member_rating = np.clip(np.random.normal(profile["rating_mean"], profile["rating_std"], count), 1, 5)

        leads = pd.DataFrame({
            'user_full_name': selected_names,
            'user_email': emails,
            'member_rating': np.round(member_rating, 2),
            'optin_days': np.random.randint(-800, 0, size=count),
            'p1': np.round(p1, 3)
        })
        leads_list.append(leads)

        for email in emails:
            purchase_count = max(0, int(np.random.normal(profile["purchase_freq_mean"], profile["purchase_freq_std"])))
            for _ in range(purchase_count):
                transactions.append({
                    "user_email": email,
                    "purchased_at": fake.date_time_between(start_date='-2y', end_date='now')
                })

    leads_df = pd.concat(leads_list, ignore_index=True)
    transactions_df = pd.DataFrame(transactions)
    purchase_freq = transactions_df.groupby("user_email").size().reset_index(name="purchase_frequency")
    customer_features = leads_df[["user_email", "p1", "member_rating"]].merge(
        purchase_freq, on="user_email", how="left"
    )
    customer_features["purchase_frequency"] = customer_features["purchase_frequency"].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_features[["purchase_frequency", "p1", "member_rating"]])
    n_clusters = len(segment_profiles)

    if clustering == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        leads_df["segment"] = model.fit_predict(X_scaled)
        algo_used = "kmeans"

    elif clustering == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        leads_df["segment"] = model.fit_predict(X_scaled)
        algo_used = "gmm"

    elif clustering == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        leads_df["segment"] = model.fit_predict(X_scaled)
        algo_used = "hierarchical"

    else:
        raise ValueError(f"Unsupported clustering algorithm: {clustering}")

    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", db_name)
    conn = sqlite3.connect(db_path)
    leads_df.to_sql("leads_scored", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Created ({algo_used}): {db_path}")


# ------------------------------------
# Full scenario definitions
# ------------------------------------

scenarios = [

    # Case 1 — High vs Low Engagement
    {
        "db_prefix": "leads_seg_case1_high_low_engagement",
        "profiles": [
            {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 4.8, "rating_std": 0.2, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.33},
            {"p1_mean": 0.8, "p1_std": 0.05, "rating_mean": 2.0, "rating_std": 0.4, "purchase_freq_mean": 1, "purchase_freq_std": 0.5, "prob": 0.33},
            {"p1_mean": 0.2, "p1_std": 0.1, "rating_mean": 1.5, "rating_std": 0.3, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.34}
        ]
    },

    # Case 2 — Loyalty vs Price Sensitivity
    {
        "db_prefix": "leads_seg_case2_price_vs_loyalty",
        "profiles": [
            {"p1_mean": 0.85, "p1_std": 0.05, "rating_mean": 4.8, "rating_std": 0.2, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.25},
            {"p1_mean": 0.7, "p1_std": 0.05, "rating_mean": 3.5, "rating_std": 0.3, "purchase_freq_mean": 3, "purchase_freq_std": 0.8, "prob": 0.25},
            {"p1_mean": 0.4, "p1_std": 0.1, "rating_mean": 4.0, "rating_std": 0.3, "purchase_freq_mean": 2, "purchase_freq_std": 0.8, "prob": 0.25},
            {"p1_mean": 0.2, "p1_std": 0.1, "rating_mean": 1.5, "rating_std": 0.3, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.25}
        ]
    },

    # Case 3 — Cluster + Outlier
    {
        "db_prefix": "leads_seg_case3_cluster_outlier",
        "profiles": [
            {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 4.9, "rating_std": 0.2, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.3},
            {"p1_mean": 0.1, "p1_std": 0.05, "rating_mean": 1.2, "rating_std": 0.2, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.3},
            {"p1_mean": 0.5, "p1_std": 0.1, "rating_mean": 3.0, "rating_std": 0.3, "purchase_freq_mean": 1, "purchase_freq_std": 0.8, "prob": 0.4}
        ]
    },

    # Case 4 — Central Mass with Small Niches
    {
        "db_prefix": "leads_seg_case4_narrow_middle",
        "profiles": [
            {"p1_mean": 0.5, "p1_std": 0.05, "rating_mean": 3.0, "rating_std": 0.2, "purchase_freq_mean": 2, "purchase_freq_std": 0.8, "prob": 0.5},
            {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 4.8, "rating_std": 0.2, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.2},
            {"p1_mean": 0.1, "p1_std": 0.05, "rating_mean": 1.0, "rating_std": 0.2, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.2},
            {"p1_mean": 0.3, "p1_std": 0.05, "rating_mean": 2.5, "rating_std": 0.2, "purchase_freq_mean": 1, "purchase_freq_std": 0.5, "prob": 0.1}
        ]
    },

    # Case 5 — Conversion Readiness
    {
        "db_prefix": "leads_seg_case5_conversion_ready",
        "profiles": [
            {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 4.9, "rating_std": 0.1, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.3},
            {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 1.5, "rating_std": 0.3, "purchase_freq_mean": 1, "purchase_freq_std": 0.5, "prob": 0.4},
            {"p1_mean": 0.2, "p1_std": 0.1, "rating_mean": 1.0, "rating_std": 0.2, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.3}
        ]
    }
]

# ------------------------------------
# Generate all combinations!
# ------------------------------------

if __name__ == "__main__":

    clustering_algos = ["kmeans", "gmm", "hierarchical"]
    n_leads_per_scenario = 10000

    for scenario in scenarios:
        for algo in clustering_algos:
            filename = f"{scenario['db_prefix']}_{algo}.db"
            generate_behavioral_segmented_db(
                db_name=filename,
                segment_profiles=scenario["profiles"],
                n_leads=n_leads_per_scenario,
                clustering=algo
            )
