import os
import sqlite3
import pandas as pd
from faker import Faker
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def generate_segmented_db(db_name, segment_definitions, n_leads=10000, n_clusters=3):
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)
    email_providers = ['gmail.com', 'hotmail.com', 'yahoo.com']
    
    leads_list = []
    used_emails = set()

    for segment_id, spec in segment_definitions.items():
        target_count = int(n_leads * spec["prob"])
        emails = set()
        selected_names = []

        while len(emails) < target_count:
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
        count = len(emails)
        # Introduce dependency between p1 and member_rating
        base_p1 = np.random.uniform(*spec["p1_range"], count)
        # member_rating is partly correlated with p1, plus noise
        base_rating = (base_p1 * 3.5) + np.random.uniform(-0.5, 1.0, count)
        base_rating = np.clip(base_rating, *spec["rating_range"])

        leads = pd.DataFrame({
            'user_full_name': selected_names,
            'user_email': emails,
            'member_rating': np.round(base_rating, 1),
            'optin_days': np.random.randint(-800, 0, size=count),
            'p1': np.round(base_p1, 3)
        })

        leads_list.append(leads)

    leads_df = pd.concat(leads_list, ignore_index=True)

    transactions = []
    for email, rating, p1 in zip(leads_df["user_email"], leads_df["member_rating"], leads_df["p1"]):
        # purchase frequency depends on both rating and p1
        expected_tx = (rating / 2.5) + (p1 * 2.0)
        n_tx = np.random.poisson(expected_tx)
        for _ in range(n_tx):
            transactions.append({
                "user_email": email,
                "purchased_at": fake.date_time_between(start_date='-2y', end_date='now')
            })

    transactions_df = pd.DataFrame(transactions)

    purchase_freq = transactions_df.groupby("user_email").size().reset_index(name="purchase_frequency")
    customer_features = leads_df[["user_email", "p1", "member_rating"]].merge(
        purchase_freq, on="user_email", how="left"
    )
    customer_features["purchase_frequency"] = customer_features["purchase_frequency"].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_features[["purchase_frequency", "p1", "member_rating"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    leads_df["segment"] = kmeans.fit_predict(X_scaled)

    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", db_name)
    conn = sqlite3.connect(db_path)
    leads_df.to_sql("leads_scored", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
    
    assert leads_df["user_email"].is_unique, "🚨 Duplicate emails in leads!"
    assert transactions_df["user_email"].isin(leads_df["user_email"]).all(), "🚨 Transactions contain unknown emails!"

    
    conn.close()
    print(f"✅ Created: {db_path}")

# ================================
# 🎯 Batch Generation Definitions
# ================================

scenarios = [
    {
        "db_name": "leads_seg_case1_high_low_engagement.db",
        "n_segments": 3,
        "definition": {
            0: {"p1_range": (0.85, 1.0), "rating_range": (4.5, 5.0), "prob": 0.3},
            1: {"p1_range": (0.4, 0.6), "rating_range": (2.5, 3.5), "prob": 0.4},
            2: {"p1_range": (0.0, 0.3), "rating_range": (1.0, 2.0), "prob": 0.3}
        }
    },
    {
        "db_name": "leads_seg_case2_price_vs_loyalty.db",
        "n_segments": 4,
        "definition": {
            0: {"p1_range": (0.8, 1.0), "rating_range": (4.5, 5.0), "prob": 0.25},
            1: {"p1_range": (0.7, 0.9), "rating_range": (1.5, 2.5), "prob": 0.25},
            2: {"p1_range": (0.2, 0.4), "rating_range": (4.0, 5.0), "prob": 0.25},
            3: {"p1_range": (0.2, 0.4), "rating_range": (1.0, 2.0), "prob": 0.25}
        }
    },
    {
        "db_name": "leads_seg_case3_cluster_outlier.db",
        "n_segments": 3,
        "definition": {
            0: {"p1_range": (0.9, 1.0), "rating_range": (4.8, 5.0), "prob": 0.3},
            1: {"p1_range": (0.1, 0.2), "rating_range": (1.0, 1.5), "prob": 0.3},
            2: {"p1_range": (0.0, 1.0), "rating_range": (1.0, 5.0), "prob": 0.4}
        }
    },
    {
        "db_name": "leads_seg_case4_narrow_middle.db",
        "n_segments": 4,
        "definition": {
            0: {"p1_range": (0.45, 0.55), "rating_range": (2.5, 3.5), "prob": 0.6},
            1: {"p1_range": (0.8, 1.0), "rating_range": (4.5, 5.0), "prob": 0.15},
            2: {"p1_range": (0.0, 0.2), "rating_range": (1.0, 2.0), "prob": 0.15},
            3: {"p1_range": (0.5, 0.6), "rating_range": (2.8, 3.2), "prob": 0.1}
        }
    },
    {
        "db_name": "leads_seg_case5_conversion_ready.db",
        "n_segments": 3,
        "definition": {
            0: {"p1_range": (0.85, 1.0), "rating_range": (4.5, 5.0), "prob": 0.25},
            1: {"p1_range": (0.85, 1.0), "rating_range": (1.0, 2.0), "prob": 0.35},
            2: {"p1_range": (0.0, 0.3), "rating_range": (1.0, 2.0), "prob": 0.4}
        }
    }
]

if __name__ == "__main__":
    for case in scenarios:
        generate_segmented_db(
            db_name=case["db_name"],
            segment_definitions=case["definition"],
            n_leads=10000,
            n_clusters=case["n_segments"]
        )

