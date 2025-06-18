import os
import sqlite3
import pandas as pd
from faker import Faker
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ========================================
# 1️⃣ Naive random segmentation (original)
# ========================================
def generate_naive_segmented_db(db_name, segment_definitions, n_leads=10000, n_clusters=3):
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)
    email_providers = ['gmail.com', 'hotmail.com', 'yahoo.com']
    
    leads_list = []
    used_emails = set()

    for segment_id, spec in segment_definitions.items():
        count = int(n_leads * spec["prob"])
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
        p1 = np.random.uniform(*spec["p1_range"], count)
        rating = np.random.uniform(*spec["rating_range"], count)

        leads = pd.DataFrame({
            'user_full_name': selected_names,
            'user_email': emails,
            'member_rating': np.round(rating, 1),
            'optin_days': np.random.randint(-800, 0, size=count),
            'p1': np.round(p1, 3)
        })
        leads_list.append(leads)

    leads_df = pd.concat(leads_list, ignore_index=True)

    transactions = []
    for email, rating in zip(leads_df["user_email"], leads_df["member_rating"]):
        n_tx = np.random.poisson(rating / 2.5)
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
    conn.close()

    print(f"✅ Created (naive): {db_path}")


# ========================================
# 2️⃣ Behavioral generation + clustering
# ========================================
def generate_behavioral_segmented_db(db_name, segment_profiles, n_leads=10000):
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    leads_df["segment"] = kmeans.fit_predict(X_scaled)

    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", db_name)
    conn = sqlite3.connect(db_path)
    leads_df.to_sql("leads_scored", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Created (behavioral + clustering): {db_path}")


# ========================================
# 3️⃣ Supervised ground-truth generator
# ========================================
def generate_groundtruth_segmented_db(db_name, segment_profiles, n_leads=10000):
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)
    email_providers = ['gmail.com', 'hotmail.com', 'yahoo.com']

    leads_list = []
    transactions = []
    used_emails = set()

    for segment_id, profile in enumerate(segment_profiles):
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
            'p1': np.round(p1, 3),
            'segment': segment_id
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

    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", db_name)
    conn = sqlite3.connect(db_path)
    leads_df.to_sql("leads_scored", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Created (ground-truth): {db_path}")

# ========================================
# 👇 Define your example profiles here
# ========================================

behavioral_profiles = [
    {"p1_mean": 0.9, "p1_std": 0.05, "rating_mean": 4.8, "rating_std": 0.2, "purchase_freq_mean": 5, "purchase_freq_std": 1, "prob": 0.25},
    {"p1_mean": 0.8, "p1_std": 0.05, "rating_mean": 2.0, "rating_std": 0.4, "purchase_freq_mean": 1, "purchase_freq_std": 0.5, "prob": 0.25},
    {"p1_mean": 0.2, "p1_std": 0.1, "rating_mean": 1.5, "rating_std": 0.3, "purchase_freq_mean": 0, "purchase_freq_std": 0.5, "prob": 0.25},
    {"p1_mean": 0.4, "p1_std": 0.1, "rating_mean": 4.0, "rating_std": 0.3, "purchase_freq_mean": 3, "purchase_freq_std": 1, "prob": 0.25}
]

naive_definition = {
    0: {"p1_range": (0.85, 1.0), "rating_range": (4.5, 5.0), "prob": 0.33},
    1: {"p1_range": (0.4, 0.6), "rating_range": (2.5, 3.5), "prob": 0.33},
    2: {"p1_range": (0.0, 0.3), "rating_range": (1.0, 2.0), "prob": 0.34}
}

# ========================================
# 🚀 Run all three generation modes
# ========================================

if __name__ == "__main__":
    generate_naive_segmented_db(
        db_name="leads_seg_case_naive.db",
        segment_definitions=naive_definition,
        n_leads=10000,
        n_clusters=3
    )

    generate_behavioral_segmented_db(
        db_name="leads_seg_case_behavioral.db",
        segment_profiles=behavioral_profiles,
        n_leads=10000
    )

    generate_groundtruth_segmented_db(
        db_name="leads_seg_case_groundtruth.db",
        segment_profiles=behavioral_profiles,
        n_leads=10000
    )
