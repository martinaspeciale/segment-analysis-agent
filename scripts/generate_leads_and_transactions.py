import pandas as pd
import numpy as np
from faker import Faker
import random
import os
import sqlite3
from datetime import datetime

# Initialize Faker and seeds
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

n_leads = 1_000_000
n_transactions = 500_000

# Save DB in ../data/ relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
db_file = os.path.join(script_dir, "..", "data", "leads_scored_segmentation.db")

# Predefined choices
country_codes = ['us', 'in', 'mx', 'co', 'nl', 'sg', 'other']
email_providers = ['gmail.com', 'hotmail.com', 'yahoo.com']
ratings = [1, 2, 3, 4, 5]

# Create data folder
os.makedirs("data", exist_ok=True)

# Generate leads with structured segments

def generate_leads(n):
    segment_definitions = {
        0: {"p1_range": (0.9, 1.0), "rating_range": (4.5, 5.0), "prob": 0.2},
        1: {"p1_range": (0.6, 0.7), "rating_range": (2.0, 3.0), "prob": 0.18},
        2: {"p1_range": (0.7, 0.8), "rating_range": (3.0, 4.0), "prob": 0.22},
        3: {"p1_range": (0.5, 0.6), "rating_range": (2.5, 3.5), "prob": 0.19},
        4: {"p1_range": (0.2, 0.4), "rating_range": (1.0, 2.0), "prob": 0.21},
    }

    leads_list = []
    used_emails = set()
    for segment_id, spec in segment_definitions.items():
        target_count = int(n * spec["prob"])
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
        leads = pd.DataFrame({
            'mailchimp_id': [fake.uuid4() for _ in range(count)],
            'user_full_name': selected_names,
            'user_email': emails,
            'member_rating': np.round(np.random.uniform(*spec["rating_range"], count)).astype(int),
            'optin_time': [fake.date_time_between(start_date='-5y', end_date='now') for _ in range(count)],
            'country_code': np.random.choice(country_codes, size=count, p=[0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.25]),
            'made_purchase': np.random.choice([0, 1], size=count, p=[0.85, 0.15]),
            'optin_days': np.random.randint(-800, 0, size=count),
            'email_provider': [email.split('@')[1] for email in emails],
            'predict': np.random.choice([0, 1], size=count, p=[0.9, 0.1]),
            'p0': np.random.rand(count),
            'p1': np.random.uniform(*spec["p1_range"], count),
            'segment': segment_id
        })
        leads_list.append(leads)

    return pd.concat(leads_list, ignore_index=True)

# Generate transactions based on segment frequency tendencies

def generate_transactions(leads_df):
    transactions = []
    leads_df_unique = leads_df.drop_duplicates(subset="user_email")
    user_info = leads_df_unique.set_index("user_email")[["user_full_name", "segment"]].to_dict("index")
    countries = ['US', 'IN', 'AU', 'NZ', 'GB', 'CA', 'MX', 'CO', 'NL', 'SG']
    product_ids = [21, 23, 31, 37, 41]

    for email, info in user_info.items():
        segment = info["segment"]
        if segment == 0:
            n_tx = np.random.poisson(4)
        elif segment == 1:
            n_tx = np.random.poisson(0.5)
        elif segment == 2:
            n_tx = np.random.poisson(2)
        elif segment == 3:
            n_tx = np.random.poisson(1)
        else:  # segment 4
            n_tx = np.random.poisson(0.1)

        for _ in range(n_tx):
            transactions.append({
                "transaction_id": fake.uuid4(),
                "user_email": email,
                "user_full_name": info["user_full_name"],
                "purchased_at": fake.date_time_between(start_date='-5y', end_date='now'),
                "charge_country": random.choice(countries),
                "product_id": random.choice(product_ids)
            })

    return pd.DataFrame(transactions)

# Run generation
print("🟡 Generating leads...")
leads_df = generate_leads(n_leads)

print("🟡 Generating transactions...")
transactions_df = generate_transactions(leads_df)

print("💾 Saving to SQLite...")
conn = sqlite3.connect(db_file)
leads_df.to_sql("leads_scored", conn, index=False, if_exists="replace")
transactions_df.to_sql("transactions", conn, index=False, if_exists="replace")
conn.close()

print(f"✅ Done! SQLite DB written to: {db_file}")