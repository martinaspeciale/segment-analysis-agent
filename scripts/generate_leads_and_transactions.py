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

'''
 Settings
 - Zero or one or more transactions (most will have zero).
 - A sparse transaction history, which is typical in real datasets — not every user buys something.
'''
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

# Generate leads
def generate_leads(n):
    full_names = [fake.name() for _ in range(n)]
    emails = [
        f"{name.lower().replace(' ', '.')}@{random.choice(email_providers)}"
        for name in full_names
    ]

    leads = {
        'mailchimp_id': [fake.uuid4() for _ in range(n)],
        'user_full_name': full_names,
        'user_email': emails,
        'member_rating': np.random.choice(ratings, size=n, p=[0.1, 0.5, 0.2, 0.15, 0.05]),
        'optin_time': [fake.date_time_between(start_date='-5y', end_date='now') for _ in range(n)],
        'country_code': np.random.choice(country_codes, size=n, p=[0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.25]),
        'made_purchase': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        'optin_days': np.random.randint(-800, 0, size=n),
        'email_provider': [email.split('@')[1] for email in emails],
        'predict': np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        'p0': np.random.rand(n),
        'p1': np.random.rand(n),
        'segment': np.random.randint(0, 5, size=n)
    }

    return pd.DataFrame(leads)

# Generate transactions
def generate_transactions(leads_df, n):
    emails = leads_df['user_email'].tolist()
    names = leads_df.set_index('user_email')['user_full_name'].to_dict()
    countries = ['US', 'IN', 'AU', 'NZ', 'GB', 'CA', 'MX', 'CO', 'NL', 'SG']
    product_ids = [21, 23, 31, 37, 41]  # Example course/product IDs

    transaction_emails = [random.choice(emails) for _ in range(n)]

    transactions = {
        'transaction_id': [fake.uuid4() for _ in range(n)],
        'user_email': transaction_emails,
        'user_full_name': [names[email] for email in transaction_emails],
        'purchased_at': [fake.date_time_between(start_date='-5y', end_date='now') for _ in range(n)],
        'charge_country': [random.choice(countries) for _ in range(n)],
        'product_id': [random.choice(product_ids) for _ in range(n)]
    }

    return pd.DataFrame(transactions)


# Generate and save
print("🟡 Generating leads...")
leads_df = generate_leads(n_leads)

print("🟡 Generating transactions...")
transactions_df = generate_transactions(leads_df, n_transactions)

print("💾 Saving to SQLite...")
conn = sqlite3.connect(db_file)
leads_df.to_sql("leads_scored", conn, index=False, if_exists="replace")
transactions_df.to_sql("transactions", conn, index=False, if_exists="replace")
conn.close()

print(f"✅ Done! SQLite DB written to: {db_file}")
