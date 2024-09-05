import pandas as pd
import re

def load_data(file_path):
    return pd.read_excel(file_path)

def generate_email(name):
    name_parts = name.split(', ')
    last_name = name_parts[0].strip()
    first_name = name_parts[1].strip() if len(name_parts) > 1 else ""
    email = f"{first_name.lower()}{last_name.lower()}@gmail.com"
    email = re.sub(r'[^a-z0-9]', '', email)  # Remove special characters
    return email

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

def save_to_tsv(df, file_path):
    df.to_csv(file_path, sep='\t', index=False)
