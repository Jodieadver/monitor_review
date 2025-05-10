import psycopg2
import pandas as pd
import openpyxl

# Remote server configuration
DB_CONFIG = {
    'dbname': 'monitor_db',
    'user': 'xxscarlett',
    'password': 'your_password',  # Add your password here
    'host': 'your_server_ip',    # Replace with your server IP
    'port': '5432'
}

# 建立连接
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("✅ Successfully connected to remote database!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# 建表SQL
create_table_query = """
CREATE TABLE IF NOT EXISTS monitor_reviews (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    brand TEXT,
    segmentation TEXT,
    pros TEXT,
    cons TEXT,
    country TEXT,
    text TEXT,
    pros_cleaned TEXT,
    cons_cleaned TEXT,
    pros_keywords TEXT[],
    cons_keywords TEXT[]
);
"""
try:
    cur.execute(create_table_query)
    conn.commit()
    print("✅ Table monitor_reviews created successfully!")
except Exception as e:
    print(f"❌ Table creation failed: {e}")
    conn.rollback()

# 输入Excel数据
try:
    excel_path = "processed_reviews.xlsx"
    df = pd.read_excel(excel_path)
    df = df.fillna('')
    print(f"✅ Successfully read Excel file, total {len(df)} records!")
except Exception as e:
    print(f"❌ Excel reading failed: {e}")
    exit(1)

insert_query = """
INSERT INTO monitor_reviews (
    model_name, brand, segmentation, pros, cons, country, text,
    pros_cleaned, cons_cleaned, pros_keywords, cons_keywords
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

try:
    for index, row in df.iterrows():
        data = (
            row['model'],
            row['brand'],
            row['segmentation'],
            row['pros'],
            row['cons'],
            row['country'],
            row['text'],
            row['pros_cleaned'],
            row['cons_cleaned'],
            row['pros_keywords'],
            row['cons_keywords']
        )
        cur.execute(insert_query, data)
    conn.commit()
    print("✅ All data inserted successfully!")
except Exception as e:
    print(f"❌ Data insertion failed: {e}")
    conn.rollback()

cur.close()
conn.close()

