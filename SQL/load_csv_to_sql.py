import pandas as pd
import sqlite3
import os

print("🚀 Script started...")

# CSV path (YOUR path)
csv_path = r"C:\Users\jeeva\Desktop\ML Project\notebook\Data\students data.csv"

# Database path
db_path = r"C:\Users\jeeva\Desktop\ML Project\data\database\students.db"

# Create folder if not exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

print("✅ CSV Loaded. Rows:", len(df))

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Insert into SQL
df.to_sql("student_performance", conn, if_exists="replace", index=False)

conn.close()

print("🎉 Data successfully inserted into SQL database!")
