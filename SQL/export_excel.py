import sqlite3
import pandas as pd
import os

# Create excel folder if not exists
os.makedirs("excel", exist_ok=True)

# Connect DB
conn = sqlite3.connect("data/database/students.db")

# Read data
df = pd.read_sql("SELECT * FROM student_performance", conn)

# Save to Excel
df.to_excel("excel/student_report.xlsx", index=False)

conn.close()

print("Excel file created successfully!")
