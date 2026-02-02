import sqlite3

conn = sqlite3.connect("data/database/students.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM student_performance LIMIT 5")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
