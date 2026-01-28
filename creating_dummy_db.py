import csv
import mysql.connector
import random

def random_popularity():
    return random.randint(30, 95)

def random_rating():
    return round(random.uniform(3.0, 4.8), 1)

def random_prestige():
    return random.randint(3, 8)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="qwerty",
    database="internship_db"
)
cursor = conn.cursor()
insert_query = """
INSERT INTO internships
(title, company, skills, location, mode, source,
 popularity, rating, company_prestige)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
csv_path = r"D:\projects\intern_recommend_0.0\dataset_collection\scraped_internship.csv"
rows = []
with open(csv_path, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append((
            row["title"],
            row["company"],
            row["skills"],
            row["location"],
            row["mode"],
            row["source"],
            random_popularity(),
            random_rating(),
            random_prestige()
        ))

cursor.executemany(insert_query, rows)
conn.commit()

cursor.close()
conn.close()

print(f"Imported {len(rows)} rows successfully with random demo values.")
