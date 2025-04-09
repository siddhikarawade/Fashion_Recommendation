import pymysql
from config import Config

# Connect to MySQL Server (without selecting a database)
conn = pymysql.connect(
    host=Config.MYSQL_HOST,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD
)

cursor = conn.cursor()

# Step 1: Create Database if not exists
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DB};")
print(f"Database '{Config.MYSQL_DB}' created or already exists.")

# Select the newly created database
cursor.execute(f"USE {Config.MYSQL_DB};")

# Step 2: Create 'users' Table
create_users_table = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone_number VARCHAR(15) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

cursor.execute(create_users_table)
print("Table 'users' created or already exists.")

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()
print("Database initialization complete!")
