# import sqlite3

# # Connect to a database file. If it doesn't exist, it will be created.
# conn = sqlite3.connect('mydatabase.db') 

# # Create a cursor object
# cursor = conn.cursor()

# # Execute SQL commands (e.g., create a table)
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY,
#         name TEXT NOT NULL,
#         email TEXT UNIQUE
#     )
# ''')

# # Commit changes and close the connection
# conn.commit()
# conn.close()

