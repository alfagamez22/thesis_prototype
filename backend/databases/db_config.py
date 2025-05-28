import os

# Absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SQLite database file path
SQLITE_DB_PATH = os.path.join(BACKEND_DIR, 'databases', 'employee_db.sqlite')

# Ensure the databases directory exists
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

class SqliteConfig:
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{SQLITE_DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    SQLALCHEMY_ECHO = False
