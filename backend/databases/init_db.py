import os
import sys

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import db, User
from werkzeug.security import generate_password_hash
from datetime import datetime
from flask import Flask
from backend.config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()
    # Check if admin user exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        # Create default admin user
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin'),
            full_name='Administrator',
            email='admin@example.com',
            role='admin',
            is_active=True,
            created_at=datetime.utcnow()
        )
        db.session.add(admin)
        db.session.commit()
        print("Created default admin user")
    print("Database initialized successfully")
