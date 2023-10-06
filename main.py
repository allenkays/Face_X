#!/usr/bin/env python3
"""
This script runs the application
"""
from flask import Flask
import secrets
from routes import main_routes

app = Flask(__name__)
app.register_blueprint(main_routes)


if __name__ == '__main__':
    SECRET_KEY = secrets.token_hex(16)
    app.secret_key = SECRET_KEY
    app.run(debug=True)
