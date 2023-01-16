from flask import Flask
from flask.helpers import get_root_path
import datetime
from config import Config
from controller.navigation import navigation_bp
import os
import pandas as pd

def create_app():
    app = Flask(__name__)
    app.config["ENV"] = Config.FLASK_ENV
    app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(minutes=Config.SESSION_LIFETIME)
    app.config["config"] = Config
    app.secret_key = os.environ.get("SECRET_KEY")
    app.register_blueprint(navigation_bp)
        
    return app

