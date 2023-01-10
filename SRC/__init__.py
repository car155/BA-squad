from flask import Flask
from flask.helpers import get_root_path
import datetime
from config import Config
from controller.navigation import navigation_bp
from dashapp1.callbacks import register_callbacks
from dashapp1.layout import register_layout
import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd

def create_app():
    app = Flask(__name__)
    app.config["ENV"] = Config.FLASK_ENV
    app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(minutes=Config.SESSION_LIFETIME)
    app.config["config"] = Config
    app.secret_key = os.environ.get("SECRET_KEY")
    app.register_blueprint(navigation_bp)
    #register_dashapps(app)
        
    return app

def register_dashapps(app):

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    dashapp1 = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/dashboard/',
                         assets_folder=get_root_path(__name__) + '/dashboard/assets/',
                         meta_tags=[meta_viewport],
                         external_stylesheets=[dbc.themes.LUX])

    with app.app_context():
      dashapp1.title = 'Dashapp 1'
      register_layout(dashapp1)
      register_callbacks(dashapp1)




