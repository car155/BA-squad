# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from config import Config
import datetime
from __init__ import create_app

app = create_app()

# main driver function
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=int(Config.APP_PORT), debug=Config.DEBUG)