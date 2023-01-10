from flask import Flask, Blueprint, render_template, request, redirect, url_for, session, flash, g
#from _dblayer.dbclient_user import UserDbClient
from DB.db import DB
import os
from dotenv import load_dotenv
import pandas as pd
import io
import random
import base64
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter, DayLocator
import plotly
import plotly.express as px
import yfinance as yf
import json

# blueprint construct
navigation_bp = Blueprint("navigation", __name__)

@navigation_bp.route('/', methods=['GET','POST'])
def home():
        #if use AWS, have to store DB_PASSWORD into AWS or inject password in python environment 
    load_dotenv()
    #stores db connection as a global object
    return render_template('pages/home.html')

#used to setup the database (creation of database, table and seeding data)
@navigation_bp.route('/create_db', methods=['GET','POST'])
def create_db ():
    connection = DB.create_server_connection()
    #if use AWS, have to store DB_PASSWORD into AWS or inject password in python environment 
    create_database_query = "CREATE DATABASE stocks"
    #DB.create_database(connection, create_database_query)
    #DB.create_table(connection)
    #DB.seed_data(connection)
    return render_template('pages/home.html')


@navigation_bp.route('/data', methods=['GET','POST'])
def data():
    try:
        posts_data = DB.read_all_data()
        print("posts data", posts_data)
        return render_template('pages/data.html', POSTS=posts_data)
    except  BaseException as error:
        print("ERR unable to get all posts data ", error)
        return render_template('pages/data.html')

@navigation_bp.route('/callback/<endpoint>', methods=['GET','POST'])
def cb(endpoint):   
    if endpoint == "getStock":
        return gm(request.args.get('data'),request.args.get('period'),request.args.get('interval'))
    elif endpoint == "getInfo":
        stock = request.args.get('data')
        st = yf.Ticker(stock)
        return json.dumps(st.info)
    else:
        return "Bad endpoint", 400

# Return the JSON data for the Plotly graph
def gm(stock,period, interval):
    st = yf.Ticker(stock)
  
    # Create a line graph
    df = st.history(period=(period), interval=interval)
    df=df.reset_index()
    df.columns = ['Date-Time']+list(df.columns[1:])
    max = (df['Open'].max())
    min = (df['Open'].min())
    range = max - min
    margin = range * 0.05
    max = max + margin
    min = min - margin
    fig = px.area(df, x='Date-Time', y="Open",
        hover_data=("Open","Close","Volume"), 
        range_y=(min,max), template="seaborn" )


    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

'''  
@navigation_bp.route('/plot.png', methods=['GET', 'POST'])
def plot_png():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("title")
    axis.set_xlabel("x-axis")
    axis.set_ylabel("y-axis")
    axis.grid()
    axis.legend(loc='upper left')
    axis.plot(range(5), range(5), "ro-")
    # Helpers to format and locate ticks for dates
    axis.xaxis.set_major_locator(DayLocator())
    axis.xaxis.set_major_formatter(DateFormatter('%m/%d'))

    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    return render_template("pages/dashboard.html", image=pngImageB64String)
'''

