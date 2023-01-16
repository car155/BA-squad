from flask import Flask, Blueprint, render_template, request, redirect, url_for, session, flash, g
#from _dblayer.dbclient_user import UserDbClient
from DB.db import DB

from dotenv import load_dotenv
import pandas as pd
import plotly
import json

import torch
from math import log, e
import numpy as np
import json
import plotly.graph_objects as go


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
    #if use AWS, have to store DB_PASSWORD into AWS or inject password in python environment 
    create_database_query = "CREATE DATABASE stocks"
    #DB.create_database(connection, create_database_query)
    #DB.create_table(connection)
    DB.seed_all_data()
    
    return render_template('pages/home.html')


@navigation_bp.route('/data', methods=['GET','POST'])
def data():
    try:
        posts_data = DB.read_all_data()
        return render_template('pages/data.html', POSTS=posts_data)
    except  BaseException as error:
        print("ERR unable to get all posts data ", error)
        return render_template('pages/data.html')

@navigation_bp.route('/dashboard', methods=['GET','POST'])
def dashboard():
    try:
        stock_list = None
        if session.get('dictionary') is None:
            dictonary = torch.load('data/out/Results.pth')
            
            des_dict = json.dumps(dictonary)
            session['dictionary'] = json.loads(des_dict)
        data = session.get('dictionary')
        
        stock_list = list(data.keys())
        return render_template('pages/dashboard.html', STOCKS=stock_list)
    except  BaseException as error:
        print("ERR unable to get dashboard ", error)
        return render_template('pages/dashboard.html', STOCKS=stock_list)

@navigation_bp.route('/callback/<endpoint>', methods=['GET','POST'])
def cb(endpoint):   
    if endpoint == "getStock":
        stocklist = request.args.get('data').split(",")
        print("stocks", stocklist)
        if len(stocklist) == 0 or (len(stocklist)==1 and stocklist[0] == ''):
            print('HERE')
            fig = go.Figure()
            fig = fig_layout(fig, ytitle= "", ytickfromat = None, xtitle= "Stocks",
                      legendtitle = "Different Stocks", type_of_plot = "Entropy Values", ticker="Tweets on Stocks")
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        dict_ent = loss(stocklist)
        
        fig = gm(dict_ent)
        fig = fig_layout(fig, ytitle= "", ytickfromat = None, xtitle= "Stocks",
                      legendtitle = "Different Stocks", type_of_plot = "Entropy Values", ticker="Tweets on Stocks")
                      
        # Create a JSON representation of the graph
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON
    
    else:
        return "Bad endpoint", 400

def loss(stocklist):
    data = torch.load('data/out/Results.pth')
    dict_ent = dict()
    base = None
    #data = session.get('dictionary')
    if data is None:
        return
    for stock in stocklist:
        stock_data = data.get(stock)
        
        total = stock_data.get('Total')
        stock_data.pop('Total')
        labels = [int(k) for k,v in stock_data.items() for i in range(v)]
        
        value,counts = np.unique(labels, return_counts=True)
        probs = counts / total
        n_classes = np.count_nonzero(probs)

        ent = 0
       
        # Compute entropy
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)
        
        dict_ent[stock] = ent
    return dict_ent
# Return the JSON data for the Plotly graph
def gm(dict_ent):
  
    dict_ent = dict(sorted(dict_ent.items(), key=lambda item: item[1]))
    stocks = list(dict_ent.keys())
    ent_values = list(dict_ent.values())
    max = (np.array(ent_values).max())
    min = (np.array(ent_values).min())
    range = max - min
    margin = range * 0.05
    max = max + margin
    min = min - margin
    
    bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    labels = ['Most negative', 'Slightly negative', 'Neutral', 'Slightly positive', 'Most positive']
    colors = {
          'Most negative': 'red',
          'Slightly negative': 'orange',
          'Neutral': 'yellow',
          'Slightly positive': 'lightgreen',
          'Most positive': 'darkgreen'}
    # Build dataframe
    df = pd.DataFrame({'y': np.array(ent_values),
                   'x': stocks,
                   'label': pd.cut(np.array(ent_values), bins=bins, labels=labels)})

    bars = []
    for label, label_df in df.groupby('label'):
        bars.append(go.Bar(x=label_df.x,
                        y=label_df.y,
                        name=label,
                        marker={'color': colors[label]}))

    fig = go.FigureWidget(data=bars) 


    return fig

def fig_layout(fig, ytitle, ytickfromat, xtitle,ticker, legendtitle, type_of_plot, yaxis_tickprefix=None):
    fig.update_layout(
        yaxis={
            "title": ytitle,
            "tickformat": ytickfromat,

        },
        yaxis_tickprefix = yaxis_tickprefix,
        paper_bgcolor= 'rgba(0,0,0,0)',
        plot_bgcolor= 'rgba(0,0,0,0)',
        # autosize=True,
        legend=dict(
            title=legendtitle,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='white'
        ),
        title={
            'text': '{} - {} <br>'.format(type_of_plot,ticker),
            'y': 0.85,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',},
        titlefont=dict(
            size=20,
            color="black"),

        template="simple_white",
        xaxis=dict(
            title=xtitle,
            showticklabels=True),
        showlegend=True,
        font=dict(
            # family="Courier New, monospace",
            size=12,
            color="black"
        ),
        
    )
    return fig
