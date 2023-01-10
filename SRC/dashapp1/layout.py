from dash import dcc
from dash import html
from flask import session, app
import pandas as pd
import numpy as np
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input,Output
from dash import callback_context

def register_layout(dashapp):
    def get_options(list_stocks):
        dict_list = []
        for i in list_stocks:
            dict_list.append({'label': i, 'value': i})

        return dict_list


    layout = html.Div([
    html.H4('Stock price analysis'),
    dcc.Graph(id="time-series-chart"),
    html.P("Select stock:"),
    dcc.Dropdown(
        id="ticker",
        options=["AMZN", "FB", "NFLX"],
        value="AMZN",
        clearable=False,
    ),
    ])
  
    dashapp.layout = layout