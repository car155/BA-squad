from dash import Dash, dcc, html, Input, Output
import plotly.express as px

def register_callbacks(dashapp):
    @dashapp.callback(Output("time-series-chart", "figure"), Input("ticker", "value"))
    def display_time_series(ticker):
        df = px.data.stocks() # replace with your own data source
        fig = px.line(df, x='date', y=ticker)
        return fig

        

