#%%
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import dash as dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly as ply
import plotly.express as px

import datetime

print("\nIMPORT SUCCESS")

#%%
# LIVE MATCHUPS
matchup_filepath = '../cloud-computing-finale/data/dash_matchups-4-7-22-vF'
matchup_df = pd.read_excel(matchup_filepath + '.xlsx', index_col='TEAM_CODE')
#matchup_df = pd.read_csv(matchup_filepath + '.csv', index_col='TEAM_CODE')

#print(matchup_df.columns)
print(matchup_df)

#%%
print(matchup_df['TEAM'])

preds_dict = {'Magic':'LOSS', 'Hornets':'WIN', 'Celtics':'LOSS', 'Bucks':'WIN', '76ers':'WIN',
              'Raptors':'LOSS', 'Trail Blazers':'LOSS', 'Pelicans':'WIN', 'Spurs':'LOSS',
              'Timberwolves':'WIN', 'Grizzlies':'LOSS', 'Nuggets':'WIN', 'LA Lakers':'LOSS', 'Warriors':'WIN'}

matchup_df['MODEL_PREDICTION'] = matchup_df.TEAM.map(preds_dict)
print(matchup_df)

#%%
# STATS LIBRARY
#tr_filepath = '../cloud-computing-finale/data/tr_data_hub_4-13-22'
#tr_data_hub = pd.read_excel(tr_filepath + '.xlsx', index_col='TEAM_CODE')
#tr_data_hub = pd.read_csv(tr_filepath + '.csv', index_col='TEAM_CODE')
#tr_data_hub.info()


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
matchup_app = dash.Dash('MATCHUP MACHINE', external_stylesheets=external_stylesheets)
application = matchup_app.server

# {current_time:%Y-%m-%d %H:%M}
#'backgroundColor': 'rgb(220, 220, 220)',

matchup_app.layout = html.Div([html.H1('NBA MATCHUP MACHINE', style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}),
                     #html.Br(),
                               html.Div([html.H3((f'MATCHUP INFORMATION - 4/7/2022'),
                                                 style={'textAlign': 'center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}), #'dodgerblue'

                               dash_table.DataTable(matchup_df.to_dict('records'), [{"name": i, "id": i} for i in matchup_df.columns],
                                                    style_data={'textAlign': 'Center', 'fontWeight': 'bold', 'border': '2px solid black'},
                                                    style_cell={'textAlign': 'Center', 'fontWeight': 'bold', 'padding': '5px'},   #324f6e - TBD  #B10DC9 - fuschia #7FDBFF - Aqua
                                                    style_header={'backgroundColor': '#7FDBFF', 'color': 'black', 'fontWeight': 'bold', 'border': '2px solid black'}, #'1px solid blue'
                                                    style_data_conditional = [{'if': {'row_index': [2,3,6,7,10,11]}, 'backgroundColor': 'rgb(211, 211, 211)'},

                                                                              {'if': {'filter_query': '{FINAL} > 105', 'column_id': 'FINAL', 'row_index': [0,1]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 105', 'column_id': 'FINAL', 'row_index': [0,1]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 123', 'column_id': 'FINAL', 'row_index': [2,3]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 123', 'column_id': 'FINAL', 'row_index': [2,3]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 116', 'column_id': 'FINAL', 'row_index': [4,5]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 116', 'column_id': 'FINAL', 'row_index': [4,5]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 100', 'column_id': 'FINAL', 'row_index': [6,7]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 100', 'column_id': 'FINAL', 'row_index': [6,7]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 124', 'column_id': 'FINAL', 'row_index': [8,9]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 124', 'column_id': 'FINAL', 'row_index': [8,9]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 112', 'column_id': 'FINAL', 'row_index': [10,11]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 112', 'column_id': 'FINAL', 'row_index': [10,11]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{FINAL} > 115', 'column_id': 'FINAL', 'row_index': [12,13]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red',
                                                                              {'if': {'filter_query': '{FINAL} < 115', 'column_id': 'FINAL', 'row_index': [12,13]},
                                                                               'color': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{MODEL_PREDICTION} = "LOSS"','column_id': 'MODEL_PREDICTION'},
                                                                               'color': 'black', 'backgroundColor': '#FF4136', 'fontWeight': 'bold'},

                                                                              {'if': {'filter_query': '{MODEL_PREDICTION} = "WIN"','column_id': 'MODEL_PREDICTION'},
                                                                               'color': 'black', 'backgroundColor': '#2ECC40', 'fontWeight': 'bold'},

                                                                              {'if': {'column_id': 'MODEL_PREDICTION'}, 'fontWeight': 'bold',},], #'#01FF70'  'backgroundColor': '#01FF70',
                                                    ),
                               ]),
                               html.Br(),
                               #html.P('METRIC COMPARISON'),
                               dcc.Graph(id='matchup-chart'),
                               html.Br(),
                               html.P('STAT A'),
                               dcc.Dropdown(id='stata',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     {'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     {'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     {'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     {'label': 'TREB%', 'value': 'TREB%'},
                                                     {'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     {'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},], value='WIN%'),
                               html.Br(),
                               html.P('STAT B'),
                               dcc.Dropdown(id='statb',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     {'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     {'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     {'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     {'label': 'TREB%', 'value': 'TREB%'},
                                                     {'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     {'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},], value='WIN%'),
                               html.Br(),

                               html.Div(id='layout')])

@matchup_app.callback(Output(component_id='layout', component_property='children'),
                      #Output(component_id='matchup-df', component_property='figure'),
                      Output(component_id='matchup-chart', component_property='figure'),
                      [Input(component_id='stata', component_property='value'),
                       Input(component_id='statb', component_property='value')])

def display_chart(stata, statb):
    fig = px.scatter(matchup_df, x=[stata], y=[statb])
    return fig

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)


#%%