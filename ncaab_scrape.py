#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import dash as dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly as ply
import plotly.express as px

import requests
from bs4 import BeautifulSoup
import re

import datetime
import sys
import os

from scipy import stats as stats
import statistics

import json
import time
import nltk

print("\nIMPORT SUCCESS")

#%%
# TEAM STATS (TEAM RANKINGS)

## URL VARIABLES

# filtered stats of interest

title_links = ['points-per-game', 'average-scoring-margin', 'field-goals-made-per-game',
               'field-goals-attempted-per-game', 'offensive-efficiency', 'defensive-efficiency', 'effective-possession-ratio',
               'effective-field-goal-pct', 'true-shooting-percentage', 'three-point-pct', 'two-point-pct',
               'free-throw-pct', 'three-pointers-made-per-game', 'three-pointers-attempted-per-game',
               'offensive-rebounds-per-game', 'defensive-rebounds-per-game', 'total-rebounds-per-game',
               'offensive-rebounding-pct', 'defensive-rebounding-pct', 'total-rebounding-percentage',
               'blocks-per-game', 'steals-per-game', 'assists-per-game', 'turnovers-per-game',
               'assist--per--turnover-ratio', 'win-pct-all-games', 'win-pct-close-games', 'possessions-per-game',
               'personal-fouls-per-game',
               'opponent-points-per-game', 'opponent-average-scoring-margin', 'opponent-shooting-pct',
               'opponent-effective-field-goal-pct', 'opponent-true-shooting-percentage',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'opponent-assists-per-game', 'opponent-turnovers-per-game', 'opponent-assist--per--turnover-ratio',
               'opponent-offensive-rebounds-per-game', 'opponent-defensive-rebounds-per-game',
               'opponent-total-rebounds-per-game', 'opponent-offensive-rebounding-pct', 'opponent-defensive-rebounding-pct',
               'opponent-blocks-per-game', 'opponent-steals-per-game', 'opponent-effective-possession-ratio',
               ]

team_links = ['points-per-game', 'average-scoring-margin',
              'offensive-efficiency', 'percent-of-points-from-2-pointers',
              'percent-of-points-from-3-pointers', 'percent-of-points-from-free-throws',
              'shooting-pct', 'effective-field-goal-pct', 'true-shooting-percentage',
              'three-point-pct', 'two-point-pct', 'free-throw-pct',
              'field-goals-made-per-game', 'field-goals-attempted-per-game',
              'three-pointers-made-per-game', 'three-pointers-attempted-per-game',
              'free-throws-made-per-game', 'free-throws-attempted-per-game',
              'three-point-rate', 'fta-per-fga', 'ftm-per-100-possessions',
              'offensive-rebounds-per-game', 'defensive-rebounds-per-game',
              'total-rebounds-per-game',
              'offensive-rebounding-pct', 'defensive-rebounding-pct',
              'total-rebounding-percentage', 'blocks-per-game',
              'steals-per-game', 'assists-per-game',
              'turnovers-per-game', 'assist--per--turnover-ratio',
              'assists-per-fgm', 'games-played',
              'possessions-per-game', 'extra-chances-per-game',
              'effective-possession-ratio',
              'win-pct-all-games', 'win-pct-close-games', ]

opponent_links = ['personal-fouls-per-game'
                  'opponent-points-per-game', 'opponent-average-scoring-margin',
                  'defensive-efficiency', 'opponent-points-from-2-pointers',
                  'opponent-points-from-3-pointers', 'opponent-percent-of-points-from-2-pointers',
                  'opponent-percent-of-points-from-3-pointers', 'opponent-percent-of-points-from-free-throws',
                  'opponent-shooting-pct', 'opponent-effective-field-goal-pct',
                  'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
                  'opponent-true-shooting-percentage',
                  'opponent-field-goals-made-per-game', 'opponent-field-goals-attempted-per-game',
                  'opponent-three-pointers-made-per-game',
                  'opponent-three-pointers-attempted-per-game', 'opponent-free-throws-made-per-game',
                  'opponent-free-throws-attempted-per-game',
                  'opponent-three-point-rate', 'opponent-two-point-rate', 'opponent-fta-per-fga',
                  'opponent-ftm-per-100-possessions',
                  'opponent-free-throw-rate', 'opponent-non-blocked-2-pt-pct',
                  'opponent-offensive-rebounds-per-game', 'opponent-defensive-rebounds-per-game',
                  'opponent-team-rebounds-per-game', 'opponent-total-rebounds-per-game',
                  'opponent-offensive-rebounding-pct', 'opponent-defensive-rebounding-pct',
                  'opponent-blocks-per-game', 'opponent-steals-per-game', 'opponent-block-pct',
                  'opponent-steals-perpossession',
                  'opponent-steal-pct', 'opponent-assists-per-game', 'opponent-turnovers-per-game',
                  'opponent-assist--per--turnover-ratio',
                  'opponent-assists-per-fgm', 'opponent-assists-per-possession', 'opponent-turnovers-per-possession',
                  'opponent-turnover-pct', 'opponent-personal-fouls-per-game',
                  'opponent-personal-fouls-per-possession', 'opponent-personal-foul-pct',
                  'opponent-effective-possession-ratio',
                  'opponent-win-pct-all-games', 'opponent-win-pct-close-games']

#%%
## TEAMRANKINGS.COM - DATA SCRAPE

tr_url = 'https://www.teamrankings.com/ncaa-basketball/stat/'
base_url = 'https://www.teamrankings.com/'

tr_cols = ['Rank', 'Team', '2021', 'Last 3', 'Last 1', 'Home', 'Away', '2020']  # , 'Stat'
tr_link_dict = {link: pd.DataFrame() for link in title_links}  # columns=tr_cols
df = pd.DataFrame()

for link in title_links:
    stat_page = requests.get(tr_url + link)
    soup = BeautifulSoup(stat_page.text, 'html.parser')
    table = soup.find_all('table')[0]
    cols = [each.text for each in table.find_all('th')]
    rows = table.find_all('tr')
    for row in rows:
        data = [each.text for each in row.find_all('td')]
        temp_df = pd.DataFrame([data])
        # df = df.append(temp_df, sort=True).reset_index(drop=True)
        tr_link_dict[link] = tr_link_dict[link].append(temp_df, sort=True).reset_index(drop=True)
        tr_link_dict[link] = tr_link_dict[link].dropna()

    tr_link_dict[link].columns = cols
    tr_link_dict[link][link] = tr_link_dict[link]['2021']
    tr_link_dict[link].index = tr_link_dict[link]['Team']
    tr_link_dict[link].drop(columns=['Rank', 'Last 3', 'Last 1', 'Home', 'Away', '2020', '2021', 'Team'], inplace=True)

print(tr_link_dict.keys())


#%%
tr_df = pd.DataFrame()

for stat in tr_link_dict:
    # tr_link_dict[stat].replace({'%',''}, regex=True)#.strip('%')
    tr_df[stat] = tr_link_dict[stat]
    # tr_link_dict[stat] = float(tr_link_dict[stat].replace('%',''))

objects = tr_df.select_dtypes(['object'])
tr_df[objects.columns] = objects.apply(lambda x: x.str.strip('%'))

for stat in tr_df:
    tr_df[stat] = pd.to_numeric(tr_df[stat])

# for col in tr_df:
# tr_df[col] = tr_df[col].astype(float)
# pd.to_numeric(df['DataFrame Column'],errors='coerce')

tr_df.head()

# tr_df[stat] = tr_df[stat].replace('%','') #, regex=True
# pd.DataFrame.from_dict(tr_link_dict.keys())
# tr# tr_df[stat] = tr_df[stat].replace('%','') #, regex=True
# print(tr_link_dict['two-point-pct'])

print(tr_df.describe())

#%%
## RAW DATA EXPORT

tr_filepath_raw = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_ncaab_data_4-05-22-raw'

tr_df.to_excel(tr_filepath_raw + '.xlsx', index=True)
tr_df.to_csv(tr_filepath_raw + '.csv', index=True)

print("\nEXPORT SUCCESS")

#%%
## RAW DATA IMPORT

tr_filepath_raw = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_ncaab_data_4-05-22-raw'

tr_df = pd.read_excel(tr_filepath_raw + '.xlsx', index_col='Team')
# tr_df = pd.read_csv(tr_filepath_raw + '.csv', index_col='Team')

print("\nIMPORT SUCCESS")

#%%
## FEATURE ENGINEERING

tr_df.info()

#%%
# SCORING MARGIN / POSSESSIONS
tr_df['net-avg-scoring-margin'] = tr_df['average-scoring-margin'] - tr_df['opponent-average-scoring-margin']
tr_df['net-points-per-game'] = tr_df['points-per-game'] - tr_df['opponent-points-per-game']
#tr_df['net-effective-possession-ratio'] = tr_df['effective-possession-ratio'] - tr_df['opponent-effective-possession-ratio']
tr_df['net-adj-efficiency'] = tr_df['offensive-efficiency'] - tr_df['defensive-efficiency']

# NET SHOOTING PERCENTAGES
tr_df['net-effective-field-goal-pct'] = tr_df['effective-field-goal-pct'] - tr_df['opponent-effective-field-goal-pct']
tr_df['net-true-shooting-percentage'] = tr_df['true-shooting-percentage'] - tr_df['opponent-true-shooting-percentage']

# STOCKS = STEALS + BLOCKS
tr_df['stocks-per-game'] = tr_df['steals-per-game'] + tr_df['blocks-per-game']
#tr_df['opponent-stocks-per-game'] = tr_df['opponent-steals-per-game'] + tr_df['opponent-blocks-per-game']
#tr_df['net-stocks-per-game'] = tr_df['stocks-per-game'] - tr_df['opponent-stocks-per-game']

# AST/TO = TURNOVERS / ASSISTS
tr_df['total-turnovers-per-game'] = tr_df['turnovers-per-game'] + tr_df['opponent-turnovers-per-game']
tr_df['net-assist--per--turnover-ratio'] = tr_df['assist--per--turnover-ratio'] - tr_df[
    'opponent-assist--per--turnover-ratio']

# REBOUNDS
tr_df['net-total-rebounds-per-game'] = tr_df['total-rebounds-per-game'] - tr_df['opponent-total-rebounds-per-game']
tr_df['net-off-rebound-pct'] = tr_df['offensive-rebounding-pct'] - tr_df['opponent-offensive-rebounding-pct']
tr_df['net-def-rebound-pct'] = tr_df['defensive-rebounding-pct'] - tr_df['opponent-defensive-rebounding-pct']

# ALTERNATE CALC - yields different performance than above
tr_df['net-off-rebound-pct'] = tr_df['offensive-rebounding-pct'] - tr_df['opponent-defensive-rebounding-pct']
tr_df['net-def-rebound-pct'] = tr_df['defensive-rebounding-pct'] - tr_df['opponent-offensive-rebounding-pct']

tr_df.info()
tr_df.columns


#%%
## FINAL DATA EXPORT

tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'

tr_df.to_excel(tr_filepath + '.xlsx', index=True)
tr_df.to_csv(tr_filepath + '.csv', index=True)

print("\nEXPORT SUCCESS")


#%%
# CLEAN DATA IMPORT
tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
kp_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/kenpom_pull_3-14-22'
tr_df = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'
#tr_df = pd.read_csv(mtr_filepath + '.csv')
kp_df = pd.read_excel(kp_filepath + '.xlsx') #index_col='Team'
#kp_df = pd.read_csv(kp_filepath + '.csv')

print(tr_df.head())

#%%
print(tr_df.info())
print(kp_df.info())
#print(tr_df.index)
#print(tr_df)

#%%
print(tr_df.columns)

#%%
# RENAME COLUMNS TO IMPROVE APP OPTICS
cols_app = {'Team': 'TEAM', 'points-per-game':'PTS/GM', 'average-scoring-margin':'AVG_MARGIN', 'win-pct-all-games':'WIN%', 'win-pct-close-games':'WIN%_CLOSE',
            'effective-field-goal-pct':'EFG%', 'true-shooting-percentage':'TS%', 'effective-possession-ratio': 'POSS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'field-goals-made-per-game':'FGM/GM', 'field-goals-attempted-per-game':'FGA/GM', 'three-pointers-made-per-game':'3PM/GM', 'three-pointers-attempted-per-game':'3PA/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF',
            'total-rebounds-per-game':'TRB/GM', 'offensive-rebounds-per-game':'ORB/GM', 'defensive-rebounds-per-game':'DRB/GM',
            'offensive-rebounding-pct':'ORB%', 'defensive-rebounding-pct':'DRB%', 'total-rebounding-percentage':'TRB%',
            'blocks-per-game':'B/GM', 'steals-per-game':'S/GM', 'assists-per-game':'AST/GM', 'turnovers-per-game':'TO/GM',
            'assist--per--turnover-ratio':'AST/TO', 'possessions-per-game':'POSS/GM', 'personal-fouls-per-game':'PF/GM',
            'opponent-points-per-game':'OPP_PTS/GM', 'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'opponent-effective-field-goal-pct':'OPP_EFG%', 'opponent-true-shooting-percentage':'OPP_TS%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%', 'opponent-shooting-pct':'OPP_FG%',
            'opponent-assists-per-game':'OPP_AST/GM', 'opponent-turnovers-per-game':'OPP_TO/GM', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'opponent-offensive-rebounds-per-game':'OPP_OREB/GM', 'opponent-defensive-rebounds-per-game':'OPP_DREB/GM', 'opponent-total-rebounds-per-game':'OPP_TREB/GM',
            'opponent-offensive-rebounding-pct':'OPP_OREB%', 'opponent-defensive-rebounding-pct':'OPP_DREB%',
            'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM',
            'opponent-effective-possession-ratio':'OPP_POSS%',
            'net-avg-scoring-margin':'NET_AVG_MARGIN', 'net-points-per-game':'NET_PTS/GM',
            'net-adj-efficiency':'NET_ADJ_EFF',
            'net-effective-field-goal-pct':'NET_EFG%', 'net-true-shooting-percentage':'NET_TS%',
            'stocks-per-game':'STOCKS/GM', 'total-turnovers-per-game':'TTL_TO/GM',
            'net-assist--per--turnover-ratio':'NET_AST/TO',
            'net-total-rebounds-per-game':'NET_TREB/GM', 'net-off-rebound-pct':'NET_OREB%', 'net-def-rebound-pct':'NET_DREB%'
            }

#%%
#tr_df['VISITOR_CODE'] = matchup_history['VISITOR'].map(team_code_dict)
tr_df.columns = tr_df.columns.map(cols_app)
tr_df.info()


#%%
print(tr_df.columns)
print(tr_df.info())


#%%


#%%

## KENPOM

# KENPOM

## DATA SCRAPE

# KENPOM DATA SCRAPE

# Base url, and a lambda func to return url for a given year
base_url = 'http://kenpom.com/index.php'
url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2021 else base_url)

# Years on kenpom's site; scrape and set as list to be more dynamic?
years = range(2021, 2022)


# Create a method that parses a given year and spits out a raw dataframe
def import_raw_year(year):
    """
    Imports raw data from a kenpom year into a dataframe
    """
    f = requests.get(url_year(year))
    soup = BeautifulSoup(f.text, "lxml")
    table_html = soup.find_all('table', {'id': 'ratings-table'})

    # Weird issue w/ <thead> in the html
    # Prevents us from just using pd.read_html
    # Find all <thead> contents and replace/remove them
    # This allows us to easily put the table row data into a dataframe using pandas
    thead = table_html[0].find_all('thead')

    table = table_html[0]
    for x in thead:
        table = str(table).replace(str(x), '')

    kp_df = pd.read_html(table)[0]
    kp_df['year'] = year
    return kp_df

# Import all the years into a singular dataframe
kp_df = None
for x in years:
    kp_df = pd.concat((kp_df, import_raw_year(x)), axis=0) if kp_df is not None else import_raw_year(2022)

# Column rename based off of original website
kp_df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'Adj EM',
                 'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank',
                 'AdjT', 'AdjT Rank', 'Luck', 'Luck Rank',
                 'SOS Adj EM', 'SOS Adj EM Rank', 'SOS OppO', 'SOS OppO Rank',
                 'SOS OppD', 'SOS OppD Rank', 'NCSOS Adj EM', 'NCSOS Adj EM Rank', 'Year']

# Lambda that returns true if given string is a number and a valid seed number (1-16)
valid_seed = lambda x: True if str(x).replace(' ', '').isdigit() \
                               and int(x) > 0 and int(x) <= 16 else False

# Use lambda to parse out seed/team
kp_df['Seed'] = kp_df['Team'].apply(lambda x: x[-2:].replace(' ', '') \
    if valid_seed(x[-2:]) else np.nan)

kp_df['Team'] = kp_df['Team'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)

# Split W-L column into Win / Loss
kp_df['Win'] = kp_df['W-L'].apply(lambda x: int(re.sub('-.*', '', x)))
kp_df['Loss'] = kp_df['W-L'].apply(lambda x: int(re.sub('.*-', '', x)))
kp_df.drop('W-L', inplace=True, axis=1)

# Reorder columns
kp_df = kp_df[['Year', 'Rank', 'Team', 'Conference', 'Win', 'Loss', 'Seed', 'Adj EM',
               'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank',
               'AdjT', 'AdjT Rank', 'Luck', 'Luck Rank',
               'SOS Adj EM', 'SOS Adj EM Rank', 'SOS OppO', 'SOS OppO Rank',
               'SOS OppD', 'SOS OppD Rank', 'NCSOS Adj EM', 'NCSOS Adj EM Rank']]

kp_df.info()

#%%
## DATA EXPORT

kp_df.to_csv('drive/My Drive/SPORTS/kenpom_pull_3-14-22.csv', index=False)
kp_df.to_excel('drive/My Drive/SPORTS/kenpom_pull_3-14-22.xlsx', index=False)

#%%
# Derive the id from the google drive shareable link.

##For the file at hand the link is as below
# URL = 'https://drive.google.com/file/d/1m0mAGzpeMR0W-BDL5BtKrs0HOZsPIAbX/view?usp=sharing'
# path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
# df = pd.read_pickle(path)
# df = pd.read_csv(path)