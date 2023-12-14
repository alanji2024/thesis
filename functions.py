import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sqlite3
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

### data preprocessing
def expand_sqlite_to_csv(path_in, path_out):
  '''Expands a .sqlite file to multiple .csv files representing each table.
  Each dataframe is given column names in all caps for consistency.

  path_in (str): filepath of sqlite
  path_out (str): directory to save files to

  Returns:
  None
  '''

  data_connection = sqlite3.connect(path_in)
  data_cursor = data_connection.cursor()
  data_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
  data_table_names = data_cursor.fetchall()

  for name_tuple in data_table_names:
    name = name_tuple[0]
    df = pd.read_sql_query("SELECT * from " + name, data_connection)
    df.columns = df.columns.str.upper()
    df.to_csv(path_out + name + ".csv",index=False)

def save_decimal_odds(path_in, path_out, home_name='HOMEML', away_name='AWAYML', none = 'NL'):
  '''
  Converts American odds to decimal odds (columns must be named "Home(Away)ML")

  path_in: path to original American odds data
  path_out: path to save new odds data
  
  Returns:
  None
  '''

  df = pd.read_csv(path_in)
  home_ml_np = df[home_name].to_numpy()
  away_ml_np = df[away_name].to_numpy()
  home_odds_np = np.array([None if x == none else (100/float(x[1:])+1 if x[0] == "-" else (float(x)+100)/100) for x in home_ml_np])
  away_odds_np = np.array([None if x == none else (100/float(x[1:])+1 if x[0] == "-" else (float(x)+100)/100) for x in away_ml_np])
  df[home_name] = home_odds_np
  df[away_name] = away_odds_np
  df.to_csv(path_out,index=False)

def combine_playoffs(games,games_playoff,path_out):
  '''
  Combines data for (non)/playoff games, creating a new column as a marker.
  '''
  games = pd.read_csv(games)
  games_playoff = pd.read_csv(games_playoff)
  games['PLAYOFF'] = 0
  games_playoff['PLAYOFF'] = 1

  pd.concat([games, games_playoff], ignore_index=True).to_csv(path_out, index=False)

def sort_date(path_in,path_out,name='GAME_DATE_AWAY'):
  '''
  Sorts odds by date (based on name of current date column, from "YYYY-MM-DD").
  '''
  df = pd.read_csv(path_in)
  df['DATE'] = pd.to_datetime(df[name])
  df = df.sort_values(by='DATE')
  df.to_csv(path_out, index=False)

def odds_wl(path_in_odds, path_in_games, path_out, wl='WL_HOME',game_id = 'GAME_ID',spread='PLUS_MINUS_HOME'):
  '''
  Combines odds with win/loss results for evaluation.
  '''
  odds_df = pd.read_csv(path_in_odds)
  wl_df = pd.read_csv(path_in_games)[[game_id, wl, spread]]
  odds_df.merge(wl_df, on=game_id, how="left").to_csv(path_out,index=False)

###########################








def save_wagers(odds_path, path_out,method=None):
  odds_df = pd.read_csv(odds_path)
  odds_df["HWAGER"], odds_df["AWAGER"]= wager(odds_path,method)
  odds_df.to_csv(path_out,index=False)

def calc_results(wagers_path, path_out, save=True):
  wagers_df = pd.read_csv(wagers_path)
  output = wagers_df.copy()
  profits = np.array([])
  for i in range(wagers_df.shape[0]):
    profit = 0
    if wagers_df.iloc[i]["WL_HOME"]=="W":
      profit = (wagers_df.iloc[i]["HOMEML"] - 1) * \
          wagers_df.iloc[i]["HWAGER"] - wagers_df.iloc[i]["AWAGER"]
    else:
      profit = (wagers_df.iloc[i]["AWAYML"] - 1) * \
          wagers_df.iloc[i]["AWAGER"] - wagers_df.iloc[i]["HWAGER"]
    profits = np.append(profits, profit)
    profits = np.nan_to_num(profits)
  output["PROFIT"] = profits
  
  wagered = output["HWAGER"].sum() + output["AWAGER"].sum()
  front = "+" if np.sum(profits) > 0 else ""
  print("\t\t\t\t\t${}{:<10.2f} on ${:>10.2f}:\t{:>6.2f}%   profit.".format(front, np.sum(profits), wagered,np.sum(profits) / wagered * 100))
  print("\t\t\t\t\t{:<6} of {:>6} games won:\t{:>6.2f}% accuracy.".format(np.sum(profits > 0),np.sum(profits != 0),np.sum(profits > 0) / np.sum(profits != 0) * 100))
  if save:
    output.to_csv(path_out,index=False)
  return np.sum(profits) / wagered * 100

def wager(odds_path,method):
  ''' Determines wager amounts for each possible game line.

  odds_path: file containing odds information

  home_wagers: array of wagers on home team
  away_wagers: array of wagers on away team
  '''

  input = pd.read_csv(odds_path)
  home_mls = list(input['HOMEML'])
  away_mls = list(input['AWAYML'])

  home_wagers = np.zeros(len(home_mls))
  away_wagers = np.zeros(len(away_mls))
  for i in range(len(home_mls)):
    if home_mls[i] >= away_mls[i]:
      home_wagers[i] = 1 
    else:
      away_wagers[i] = 1 
  return home_wagers, away_wagers

def predict(odds_path,full_data):
  odds = pd.read_csv(odds_path)
  full_data = pd.read_csv(full_data)
  data = pd.DataFrame(odds['GAME_ID'])
  data = data.merge(full_data, on="GAME_ID", how="left")

  drops = ['TEAM_ID_AWAY','TEAM_ABBREVIATION_AWAY','TEAM_NAME_AWAY','GAME_DATE_AWAY','MATCHUP_AWAY','WL_AWAY','TEAM_ID_HOME','TEAM_ABBREVIATION_HOME','TEAM_NAME_HOME','GAME_DATE_HOME','MATCHUP_HOME','DATE']
  data = data.drop(columns=drops, axis=1)
  data = data.dropna()
  print(data.columns)
  target_column = 'WL_HOME'
  X = data.drop(target_column, axis=1)
  model = load_model('DATA/model.keras')
  predictions = model.predict(X)
  print(predictions)

def train(data,path_out):
  data = pd.read_csv(data)
  drops = ['TEAM_ID_AWAY','TEAM_ABBREVIATION_AWAY','TEAM_NAME_AWAY','GAME_DATE_AWAY','MATCHUP_AWAY','WL_AWAY','TEAM_ID_HOME','TEAM_ABBREVIATION_HOME','TEAM_NAME_HOME','GAME_DATE_HOME','MATCHUP_HOME','DATE']
  data = data.drop(columns=drops, axis=1)
  data['WL_HOME'] = data['WL_HOME'].map({'W': 1, 'L': 0})
  data = data.dropna()
  
  target_column = 'WL_HOME'
  X = data.drop(target_column, axis=1)
  y = data[target_column]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = Sequential()
  model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=5, batch_size=3,verbose=1)
  loss, accuracy = model.evaluate(X_test, y_test)
  print("{:<6.2f}".format(accuracy*100))

  model.save(path_out)

  predictions = model.predict(X)
  print(predictions)

def plot_linear(x_data, y_data, title="", xlabel="", ylabel="", oneline=False):
  x = np.array(x_data).reshape((-1, 1))
  y = np.array(y_data)
  model = LinearRegression().fit(x, y)
  a = model.coef_[0]
  b = model.intercept_

  fig, ax = plt.subplots()
  plt.scatter(x_data, y_data)
  # ax.fill_between(y_data, (y_data - ), (y_data + 0.1), color='b', alpha=.1)
  if oneline:
    ax.axline((0,0), slope=1, color="b")
  ax.axline((0, b), slope=a, color="m")
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.show()