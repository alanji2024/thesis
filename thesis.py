from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print()
##################################

# # expand sqlite of odds data into csv files (default names)
# expand_sqlite_to_csv("DATA_ODDS/basketball-final.sqlite","DATA_ODDS/")

# # re-save odds csv, converting to decimal odds
# save_decimal_odds("DATA_ODDS/BettingOdds_History.csv","DATA_ODDS/BettingOdds_History.csv")

# # combine all game data (including playoffs)
# combine_playoffs("DATA_ODDS/Game_Inactive_Players.csv","DATA_ODDS/Game_Playoffs_Inactive_Players.csv","DATA/GAME_INACTIVE_PLAYERS.csv")
# # sort data by game date
# sort_date("DATA/GAME_FULL.csv","DATA/GAME_FULL.csv")

# # merge odds with W/L data
# odds_ml_wl("DATA_ODDS/BettingOdds_History.csv","DATA/GAME.csv","DATA/ML.csv")

# ##################################

# train("DATA/GAME.csv","DATA/model.keras")
# # add desired wager portfolio
save_wagers("DATA/ODDS.csv","DATA/WAGERS.csv",method=None)
# # calculate profits
calc_results("DATA/WAGERS.csv","DATA/RESULTS.csv")


# predict("DATA/ML.csv","DATA/GAME.csv")