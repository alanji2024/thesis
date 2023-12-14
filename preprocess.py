from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# # process and expand sqlite data file
# expand_sqlite_to_csv('basketball-final/basketball-final.sqlite','basketball-final/')

# # combine multiple datasets to include playoffs
# combine_playoffs('basketball-final/Game.csv','basketball-final/Game_Playoffs.csv','DATA/GAME.csv')
# combine_playoffs('basketball-final/Game_FullDetails.csv','basketball-final/Game_Playoffs_FullDetails.csv','DATA/GAME_FULL_DETAILS.csv')
# combine_playoffs('basketball-final/Game_Inactive_Players.csv','basketball-final/Game_Playoffs_Inactive_Players.csv','DATA/GAME_INACTIVE_PLAYERS.csv')
# combine_playoffs('basketball-final/Game_Officials.csv','basketball-final/Game_Playoffs_Officials.csv','DATA/GAME_OFFICIALS.csv')

# # convert from american to decimal odds
# save_decimal_odds('basketball-final/BettingOdds_History.csv','DATA/ODDS.csv')
# # combine odds data with w/l results
# odds_wl('DATA/ODDS.csv','DATA/GAME.csv','DATA/ODDS.csv')

