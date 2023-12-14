from functions import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')


opp_stats = pd.read_csv('DATA_SPORT/Opponent Stats Per 100 Poss.csv')
team_stats = pd.read_csv('DATA_SPORT/Team Stats Per 100 Poss.csv')
game = pd.read_csv('DATA/GAME.csv')
odds = pd.read_csv('DATA/ODDS.csv')

useful_cols = ['season','abbreviation','fg_per_100_poss','x3p_per_100_poss','x2p_per_100_poss','ft_per_100_poss','orb_per_100_poss','drb_per_100_poss','ast_per_100_poss','stl_per_100_poss','blk_per_100_poss','tov_per_100_poss','pf_per_100_poss','pts_per_100_poss']
team_stats = team_stats[useful_cols]
useful_cols = ['season', 'abbreviation', 'opp_fg_per_100_poss', 'opp_x3p_per_100_poss', 'opp_x2p_per_100_poss', 'opp_ft_per_100_poss', 'opp_orb_per_100_poss', 'opp_drb_per_100_poss', 'opp_ast_per_100_poss', 'opp_stl_per_100_poss', 'opp_blk_per_100_poss', 'opp_tov_per_100_poss', 'opp_pf_per_100_poss', 'opp_pts_per_100_poss']
opp_stats = opp_stats[useful_cols]
stats = pd.merge(team_stats,opp_stats,on=['season','abbreviation'])

useful_cols = ['GAME_ID','HOMETEAM','AWAYTEAM','WL_HOME','PLUS_MINUS_HOME','HOMEML','HOMESPREAD_ATOPEN']
odds = odds[useful_cols]
odds = odds[odds['GAME_ID']!=0]
odds = odds[odds['HOMEML'].notna()]
game_dates = game[['GAME_ID','GAME_DATE_HOME']].copy()
game_dates['YEAR'] = pd.to_datetime(game_dates['GAME_DATE_HOME']).dt.year
game_dates = game_dates.drop('GAME_DATE_HOME',axis=1)
odds = pd.merge(odds,game_dates,on='GAME_ID',how='left')
odds['YEAR-1'] = odds['YEAR'] - 1
odds = pd.merge(odds, stats, left_on=['HOMETEAM', 'YEAR-1'], right_on=['abbreviation', 'season'], suffixes=('', '_Home'))
odds = pd.merge(odds, stats, left_on=['AWAYTEAM', 'YEAR-1'], right_on=['abbreviation', 'season'], suffixes=('', '_Away'))
drop_cols = ['YEAR-1','season','abbreviation','season_Away','abbreviation_Away','YEAR']
odds = odds.drop(drop_cols,axis=1)

useful_cols = ['GAME_ID','TEAM_ABBREVIATION_HOME','TEAM_ABBREVIATION_AWAY','WL_HOME','PLUS_MINUS_HOME','GAME_DATE_HOME']
game = game[useful_cols]
game['YEAR'] = pd.to_datetime(game['GAME_DATE_HOME']).dt.year
game['YEAR-1'] = game['YEAR'] - 1
game = pd.merge(game, stats, left_on=['TEAM_ABBREVIATION_HOME', 'YEAR-1'], right_on=['abbreviation', 'season'], suffixes=('', '_Home'))
game = pd.merge(game, stats, left_on=['TEAM_ABBREVIATION_AWAY', 'YEAR-1'], right_on=['abbreviation', 'season'], suffixes=('', '_Away'))
drop_cols = ['YEAR-1','season','abbreviation','season_Away','abbreviation_Away','YEAR','GAME_DATE_HOME']
game = game[game['x3p_per_100_poss'].notna()].copy()
game = game.drop(drop_cols,axis=1)

X_train = game.loc[:, 'fg_per_100_poss':]
y_train_reg = game['PLUS_MINUS_HOME']
y_train_cla = game['WL_HOME'].map({'W':1,'L':0})
X_test = odds.loc[:, 'fg_per_100_poss':]
y_test_reg = odds['PLUS_MINUS_HOME']
y_test_cla = odds['WL_HOME'].map({'W':1,'L':0})

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = Sequential([
	Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
	Dense(32, activation='relu'),
	Dense(1)  # Single output node for regression
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train_reg, epochs=2, batch_size=4, verbose=1)
loss = model.evaluate(X_test, y_test_reg, verbose=0)
print(f"\nTest Loss: {loss}")

predictions = model.predict(X_test,verbose=0)[:,0]
bin_truth = np.array(y_test_reg) > 0
bin_pred = predictions > 0
count = 0 
for i in range(len(bin_truth)):
	if bin_truth[i] == bin_pred[i]:
		count +=1
print("Accuracy: {:<6.2f}%\n".format(count/len(predictions)*100))