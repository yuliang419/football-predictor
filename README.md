# football-predictor

## Overview
This project uses a neural network to predict the outcomes of 
English Premier League football matches based on each team's 
recent performance.

## Dataset preparation
The raw data were downloaded from http://www.football-data.co.uk.
All match results from the 2011-12 to 2021-22 seasons are
saved in `data/EPL_results_all.csv`. Premier League standings 
from 2010-2021 were downloaded from 
[Kaggle](https://www.kaggle.com/quadeer15sh/premier-league-standings-11-seasons-20102021)
and saved in `data/EPL_Standings.csv`.

To extract useful features from the raw data and rewrite them
in a usable format, run `dataset.py` or see the beginning
of `dataset_visualize.ipynb`. For each match, the script
retrieves the last 5 home and away games each team has played
and averages certain statistics to use as features. The 
results are then saved in `data/EPL_processed_results.csv`.
Here are the available columns to use as features in training:

```commandline
'StandingDiff': difference between the home and away teams' standings from last season
'HomeWins': home team's percentage of wins in last 5 home games
'AwayWins': away team's percentage of wins in last 5 away games
'HomeDraws': home team's percentage of draws in last 5 home games
'AwayDraws': away team's percentage of draws in last 5 away games
'AvgHomeGoals': home team's average # of goals scored in the last 5 home matches
'AvgAwayGoals': away team's average # of goals scored in the last 5 away matches
'AvgHomeShots': home team's average # of shots in the last 5 home matches
'AvgAwayShots': away team's average # of shots in the last 5 away matches 
'AvgHomeShotsOnTarget': home team's average # of shots on target in the last 5 home matches 
'AvgAwayShotsOnTarget': away team's average # of shots on target in the last 5 away matches
'AvgHomeCorners': home team's average # of corners in the last 5 home matches 
'AvgAwayCorners': away team's average # of corners in the last 5 away matches 
'AvgHomeGoalsConceded': home team's average # of goals conceded in the last 5 home matches 
'AvgAwayGoalsConceded': away team's average # of goals conceded in the last 5 away matches 
'AvgHomeShotsConceded': home team's average # of shots on target in the last 5 home matches 
'AvgAwayShotsConceded': away team's average # of shots on target in the last 5 away matches 
```
Among the remaining columns, `HomeGoals` and `AwayGoals` show
the number of goals scored by the home and away teams in 
the given match, and `FTR` is the final result (home win `H`,
draw `D` or away win `A`).