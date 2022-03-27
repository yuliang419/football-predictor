import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union

standings = pd.read_csv('data/EPL_Standings.csv')
all_matches = pd.read_csv('data/EPL_results_all.csv')
all_matches['Date'] = pd.to_datetime(all_matches['Date'], dayfirst=True)
label_map = {'H': [1, 0, 0], 'D': [0, 1, 0], 'A': [0, 0, 1]}


def aggregate_matches(home: bool, team: str, date: pd.Timestamp, matches: int = 5) -> pd.Series:
    """
    Aggregate stats from the last few home or away matches for a given team.
    :param home: If True, aggregate data from home matches. Otherwise aggregate away matches.
    :param team: name of team
    :param date: aggregate last few matches before this date
    :param matches: number of matches to aggregate
    :return: stats averaged over the last few matches
    """
    if home:
        team_label = 'HomeTeam'
        res_label = 'H'
    else:
        team_label = 'AwayTeam'
        res_label = 'A'

    last_matches = all_matches.loc[(all_matches[team_label] == team) & (all_matches['Date'] < date)][-matches:]

    # If not enough data available, return empty series
    if len(last_matches) < matches:
        return pd.Series(index=['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D',
                                'B365A', 'WinPercentage', 'DrawPercentage'], dtype='float64')

    results = last_matches.FTR.value_counts()
    win_percentage = results[res_label] / matches * 100 if res_label in results.index else 0
    draw_percentage = results['D'] / matches * 100 if 'D' in results.index else 0

    aggregated_results = last_matches.mean(numeric_only=True)
    aggregated_results['WinPercentage'] = win_percentage
    aggregated_results['DrawPercentage'] = draw_percentage

    return aggregated_results


def get_team_stats(team: str, date: pd.Timestamp, season: str, matches: int = 5) -> dict:
    """
    Retrieve stats for a given team on a given date and organize into dict.
    :param team: name of team
    :param date: retrieve stats before this date
    :param season: current season (should match the date parameter), e.g. 2020-21
    :param matches: number of matches to aggregate
    :return:
    """
    last_season = '%s-%s' % (int(season[:4]) - 1, int(season[5:7]) - 1)
    try:
        last_season_standing = standings.loc[(standings['Team'] == team) & (standings['Season'] == last_season),
                                             'Pos'].iloc[0]
    except IndexError:
        # Assign standing 18 to all newly promoted teams
        last_season_standing = 18

    last_home_results = aggregate_matches(True, team, date, matches)
    last_away_results = aggregate_matches(False, team, date, matches)

    # All missing results will be NaN
    return {
        'LastSeasonStanding': last_season_standing,
        'HomeWins': last_home_results['WinPercentage'],
        'HomeDraws': last_home_results['DrawPercentage'],
        'AwayWins': last_away_results['WinPercentage'],
        'AwayDraws': last_away_results['DrawPercentage'],
        'HomeGoals': last_home_results['FTHG'],
        'AwayGoals': last_away_results['FTAG'],
        'HomeShots': last_home_results['HS'],
        'AwayShots': last_away_results['AS'],
        'HomeShotsOnTarget': last_home_results['HST'],
        'AwayShotsOnTarget': last_away_results['AST'],
        'HomeCorners': last_home_results['HC'],
        'AwayCorners': last_away_results['AC'],
        'HomeGoalsConceded': last_home_results['FTAG'],
        'AwayGoalsConceded': last_away_results['FTHG'],
        'HomeShotsConceded': last_home_results['AS'],
        'AwayShotsConceded': last_away_results['HS']
    }


def get_match_stats(row: pd.Series, matches: int = 5) -> Union[dict, None]:
    """
    Retrieve stats for both home and away teams in each given match. Only home stats are aggregated for the home team,
    and away stats for the away team.
    :param row: single row from all_matches containing info on one match
    :param matches: number of previous matches to aggregate
    :return: dict containing aggregated stats for home and away teams, or None if one team is missing data
    """
    date, season, home, away = row['Date'], row['Season'], row['HomeTeam'], row['AwayTeam']
    home_results = get_team_stats(home, date, season, matches)
    away_results = get_team_stats(away, date, season, matches)

    if pd.isnull(home_results['HomeGoals']) or pd.isnull(away_results['AwayGoals']):
        return

    return {
        'Season': season,
        'Date': date,
        'HomeTeam': home,
        'AwayTeam': away,
        'FTR': row['FTR'],
        'HomeGoals': row['FTHG'],
        'AwayGoals': row['FTAG'],
        'StandingDiff': home_results['LastSeasonStanding'] - away_results['LastSeasonStanding'],
        'HomeWins': home_results['HomeWins'],
        'AwayWins': away_results['AwayWins'],
        'HomeDraws': home_results['HomeDraws'],
        'AwayDraws': away_results['AwayDraws'],
        'AvgHomeGoals': home_results['HomeGoals'],
        'AvgAwayGoals': away_results['AwayGoals'],
        'AvgHomeShots': home_results['HomeShots'],
        'AvgAwayShots': away_results['AwayShots'],
        'AvgHomeShotsOnTarget': home_results['HomeShotsOnTarget'],
        'AvgAwayShotsOnTarget': away_results['AwayShotsOnTarget'],
        'AvgHomeCorners': home_results['HomeCorners'],
        'AvgAwayCorners': away_results['AwayCorners'],
        'AvgHomeGoalsConceded': home_results['HomeGoalsConceded'],
        'AvgAwayGoalsConceded': away_results['AwayGoalsConceded'],
        'AvgHomeShotsConceded': home_results['HomeShotsConceded'],
        'AvgAwayShotsConceded': away_results['AwayShotsConceded']
    }


def generate_train_val_test_sets(processed_data: pd.DataFrame, train_fraction: float = 0.8, random_seed: int = 123) -> \
        tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Shuffle and split dataset into train, validation and test sets. One-hot encode the FTR column.
    The last available season is always used as the test set.
    :param processed_data: dataframe containing all matches, created by running get_match_stats on each row of
    all_matches
    :param train_fraction: fraction of entries (after removing test split) to put into train split
    :param random_seed: random seed for shuffling
    :return: the shuffled train and val splits, and test split (features and labels for each)
    """
    last_season = processed_data['Season'].values[-1]
    test = processed_data[processed_data['Season'] == last_season]
    train_val = processed_data[processed_data['Season'] != last_season]

    # Shuffle train and val sets
    np.random.seed(random_seed)
    n_samples = len(train_val)
    shuffled_data = train_val.iloc[np.random.permutation(n_samples)]
    train_cutoff = int(train_fraction * n_samples)
    train = shuffled_data[:train_cutoff]
    val = shuffled_data[train_cutoff:]

    drop_cols = ['Season', 'Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'AvgHomeCorners', 'AvgAwayCorners']

    X_train = train.drop(columns=drop_cols)
    y_train = X_train.pop('FTR')
    y_train = label_binarizer(y_train)

    X_val = val.drop(columns=drop_cols)
    y_val = X_val.pop('FTR')
    y_val = label_binarizer(y_val)

    X_test = test.drop(columns=drop_cols)
    y_test = X_test.pop('FTR')
    y_test = label_binarizer(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def label_binarizer(labels: pd.Series):
    """
    One-hot encode labels for prediction.
    :param labels: full time results H/D/A.
    :return: array of one-hot encoded labels. H=[1,0,0], D=[0,1,0], A=[0,0,1]
    """
    one_hot = [label_map[label] for label in labels.values]
    return np.array(one_hot)


if __name__ == '__main__':
    data = pd.DataFrame(columns=['Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HomeGoals', 'AwayGoals',
                                 'StandingDiff', 'HomeWins', 'AwayWins', 'HomeDraws', 'AwayDraws', 'AvgHomeGoals',
                                 'AvgAwayGoals', 'AvgHomeShots', 'AvgAwayShots', 'AvgHomeShotsOnTarget',
                                 'AvgAwayShotsOnTarget', 'AvgHomeCorners', 'AvgAwayCorners', 'AvgHomeGoalsConceded',
                                 'AvgAwayGoalsConceded', 'AvgHomeShotsConceded', 'AvgAwayShotsConceded'])
    skipped = 0

    for ind, row in tqdm(all_matches.iterrows(), total=all_matches.shape[0]):
        match_stats = get_match_stats(row, 5)
        if match_stats is not None:
            data = data.append([match_stats], ignore_index=True)
        else:
            skipped += 1

    print('%d matches skipped due to insufficient data. %d valid matches loaded' % (skipped, len(data)))
    data.to_csv('data/EPL_processed_results.csv')
