import pandas as pd

stats = pd.read_csv('2023_nba_player_stats.csv')
salaries = pd.read_csv('Nba Player Salaries.csv')


print(stats.head())
print(salaries.head())
