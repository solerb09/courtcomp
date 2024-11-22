import pandas as pd

stats = pd.read_csv('2023_nba_player_stats.csv')
salaries = pd.read_csv('Nba Player Salaries.csv')

salaries.rename(columns={'Player Name': 'PName'}, inplace=True)
print(salaries.columns)  # Check columns in stats DataFrame


# Select only the columns for player name and the 2022/2023 season
salaries_2022_2023 = salaries[["PName","2022/2023"]]



# Merge the filtered salaries with the stats DataFrame
merged_df = pd.merge(stats, salaries_2022_2023, on="PName", how="inner")


merged_df.rename(columns={'2022/2023': 'Salary'}, inplace=True)

merged_df["Salary"] = (
    merged_df["Salary"]
    .str.replace("$", "", regex=False)  # Remove the dollar sign
    .str.replace(",", "", regex=False)  # Remove commas
    .astype(int)  # Convert to integer
)

print(merged_df.head())




