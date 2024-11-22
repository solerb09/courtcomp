import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


stats = pd.read_csv('2023_nba_player_stats_with_advanced.csv')
salaries = pd.read_csv('Nba Player Salaries.csv')

salaries.rename(columns={'Player Name': 'Player'}, inplace=True)
print(salaries.columns)  # Check columns in stats DataFrame


#Select only the columns for player name and the 2022/2023 season
salaries_2022_2023 = salaries[["Player","2022/2023"]]



#Merge the filtered salaries with the stats DataFrame
merged_df = pd.merge(stats, salaries_2022_2023, on="Player", how="inner")


merged_df.rename(columns={'2022/2023': 'Salary'}, inplace=True)

merged_df["Salary"] = (
    merged_df["Salary"]
    .str.replace("$", "", regex=False)  # Remove the dollar sign
    .str.replace(",", "", regex=False)  # Remove commas
    .astype(int)  # Convert to integer
)

merged_df.dropna(inplace=True)  # Drop rows with missing values

merged_df["PTS_per_Min"] = merged_df["PTS"] / merged_df["Min"]
merged_df["Efficient_Score"] = merged_df["PTS"] * merged_df["FG%"]
# Select the columns to be used for the model
scaler = StandardScaler()
X = merged_df[["Age", "Min", "PTS_per_Min", "Efficient_Score", "3PM", "3PA", "3P%", "FT%",
            "REB", "AST", "STL", "BLK", "PF", "DBPM",
               "+/-",]]
X_scaled = scaler.fit_transform(X)

y = merged_df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)


# Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Linear Regression - MAE: {mae_lr}, R2: {r2_lr}")
print(f"Random Forest - MAE: {mae_rf}, R2: {r2_rf}")


feature_importances = rf_model.feature_importances_
features = ["Age", "Min", "PTS_per_Min","Efficient_Score", "3PM", "3PA", "3P%", "FT%", 
            "REB", "AST",  "STL", "BLK", "PF", "DBPM", 
            "+/-",]


numeric_data = merged_df.select_dtypes(include=[np.number])  # Select only numeric columns
corr_matrix = numeric_data.corr()


plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()