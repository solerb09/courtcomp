import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
stats = pd.read_csv('2023_nba_player_stats_with_advanced.csv')
salaries = pd.read_csv('Nba Player Salaries.csv')

# Rename columns to align datasets
salaries.rename(columns={'Player Name': 'Player'}, inplace=True)

# Select relevant columns from salaries dataset
salaries_2022_2023 = salaries[["Player", "2022/2023"]]

# Merge the datasets on the Player column
merged_df = pd.merge(stats, salaries_2022_2023, on="Player", how="inner")

# Clean up and prepare the Salary column
merged_df.rename(columns={'2022/2023': 'Salary'}, inplace=True)
merged_df["Salary"] = (
    merged_df["Salary"]
    .str.replace("$", "", regex=False)  # Remove dollar sign
    .str.replace(",", "", regex=False)  # Remove commas
    .astype(int)  # Convert to integer
)

# Calculate % of Salary Cap
salary_cap = 123655000  # NBA salary cap for 2022-2023
merged_df["%Cap"] = (merged_df["Salary"] / salary_cap) * 100

# Drop rows with missing values
merged_df.dropna(inplace=True)

# Feature engineering
merged_df["PTS_per_Min"] = merged_df["PTS"] / merged_df["Min"]
merged_df["Efficient_Score"] = merged_df["PTS"] * merged_df["FG%"]
merged_df["Efficient_Score_Normalized"] = merged_df["Efficient_Score"] / merged_df["Min"]
merged_df["Weighted_Efficiency"] = 0.7 * merged_df["Efficient_Score_Normalized"] + 0.3 * merged_df["PTS"]

# Filter out players with very low minutes (optional)
merged_df = merged_df[merged_df["Min"] > 1000]

# Select features and target variable
features = ["Age", "Min", "Weighted_Efficiency", "PTS_per_Min", "3PM", "3PA", "3P%", "FT%",
            "REB", "AST", "STL", "BLK", "PF", "DBPM", "+/-"]

X = merged_df[features]
y = merged_df["%Cap"]  # Predict % of Salary Cap instead of raw salary

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Evaluate Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print evaluation metrics
print(f"Linear Regression - MAE (% of Cap): {mae_lr}, R2: {r2_lr}")
print(f"Random Forest - MAE (% of Cap): {mae_rf}, R2: {r2_rf}")

# Convert Predicted %Cap to Salary
merged_df["Predicted_%Cap"] = rf_model.predict(scaler.transform(X))
merged_df["Predicted_Salary"] = (merged_df["Predicted_%Cap"] / 100) * salary_cap

# Calculate salary difference (overpaid or underpaid)
merged_df["Value_Difference"] = merged_df["Salary"] - merged_df["Predicted_Salary"]

# Remove duplicate player names (if any)
merged_df = merged_df.drop_duplicates(subset=["Player"], keep="first")

# Highlight Most Overpaid and Underpaid Players
overpaid = merged_df[merged_df["Value_Difference"] > 0].sort_values(by="Value_Difference", ascending=False)
underpaid = merged_df[merged_df["Value_Difference"] < 0].sort_values(by="Value_Difference")

most_overpaid = overpaid.iloc[0] if not overpaid.empty else None
most_underpaid = underpaid.iloc[0] if not underpaid.empty else None

# Display All Players Stats
print("All Players Stats:\n")
for index, row in merged_df.iterrows():
    player_info = f"{row['Player']} | Age: {row['Age']} | Salary: {row['Salary']} | Predicted Salary: {row['Predicted_Salary']:.2f} | Value Difference: {row['Value_Difference']:.2f}  Actual %Cap: {row['%Cap']:.2f} | Predicted %Cap: {row['Predicted_%Cap']:.2f} | Value Difference: {row['Value_Difference']:.2f}"
    if most_overpaid is not None and row['Player'] == most_overpaid['Player']:
        print(f"**MOST OVERPAID**: {player_info}")
    elif most_underpaid is not None and row['Player'] == most_underpaid['Player']:
        print(f"**MOST UNDERPAID**: {player_info}")
    else:
        print(player_info)

# Visualize Actual vs Predicted Salary
plt.scatter(merged_df["Salary"], merged_df["Predicted_Salary"], alpha=0.7)
plt.plot([0, max(merged_df["Salary"])], [0, max(merged_df["Salary"])], color="red", linestyle="--")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

# Sample data
data = {
    'Player': ['Jayson Tatum', 'Joel Embiid', 'Shai Gilgeous-Alexander', 'Giannis Antetokounmpo', 'Anthony Edwards'],
    'Actual Salary': [30351780, 33616770, 30913750, 42492492, 10733400],
    'Predicted Salary': [29870519.27, 35137593.69, 29852785.34, 39451985.21, 12314104.53]
}
df = pd.DataFrame(data)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.4
index = range(len(df))

# Actual salaries
plt.bar(index, df['Actual Salary'], bar_width, label='Actual Salary', alpha=0.8)

# Predicted salaries
plt.bar([i + bar_width for i in index], df['Predicted Salary'], bar_width, label='Predicted Salary', alpha=0.8)

# Adding labels and title
plt.xlabel('Player', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.title('Actual vs Predicted Salary', fontsize=14)
plt.xticks([i + bar_width / 2 for i in index], df['Player'], rotation=45, ha='right')
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()