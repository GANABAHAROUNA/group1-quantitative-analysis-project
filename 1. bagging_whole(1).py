import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pickle

# Load data
file1 = 'TRD_Dalyr.csv'
file2 = 'TRD_Dalyr1.csv'

df1 = pd.read_csv(file1,low_memory=False)
df2 = pd.read_csv(file2,low_memory=False)
new_df = pd.concat([df1, df2], ignore_index=True)  # Combine dataframes

# Ensure 'Trddt' column is in datetime format
new_df['Trddt'] = pd.to_datetime(new_df['Trddt'])
# Sort the DataFrame by date to ensure temporal order
new_df = new_df.sort_values(by='Trddt')

returns = []

print(new_df.columns)

new_df['max_high_in_period'] = new_df.groupby('Stkcd')['Hiprc'].rolling(window=3).max().reset_index(0,drop=True)
new_df['closing_price_at_beginning'] = new_df.groupby('Stkcd')['Clsprc'].shift(4)
new_df['RET'] = (new_df['max_high_in_period'] - new_df['closing_price_at_beginning']) / new_df['closing_price_at_beginning']
new_df = new_df.dropna()

# Gradient Boosting
features = ['Clsprc', 'Hiprc', 'Loprc', 'Dnvaltrd', 'Dsmvtll']  # closing price, highprice, lowprice, shares traded, shares outstanding
target = 'RET'

# Feature Engineering
features_df = new_df.copy()

# Create rolling statistics features
for feature in features:
    features_df[f'{feature}_rolling_mean'] = features_df.groupby('Stkcd')[feature].rolling(window=3).mean().reset_index(0, drop=True)
    features_df[f'{feature}_rolling_std'] = features_df.groupby('Stkcd')[feature].rolling(window=3).std().reset_index(0, drop=True)

# Create relative change features
for feature in features:
    features_df[f'{feature}_rel_change'] = features_df.groupby('Stkcd')[feature].pct_change()

# Create interaction features
features_df['Dnvaltrd_Dsmvtll_interaction'] = features_df['Dnvaltrd'] * features_df['Dsmvtll']
features_df['Hiprc_Loprc_interaction'] = features_df['Hiprc'] * features_df['Loprc']

# Drop rows with NaN values resulting from new features
features_df = features_df.dropna()

# Create lagged features
for feature in features:
    for i in range(1, 4):  # Adjust the lag window as needed
        features_df[f'{feature}_lag{i}'] = features_df.groupby('Stkcd')[feature].shift(i)

# Drop rows with NaN values resulting from lagged features
features_df = features_df.dropna()

# Split the dataset into features and target variable
X = features_df[features + [f'{feature}_lag{i}' for feature in features for i in range(1, 4)]]
y = features_df[target]

# Set the training-testing split proportion
split_proportion = 0.8
split_index = int(split_proportion * len(features_df))

# Split the data into training and testing sets
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize the Bagging Regressor with a base estimator (RandomForestRegressor in this case)
base_estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbose=100)
bagging_model = BaggingRegressor(base_estimator=base_estimator, n_estimators=100, n_jobs=-1, random_state=42, verbose=100)

# Fit the model on the training data
bagging_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = bagging_model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# save the model for later use
filename = 'model_Bagging.sav'
pickle.dump(bagging_model, open(filename, 'wb'))
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)
#print(result)

# Save the MAE for later use
with open('MAE_Bagging.txt', 'w') as f:
    f.write(str(mae))

# Plot actual vs. predicted returns
plt.figure(figsize=(10, 6))
plt.plot(features_df['Trddt'][split_index:], y_test, label='Actual Returns', marker='o')
plt.plot(features_df['Trddt'][split_index:], y_pred, label='Predicted Returns', marker='o')
plt.title('Actual vs. Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
#plt.show()
plt.savefig('Whole_Bagging_new.png')
plt.draw()
plt.pause(0.05)  # Adjust pause duration as needed
# Keep plots open at the end
plt.show(block=True)