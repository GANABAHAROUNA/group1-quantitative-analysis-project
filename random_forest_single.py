import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time

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

#end_date = new_df['Trddt'].max()
#start_date = end_date - pd.DateOffset(months=12)
#new_df = new_df[(new_df['Trddt'] >= start_date) & (new_df['Trddt'] <= end_date)]

# Define features and target variable
new_df['max_high_in_period'] = new_df.groupby('Stkcd')['Hiprc'].rolling(window=3).max().reset_index(0,drop=True)
new_df['closing_price_at_beginning'] = new_df.groupby('Stkcd')['Clsprc'].shift(4)
new_df['RET'] = (new_df['max_high_in_period'] - new_df['closing_price_at_beginning']) / new_df['closing_price_at_beginning']
new_df = new_df.dropna()

# Gradient Boosting
features = ['Clsprc', 'Hiprc', 'Loprc', 'Dnvaltrd', 'Dsmvtll']  # closing price, highprice, lowprice, shares traded, shares outstanding
target = 'RET'

# Feature Engineering
# Create a new DataFrame to store features
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

#New feature
features_df['Hiprc_Loprc_difference'] = features_df['Hiprc'] - features_df['Loprc']

# Drop rows with NaN values resulting from new features
features_df = features_df.dropna()

# Create lagged features
for feature in features:
    for i in range(1, 4):  # Adjust the lag window as needed
        features_df[f'{feature}_lag{i}'] = features_df.groupby('Stkcd')[feature].shift(i)

# Drop rows with NaN values resulting from lagged features
features_df = features_df.dropna()

# Set the training-testing split proportion
split_proportion = 0.8

# Initialize an empty dictionary to store models and predictions
models = {}
predictions = {}

print("Total stocks : ", len(features_df['Stkcd'].unique()))
time.sleep(1)

ticker_counts = features_df['Stkcd'].value_counts()

# Select randomly 3 tickers (test mode)
#selected_tickers = np.random.choice(features_df['Stkcd'], size=3)
selected_tickers = features_df['Stkcd'].index.unique() #for All tickers

for stock in selected_tickers.unique():
#for stock in selected_tickers:
    stock_df = features_df[features_df['Stkcd'] == stock]    

    # Define your new features
    new_features = [f'{feature}_rolling_mean', f'{feature}_rolling_std', f'{feature}_rel_change', 'Dnvaltrd_Dsmvtll_interaction', 'Hiprc_Loprc_interaction', 'Hiprc_Loprc_difference']

    # Add the new features to the existing ones
    all_features = features + new_features
    
    # Include lagged features
    lagged_features = [f'{feature}_lag{i}' for feature in features for i in range(1, 4)]
    all_features += lagged_features
    
    # Split the data into features and target variable
    X = stock_df[all_features]
    y = stock_df[target]

    # Determine the split index based on the split proportion
    split_index = int(split_proportion * len(stock_df))

    # Split the data into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_dates = stock_df['Trddt'][split_index:]

    # Initialize the Random Forest Regressor for each ticker
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        n_jobs=-1, 
        random_state=42, 
        verbose=100)

    try:
        rf_model.fit(X_train, y_train)
    except ValueError:
        continue
    
    # Predict on the testing data
    y_pred = rf_model.predict(X_test)

    # Calculate MAE and MSE
    mae = mean_absolute_error(y_test, y_pred).round(4)

    print(f"Ticker: {stock}, Mean Absolute Error (MAE): {mae}")

    # Store the model and predictions for each ticker
    models[stock] = rf_model
    predictions[stock] = {'actual': y_test, 'predicted': y_pred}

    print(stock, "Training : ", len(X_train), "Testing : ", len(predictions[stock]['actual']), "Total Records :", len(stock_df))

"""
    # Plotting actual vs. predicted returns
    plt.figure(figsize=(10, 6))
       
    # Plot the actual and predicted returns for the test periods
    plt.plot(test_dates, predictions[stock]['actual'], label='Actual Returns', marker='o')
    plt.plot(test_dates, predictions[stock]['predicted'], label='Predicted Returns', marker='o')
    
    plt.title(f'Random Forest Regression for {stock} MAE = {mae}')
    plt.xlabel('Date')
    plt.ylabel('Returns')

    plt.legend()
    plt.show()
    #plt.savefig(str(ticker) + '_RF.png')
    plt.draw()
    plt.pause(0.05)  # Adjust pause duration as needed
    
# Keep plots open at the end
plt.show(block=True)
"""