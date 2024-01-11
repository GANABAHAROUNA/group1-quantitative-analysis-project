# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
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
print(new_df.info())
time.sleep(1)
#end_date = new_df['Trddt'].max()
#start_date = end_date - pd.DateOffset(days=30)

#new_df = new_df[(new_df['Trddt'] >= start_date) & (new_df['Trddt'] <= end_date)]

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
features_df['Hiprc_Loprc_interaction'] = features_df['Hiprc'] * features_df['Loprc']
features_df['Dnvaltrd_Dsmvtll_interaction'] = features_df['Dnvaltrd'] * features_df['Dsmvtll']

#Additional feature
features_df['Hiprc_Loprc_diff_interaction'] = features_df['Hiprc'] - features_df['Loprc']

# Drop rows with NaN values resulting from new features
features_df = features_df.dropna()

# Create lagged features
for feature in features:
    for i in range(1, 4):  # Adjust the lag window as needed
        features_df[f'{feature}_lag{i}'] = features_df.groupby('Stkcd')[feature].shift(i)

# Drop rows with NaN values resulting from lagged features
features_df = features_df.dropna()
print(features_df.info())
time.sleep(2)

# Set the training-testing split proportion
split_proportion = 0.8

# Initialize an empty dictionary to store models and predictions
models = {}
predictions = {}

print("Total Tickers : ", len(features_df['Stkcd'].unique()))
# Name and index of the ticker with the highest number of observations
ticker_counts = features_df['Stkcd'].value_counts()

# Select the top 3 tickers with the highest counts (testing stage)
#selected_tickers = ticker_counts.nlargest(5).index

# Dictionary to store predictions for the next three days
next_four_days_predictions = {}
#for ticker in selected_tickers.unique():
for stock in features_df['Stkcd'].unique():
    stock_df = features_df[features_df['Stkcd'] == stock]

    # Define your new features
    new_features = [f'{feature}_rolling_mean', f'{feature}_rolling_std', f'{feature}_rel_change', 'Dnvaltrd_Dsmvtll_interaction', 'Hiprc_Loprc_interaction','Hiprc_Loprc_diff_interaction']#, 'Hiprc_Loprc_difference']

    # Add the new features to the existing ones
    all_features = features + new_features

    # Include lagged features
    lagged_features = [f'{feature}_lag{i}' for feature in features for i in range(1, 4)]
    all_features += lagged_features

    # Split the data into features and target variable
    X = stock_df[all_features]
    y = stock_df[target]

    # Split the data into different periods
    # Create periods based on date
    stock_df['period'] = pd.qcut(stock_df['Trddt'], q=100, labels=False,duplicates='drop')

    # Get unique periods
    unique_periods = stock_df['period'].unique()

    # Randomly sample periods for training
    # Ensure that the last bin is included in the testing set
    train_periods = pd.Series(unique_periods[:-1]).sample(frac=split_proportion, random_state=42)

    # The remaining periods are for testing
    test_periods = pd.Series(unique_periods).drop(train_periods.index)

    # Include the last bin in the testing set
    last_bin = unique_periods[-1]
    test_periods = test_periods._append(pd.Series([last_bin]))

    # Filter the original dataframe based on the periods
    train_mask = stock_df['period'].isin(train_periods)
    test_mask = stock_df['period'].isin(test_periods)

    # Create the training and test sets
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Get the dates for the test periods
    test_dates = stock_df['Trddt'][test_mask]

    # Output lengths for verification
    print(len(X_train), len(X_test))
    print(len(y_train), len(y_test))

    # Adjust parameters to add more regularization
    gb_model = GradientBoostingRegressor(
        n_estimators=100,  # Decrease the number of trees
        learning_rate=0.2,  # Increase the learning rate
        max_depth=3,  # Limit the depth of the trees
        min_samples_split=10,  # Require more samples to split a node
        min_samples_leaf=5,  # Require more samples in a leaf node
        random_state=42,
        verbose=100
    )

    try:
        gb_model.fit(X_train, y_train)
    except ValueError:
        continue
    
    # Predict on the testing data
    y_pred = gb_model.predict(X_test)

        # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred).round(4)
    #accuracy = accuracy_score(y_test,y_pred).round(4)
    print(f'Ticker: {stock}, Mean Absolute Error (MAE): {mae}')

    # Store the model and predictions for each ticker
    models[stock] = gb_model
    predictions[stock] = {'actual': y_test, 'predicted': y_pred}

    print(stock, "Training : ", len(X_train), "Testing : ", len(predictions[stock]['actual']), "Total Records :", len(stock_df))


#Plotting actual vs. predicted returns
    plt.figure(figsize=(10, 6))
       
    # Plot the actual and predicted returns for the test periods
    plt.plot(test_dates, predictions[stock]['actual'], label='Actual Returns', marker='o')
    plt.plot(test_dates, predictions[stock]['predicted'], label='Predicted Returns', marker='o')
    
    plt.title(f'Gradient Boosting Regression Prediction for {stock} MAE = {mae}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    #plt.show()
    plt.savefig(str(stock) + '_GB.png')
    plt.draw()
    plt.pause(0.5)  # Adjust pause duration as needed
    
    # Keep plots open at the end
    #plt.show(block=True)

    # Predict on the next four days
    next_four_days_features = X.tail(4)  # Assuming the data is sorted by date
    next_four_days_predictions[stock] = gb_model.predict(next_four_days_features)

four_days = ['2023-12-22','2023-12-25','2023-12-26','2023-12-27']

top_three_tickers = sorted(next_four_days_predictions, key=lambda x: next_four_days_predictions[x][-1], reverse=True)[:3]

print("Predictions for December 28, stocks with highest possible returns:\n")

for t in top_three_tickers:
    print(f"Stock: {t} has a possible return of", next_four_days_predictions[t][-1].round(4))
# Display predictions for all stocks in the next three days
#for ticker, predictions_array in next_four_days_predictions.items():
    #print(f'Ticker: {ticker}, Predictions for the next four days: {predictions_array}')


for stocks in top_three_tickers:
    plt.figure(figsize=(10,6))
    plt.plot(four_days,next_four_days_predictions[stocks],label=f'{stocks} Predictions')    
    plt.title(f'Predicted Returns for the next four days for Stock: {stocks}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Returns')
    plt.legend()
    #plt.show()
    plt.draw()
    plt.savefig(str(stocks) + '_GB.png')

    plt.pause(3)  # Adjust pause duration as needed
