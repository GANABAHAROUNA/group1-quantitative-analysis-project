import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from data import fetch_data

# Load the data for analysis
df = fetch_data('2023-12-01')

# Prepare for predictions
results = []
metrics = []

# Iterate through each unique stock in the dataset
for stock in df['Stkcd'].unique():
    # Filter data for the current stock
    stock_data = df[df['Stkcd'] == stock].copy()

    # Calculate 5-day moving averages for closing and high prices
    stock_data['moving_average_clsprc'] = stock_data['Clsprc'].rolling(window=5).mean()
    stock_data['moving_average_hiprc'] = stock_data['Hiprc'].rolling(window=5).mean()

    # Prepare data for linear regression
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y_clsprc = stock_data['Clsprc']
    y_hiprc = stock_data['Hiprc']

    # Train linear regression models on the entire dataset for both closing and high prices
    model_clsprc = LinearRegression().fit(X, y_clsprc)
    model_hiprc = LinearRegression().fit(X, y_hiprc)

    # Predict the closing price and high prices for the upcoming days
    predicted_clsprc = model_clsprc.predict([[len(X)]])
    predicted_hiprc_4 = model_hiprc.predict([[len(X) + 3]])
    predicted_hiprc_5 = model_hiprc.predict([[len(X) + 4]])
    predicted_hiprc_6 = model_hiprc.predict([[len(X) + 5]])

    # Determine the maximum predicted high price
    max_predicted_hiprc = max(predicted_hiprc_4[0], predicted_hiprc_5[0], predicted_hiprc_6[0])

    # Calculate the predicted rate of return using the formula
    predicted_return = (max_predicted_hiprc - predicted_clsprc[0]) / predicted_clsprc[0]
    results.append((stock, predicted_return))

# Sort the stocks based on predicted return and select the top 3
top_stocks = sorted(results, key=lambda x: x[1], reverse=True)[:3]

# Calculate and print performance metrics for the top stocks
for stock, _ in top_stocks:
    stock_data = df[df['Stkcd'] == stock]

    X = np.arange(len(stock_data)).reshape(-1, 1)
    y_clsprc = stock_data['Clsprc']
    y_hiprc = stock_data['Hiprc']

    # Retrain models for performance evaluation
    model_clsprc = LinearRegression().fit(X, y_clsprc)
    model_hiprc = LinearRegression().fit(X, y_hiprc)

    # Predictions for both closing and high prices
    predictions_clsprc = model_clsprc.predict(X)
    predictions_hiprc = model_hiprc.predict(X)

    # Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE)
    mse_clsprc = mean_squared_error(y_clsprc, predictions_clsprc)
    mae_clsprc = mean_absolute_error(y_clsprc, predictions_clsprc)
    mape_clsprc = np.mean(np.abs((y_clsprc - predictions_clsprc) / y_clsprc)) * 100

    mse_hiprc = mean_squared_error(y_hiprc, predictions_hiprc)
    mae_hiprc = mean_absolute_error(y_hiprc, predictions_hiprc)
    mape_hiprc = np.mean(np.abs((y_hiprc - predictions_hiprc) / y_hiprc)) * 100

    # Print the metrics for each top stock
    print(f"MSE for Clsprc of Stock {stock}:", mse_clsprc)
    print(f"MAE for Clsprc of Stock {stock}:", mae_clsprc)
    print(f"MAPE for Clsprc of Stock {stock}:", mape_clsprc)
    print(f"MSE for Hiprc of Stock {stock}:", mse_hiprc)
    print(f"MAE for Hiprc of Stock {stock}:", mae_hiprc)
    print(f"MAPE for Hiprc of Stock {stock}:", mape_hiprc)

# Print the top 3 stocks with their predicted rate of return
print("Top 3 Stocks:", top_stocks)