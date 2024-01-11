import pandas as pd

# Function that reads data and manipulates it so it can be used for building the models
def fetch_data(date): # date in format '2023-01-01' (YYYY-MM-DD)
    
    file1 = 'TRD_Dalyr.csv'
    file2 = 'TRD_Dalyr1.csv'

    df1 = pd.read_csv(file1, dtype={'Stkcd': str})
    df2 = pd.read_csv(file2, dtype={'Stkcd': str})
    df_raw = pd.concat([df1, df2], ignore_index=True) #merge dataframes

    df = df_raw[['Stkcd','Trddt','Opnprc','Loprc','Dnshrtrd','Dnvaltrd','Hiprc','Clsprc', 'Dsmvosd','Dsmvtll','Dretwd','Dretnd','Capchgdt','Ahshrtrd_D','Ahvaltrd_D']].copy()

    ## Maniupulate data set to calculate rate of return
    # Add column with maximum high price of the 4./5./6. next day
    df['high_4'] = df['Hiprc'].shift(-4)
    df['high_5'] = df['Hiprc'].shift(-5)
    df['high_6'] = df['Hiprc'].shift(-6)
    df['max_high_4_5_6'] = df[['high_4', 'high_5', 'high_6']].max(axis=1)

    # Add column with closing price of next day
    df['next_day_close'] = df['Clsprc'].shift(-1)

    # Only data from December
    df = df[df['Trddt'] >= date]
    df['Trddt'] = pd.to_datetime(df['Trddt'])

    # Fill NaN Values
    for column in df.columns:
        if (column != 'Trddt') and (column != 'Stkcd'):
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

    # Add column with rate of return
    df['rate_of_return'] = (df['max_high_4_5_6'] - df['next_day_close']) / df['next_day_close']

    # Add column with high - low
    df['high_low'] = df['Hiprc'] - df['Loprc']

    return df


