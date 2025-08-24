import pandas as pd

# Load CSV and parse TransactionDate as datetime
df = pd.read_csv("purchase_price.csv", parse_dates=['TransactionDate'])

# Sort by ItemName and date to keep time order
df = df.sort_values(by=['ItemName', 'TransactionDate'])

# Calculate expected price using Exponential Weighted Moving Average (EWMA)
df['ExpectedPrice'] = (
    df.groupby('ItemName')['UnitPrice']
      .transform(lambda x: x.ewm(span=60, adjust=False).mean())
)

# Calculate deviation from expected price
df['Deviation'] = (df['UnitPrice'] - df['ExpectedPrice']) / df['ExpectedPrice']

# Rolling mean of deviation (per ItemName)
df['RollingMean'] = (
    df.groupby('ItemName')['Deviation']
      .transform(lambda x: x.rolling(window=60, min_periods=10).mean())
)

# Rolling std deviation of deviation (per ItemName)
df['RollingStd'] = (
    df.groupby('ItemName')['Deviation']
      .transform(lambda x: x.rolling(window=60, min_periods=10).std())
)

# Flag anomalies where deviation is significantly higher than normal
df['IsAnomaly'] = df['Deviation'] > (df['RollingMean'] + 2 * df['RollingStd'])

# Save results
df.to_csv("time_aware_anomalies.csv", index=False)
