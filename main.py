import pandas as pd

df = pd.read_csv("purchase_price.csv", parse_dates=['TransactionDate'])
df = df.sort_values(by=['ItemName', 'TransactionDate'])
df['ExpectedPrice'] = df.groupby('ItemName')['UnitPrice'].transform(lambda x: x.ewm(span=60, adjust=False).mean())
df['Deviation'] = (df['UnitPrice'] - df['ExpectedPrice']) / df['ExpectedPrice']
df['RollingMean'] = df.groupby('ItemName')['Deviation'].transform(lambda x: x.rolling(window=60, min_periods=10).mean())
df['RollingStd'] = df.groupby('ItemName')['Deviation'].transform(lambda x: x.rolling(window=60, min_periods=10).std())
df['IsAnomaly'] = df['Deviation'] > (df['RollingMean'] + 2*df['RollingStd'])
df.to_csv("time_aware_anomalies.csv", index=False)