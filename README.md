# Time-Aware Price Anomaly Detection

This project provides a **Python + Pandas implementation** for detecting anomalies in product purchase prices over time.  
It uses **Exponential Weighted Moving Average (EWMA)** and **rolling window statistics** to model expected price behavior and flag unusual deviations.

---

## Features
- Load and preprocess transactional purchase data from CSV  
- Compute **Expected Price** using EWMA  
- Measure deviation between actual and expected prices  
- Apply **rolling mean & rolling standard deviation** per item  
- Flag anomalies when deviation is significantly higher than normal  
- Export results into a new CSV with anomaly flags  

---

## Input Data Format

The input CSV (`purchase_price.csv`) should contain at least:

| ItemName   | TransactionDate | UnitPrice |
|------------|-----------------|-----------|
| Apple      | 2023-01-01      | 10.0      |
| Apple      | 2023-01-02      | 10.5      |
| Orange     | 2023-01-01      | 5.2       |

---

## How It Works
1. Sort by `ItemName` and `TransactionDate` to maintain time order.  
2. Use **EWMA** (`span=60`) to calculate the smoothed expected price.  
3. Calculate deviation:  

Deviation = (UnitPrice - ExpectedPrice) / ExpectedPrice


4. Compute rolling mean and rolling standard deviation of deviation.  
5. Flag anomaly when:  



Deviation > RollingMean + 2 × RollingStd


6. Save the results to `time_aware_anomalies.csv`.  

---

## Example Output

| ItemName | TransactionDate | UnitPrice | ExpectedPrice | Deviation | RollingMean | RollingStd | IsAnomaly |
|----------|-----------------|-----------|---------------|-----------|-------------|------------|-----------|
| Apple    | 2023-01-01      | 10.0      | 10.0          | 0.00      | NaN         | NaN        | False     |
| Apple    | 2023-01-02      | 15.0      | 10.2          | 0.47      | 0.05        | 0.10       | True      |

---

## Usage

Clone the repository and run the script:

```bash
git clone https://github.com/yourusername/time-aware-anomaly-detection.git
cd time-aware-anomaly-detection
python detect_anomalies.py
```

Make sure your input CSV is named purchase_price.csv (or update the script accordingly).
---

## Use Cases

Procurement monitoring – detect suppliers charging unusually high prices
Retail analytics – spot abnormal price spikes or drops
Financial auditing – flag unusual price patterns for investigation
Manufacturing – monitor raw material costs and spot sudden fluctuations
Fraud detection – detect manipulated or erroneous transaction entries


## Requirements

Python 3.8+
Pandas


## Install dependencies:

```bash
pip install pandas
```
