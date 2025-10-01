# Time-Aware Price Anomaly Detection

During an internal audit, a senior manager from a real estate company wanted to identify unusual purchase transactions that might indicate errors or irregularities in the procurement process. To assist, I built a Python script that analyzes purchase data and highlights anomalies, making it easier for the audit team to review suspicious entries quickly.

This repository contains a simplified, open-source version of that idea, rewritten with Streamlit for an interactive experience. Users can upload a CSV file containing purchase records with the columns:

- ItemName
- TransactionDate
- UnitPrice

The app then processes the data and generates a downloadable report with additional columns indicating whether a record is flagged as an anomaly. This makes the tool easy to adapt for learning, experimentation, or use in similar data-checking scenarios.

---

## Features
- Upload and preprocess transactional purchase data from CSV or Excel
- Estimate Expected Price for each item (based on historical data)
- Calculate deviation between actual and expected prices
- Apply rolling mean & rolling standard deviation per item (time-aware)
- Flag anomalies when deviations exceed normal variation thresholds
- Export processed results with anomaly flags into a downloadable CSV


## Usage

Clone the repository and run the Streamlit app locally:

```bash
git clone https://github.com/mcseiu05/purchase-anomaly.git
cd purchase-anomaly
streamlit run main.py
```
Or access the deployed app directly here:
https://purchase-anomaly.streamlit.app


## Use Cases

- **Procurement Monitoring** – identify suppliers charging unusually high prices  
- **Retail Analytics** – detect abnormal spikes or drops in product prices  
- **Financial Auditing** – flag unusual purchase patterns for further review  
- **Manufacturing** – track raw material costs and highlight sudden fluctuations  
- **Fraud Detection** – uncover manipulated or erroneous transaction entries  


