# ⏱️ Time-Aware Price Anomaly Detection

During an internal audit, a real estate company wanted to identify unusual purchase transactions that might indicate errors or irregularities in their procurement process. To support this need, I built a script that analyzes purchase data and highlights anomalies, making it easier for auditors to review suspicious entries.

This repository contains a simplified, open-source version, rewritten with Streamlit for an interactive experience. Users can upload a CSV/Excel file containing purchase records with the following columns:

- ItemName
- TransactionDate
- UnitPrice

The app processes the data, detects anomalies, and generates a downloadable report with additional columns showing flagged records. This makes it easy to adapt for learning, experimentation, or real-world data-checking scenarios.

---

## ✨Features
-  📂 Upload and preprocess transactional purchase data (CSV/Excel)
- 📈 Estimate expected price per item (based on historical patterns)
- 🔎 Calculate deviations between actual and expected prices
- 🌀 Apply rolling mean & standard deviation per item (time-aware)
- 🚩 Flag anomalies when deviations exceed normal variation thresholds
- 💾 Export processed results with anomaly flags into a downloadable CSV


## 🚀 Usage

Clone the repository and run the Streamlit app locally:

```bash
git clone https://github.com/mcseiu05/purchase-anomaly.git
cd purchase-anomaly
streamlit run main.py
```
Or access the deployed app directly here:
https://purchase-anomaly.streamlit.app


## 🔧Use Cases

- 🛒**Procurement Monitoring** – identify suppliers charging unusually high prices  
- 📊**Retail Analytics** – detect abnormal spikes or drops in product prices  
- 🧾**Financial Auditing** – flag unusual purchase patterns for further review  
- ⚙️**Manufacturing** – track raw material costs and highlight sudden fluctuations  
- 🚨**Fraud Detection** – uncover manipulated or erroneous transaction entries


