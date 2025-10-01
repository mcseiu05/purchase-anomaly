import streamlit as st
import pandas as pd
from io import BytesIO

# Page config
st.set_page_config(page_title="Purchase Anomaly Detection App", layout="wide")

st.title("📊 Purchase Anomaly Detection")
st.write(
    "Upload your dataset with columns: **ItemName, TransactionDate, UnitPrice**. "
    "The app will detect unusual price deviations and provide a downloadable results file."
)

st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    required_cols = {"ItemName", "TransactionDate", "UnitPrice"}
    if not required_cols.issubset(df.columns):
        st.error(f"Your dataset must contain the following columns: {required_cols}")
    else:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        df['ExpectedPrice'] = df.groupby('ItemName')['UnitPrice'].transform('mean')
        df['Deviation'] = (df['UnitPrice'] - df['ExpectedPrice']) / df['ExpectedPrice']
        df = df.sort_values(by=['ItemName', 'TransactionDate'])
        df['RollingMean'] = (
            df.groupby('ItemName')['Deviation']
              .transform(lambda x: x.rolling(window=60, min_periods=10).mean())
        )
        df['RollingStd'] = (
            df.groupby('ItemName')['Deviation']
              .transform(lambda x: x.rolling(window=60, min_periods=10).std())
        )
        df['IsAnomaly'] = df['Deviation'] > (df['RollingMean'] + 2 * df['RollingStd'])

        st.subheader("Processed Data with Anomalies")
        st.dataframe(df.head(20))

        # Download button
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="purchase_anomalies.csv",
            mime="text/csv"
        )
