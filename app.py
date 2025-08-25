import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime

# Load trained model
model = joblib.load("xgboost_model.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Stock VWAP Predictor", page_icon="📈", layout="centered")
st.title("📊 Real-Time Stock VWAP Predictor")

# Mode Selection
mode = st.sidebar.radio("Select Prediction Mode:", ["Auto Fetch (yfinance)", "Manual Input"])

if mode == "Auto Fetch (yfinance)":
    st.subheader("🔗 Auto Fetch Mode - Predict VWAP from Yahoo Finance Data")

    stock_dict = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices, Inc.",
        "WIPRO.NS": "Wipro Limited (India)"
    }

    ticker_display = [f"{symbol} ({name})" for symbol, name in stock_dict.items()]
    selection = st.selectbox("Select Stock Symbol:", ticker_display)
    ticker = selection.split(" ")[0]

    today = datetime.today().date()
    start_date = st.date_input("Start Date", pd.to_datetime("2025-04-01"), max_value=today)
    end_date = st.date_input("End Date", pd.to_datetime("2025-04-10"), max_value=today)

    if st.button("Fetch and Predict"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                st.error("❌ No data fetched! Check symbol or date range.")
            else:
                data.reset_index(inplace=True)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                data.columns = [col.replace(f" {ticker}", "").strip() for col in data.columns]
                data['Volume'] = data['Volume'].fillna(0)

                expected_features = ["Open", "High", "Low", "Close", "Volume"]
                if all(col in data.columns for col in expected_features):
                    X = data[expected_features]
                    predictions = model.predict(X)
                    data["Predicted_VWAP"] = predictions

                    st.success("✅ Predictions completed!")
                    st.dataframe(data[["Date"] + expected_features + ["Predicted_VWAP"]])

                    st.subheader("📊 VWAP Prediction Chart")
                    st.line_chart(data.set_index("Date")[["Close", "Predicted_VWAP"]])

                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button("📅 Download Predictions as CSV", data=csv, file_name=f"{ticker}_vwap_predictions.csv")
                else:
                    st.error("❌ Required columns not found after cleaning!")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

elif mode == "Manual Input":
    st.subheader("✍️ Manual Mode - Enter Stock Data for VWAP Prediction")

    manual_method = st.radio("Select Manual Prediction Method:", ["Single Input", "CSV Upload by Date", "CSV Upload by Date Range"])

    if manual_method == "Single Input":
        try:
            open_price = st.number_input("Open Price:", min_value=0.0, value=0.00, step=0.1)
            high_price = st.number_input("High Price:", min_value=0.0, value=0.00, step=0.1)
            low_price = st.number_input("Low Price:", min_value=0.0, value=0.00, step=0.1)
            close_price = st.number_input("Close Price:", min_value=0.0, value=0.00, step=0.1)
            volume = st.number_input("Volume:", min_value=0.0, value=0.00, step=1000.0)

            if st.button("Predict VWAP"):
                input_data = np.array([[open_price, high_price, low_price, close_price, volume]])
                predicted_vwap = model.predict(input_data)[0]
                st.success(f"📊 Predicted VWAP: {predicted_vwap:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    elif manual_method == "CSV Upload by Date":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                st.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
                target_date = st.date_input("Select date", value=None)
                row = df[df['Date'] == target_date]
                if not row.empty:
                    X = row[["Open", "High", "Low", "Close", "Volume"]]
                    df.loc[df['Date'] == target_date, 'Predicted_VWAP'] = model.predict(X)
                    st.success("✅ Prediction Done!")
                    st.dataframe(df[df['Date'] == target_date])
                else:
                    st.warning("No data available for selected date")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    elif manual_method == "CSV Upload by Date Range":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="range")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                # Ensure date is properly parsed
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df.reset_index(drop=True, inplace=True)

                st.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")

                start = st.date_input("Select start date", value=None, key="start_manual")
                end = st.date_input("Select end date", value=None, key="end_manual")

                if start and end:
                    if (end - start).days > 7:
                        st.warning("⚠️ Please select a range of 7 days or less.")
                    else:
                        mask = (df['Date'] >= start) & (df['Date'] <= end)
                        subset = df.loc[mask].copy()  # create a copy to avoid SettingWithCopyWarning

                        if not subset.empty:
                            X = subset[["Open", "High", "Low", "Close", "Volume"]]
                            subset['Predicted_VWAP'] = model.predict(X)

                            st.success("✅ Predictions for date range completed!")
                            st.dataframe(subset)

                            # Plot using subset (since index is Date)
                            st.line_chart(subset.set_index("Date")[["Close", "Predicted_VWAP"]])
                        else:
                            st.warning("No data available in selected date range")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")



st.sidebar.markdown("---")
st.sidebar.write("Made with using Streamlit, yfinance, and XGBoost")
