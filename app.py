import streamlit as st
import pandas as pd
import ta
import xgboost as xgb
import joblib
import numpy as np
import ccxt
import time

# ✅ Set the title of the Streamlit App
st.title("🚀 Real-Time Crypto Pump Prediction App")

# ✅ Load your trained model
model = xgb.XGBClassifier()
model.load_model("best_xgb_model.json")

# ✅ Define the features used during training
features = ['rsi', 'roc', 'stoch_k', 'macd', 'adx', 'aroon_up', 'aroon_down', 'obv',
            'mfi', 'vwap', 'bb_upper', 'bb_lower', 'bb_width', 'atr', 'williams_r',
            'cci', 'tsi', 'ichimoku_base', 'psar', 'force_index', 'adl', 'kc_width',
            'donchian_width', 'rolling_mean_7', 'ema_30', 'close_1', 'close_5',
            'close_10', 'volume_1', 'volume_5', 'roc_5', 'macd_5']

# ✅ Function to collect live data from Binance
def fetch_live_data(symbol='ETH/USDT', timeframe='1m', limit=100):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ✅ Function to preprocess data and calculate features
def preprocess_data(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    aroon = ta.trend.AroonIndicator(df['high'], df['low'])
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()
    df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
    df['ichimoku_base'] = ta.trend.IchimokuIndicator(df['high'], df['low']).ichimoku_base_line()

    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['force_index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
    df['adl'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()

    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['kc_width'] = kc.keltner_channel_wband()
    donchian = ta.volatility.DonchianChannel(df['high'], df['low'])
    df['donchian_width'] = donchian.donchian_channel_wband()

    df['rolling_mean_7'] = df['close'].rolling(window=7).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()

    df['close_1'] = df['close'].shift(1)
    df['close_5'] = df['close'].shift(5)
    df['close_10'] = df['close'].shift(10)
    df['volume_1'] = df['volume'].shift(1)
    df['volume_5'] = df['volume'].shift(5)
    df['roc_5'] = df['roc'].shift(5)
    df['macd_5'] = df['macd'].shift(5)

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

# ✅ Make a prediction
def make_prediction(df, model, threshold=0.2777):
    latest_data = df[features].iloc[-1:].values
    prediction_prob = model.predict_proba(latest_data)[0][1]
    prediction = int(prediction_prob >= threshold)
    return prediction, prediction_prob

# ✅ Streamlit UI
symbol = st.sidebar.text_input("Enter Trading Pair (e.g., ETH/USDT):", value="ETH/USDT")
threshold = st.sidebar.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.2777)

if st.sidebar.button("Predict"):
    df = fetch_live_data(symbol=symbol)
    df = preprocess_data(df)
    prediction, prediction_prob = make_prediction(df, model, threshold)

    if prediction == 1:
        st.success(f"🚀 Pump Detected! Probability: {prediction_prob:.4f}")
    else:
        st.warning(f"📉 No Pump Detected. Probability: {prediction_prob:.4f}")

    st.subheader("📊 Latest Data")
    st.write(df.tail(10))

    st.subheader("📈 Price Chart")
    st.line_chart(df['close'][-30:])
