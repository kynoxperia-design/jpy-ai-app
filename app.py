import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dashboard Ultra", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 8px; min-height: 90px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: bold !important; }
    .time-header { font-size: 1.1rem; font-weight: bold; text-align: center; margin-bottom: 5px; color: #00ff00; border-bottom: 2px solid #00ff00; padding-bottom: 5px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; margin-top: 10px; text-align: center; font-weight: bold; }
    .price-subtext { font-size: 0.75rem; color: #888888; text-align: center; margin-top: -5px; }
    .tech-subtext { font-size: 0.7rem; color: #55aaff; text-align: center; margin-top: 2px; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安定したデータ取得 ---
@st.cache_data(ttl=60)
def fetch_fx_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 10: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# 最新価格
data_latest = fetch_fx_data("JPY=X", "5d", "1m")
current_price = float(data_latest['Close'].iloc[-1]) if data_latest is not None else 0.0
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 精度特化型：予測エンジン ---
def predict_engine_ultra(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data(ticker, period, interval)
    if df is None or len(df) < 50: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # --- 特徴量エンジニアリング（プロ仕様） ---
        # 1. 基本指標
        df['RSI'] = ta.rsi(df['Close'], length=14)
        # 2. トレンドの勢い (ADX)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']
        # 3. ボリンジャーバンドの幅（ボラティリティ収束・拡散）
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / df['Close']
        # 4. 上位足のトレンドを擬似的に取り込む (長期移動平均との乖離)
        df['EMA200_Dist'] = (df['Close'] - ta.ema(df['Close'], length=200)) / df['Close']
        
        # 目的変数
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去価格特定
        idx = (-2 if is_daily else -offset) if offset > 0 else -1
        past_price = float(df['Close'].iloc[idx])
        past_row = df.iloc[[idx]]

        # 学習データのクレンジング
        df_train = df.dropna()
        features = ['RSI', 'ADX', 'BB_Width', 'EMA200_Dist']
        X = df_train[features]
        y = df_train['Target']
        
        # モデル最適化 (決定木を500本、より細かな分岐を許可)
        model = RandomForestClassifier(
            n_estimators=500, 
            max_depth=15, 
            min_samples_split=4,
            random_state=42
        )
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        # 予測
        eval_row = df.dropna().tail(1) if offset == 0 else past_row.fillna(method='ffill')
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. メイン表示 ---
st.title("🦅 FX-AI Dashboard Ultra")
st.caption(f"高密度学習モデル稼働中 | 日本時間: {jst_now.strftime('%H:%M:%S')}")

st.markdown(f"""
    <div style="background-color: #000000; padding: 10px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 10px;">
        <p style="color: #00ff00; margin: 0; font-size: 0.9rem;">USD/JPY LIVE</p>
        <p style="color: #00ff00; margin: 0; font-size: 2.8rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

if st.button('🔄 市場データを再学習（深層分析）'): st.rerun()

st.divider()

timeframes = {
    "10分": {"p": ("2d","1m",10), "o": 10, "d": False},
    "1時間": {"p": ("7d","5m",12), "o": 12, "d": False},
    "4時間": {"p": ("30d","15m",16), "o": 16, "d": False},
    "1日": {"p": ("2y","1d",1), "o": 1, "d": True}
}

cols = st.columns(4)

for i, (label, cfg) in enumerate(timeframes.items()):
    with cols[i]:
        st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
        
        # 実績と予測
        p_val, p_dir, _, _ = predict_engine_ultra("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
        _, f_dir, f_prob, f_rsi = predict_engine_ultra("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
        
        st.markdown(f'<p class="section-label">実績</p>', unsafe_allow_html=True)
        if p_val > 0:
            diff = current_price - p_val
            st.metric("", "📈上昇中" if diff > 0 else "📉下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f}→{current_price:.2f}</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p class="section-label">AI予測</p>', unsafe_allow_html=True)
        # 判定をより厳格に (確信度 54%以下は中立)
        if max(f_prob) < 0.54:
            st.metric("", "⚖️中立", "低確信")
        else:
            st.metric("", "📈上昇" if f_dir == 1 else "📉下落", f"{max(f_prob)*100:.1f}%")
        st.markdown(f'<p class="tech-subtext">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)

st.divider()
st.subheader("📅 重要経済指標")
st.link_button("🌐 GMO外貨 指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
