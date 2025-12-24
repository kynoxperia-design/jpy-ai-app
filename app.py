import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dashboard Ultra", layout="wide") # 比較しやすくワイド画面に

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 8px; min-height: 90px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: bold !important; }
    .time-header { font-size: 1.2rem; font-weight: bold; text-align: center; margin-bottom: 5px; color: #00ff00; border-bottom: 2px solid #00ff00; padding-bottom: 5px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; margin-top: 10px; text-align: center; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    .price-subtext { font-size: 0.8rem; color: #ffffff; text-align: center; margin-top: -5px; background: #262730; border-radius: 5px; padding: 2px; }
    .tech-subtext { font-size: 0.75rem; color: #55aaff; text-align: center; margin-top: 4px; border-top: 1px solid #333; padding-top: 2px; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #00ff00; font-weight: bold; }
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

# 現在レート取得
data_latest = fetch_fx_data("JPY=X", "5d", "1m")
current_price = float(data_latest['Close'].iloc[-1]) if data_latest is not None else 0.0
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 精度特化型：予測エンジン（比較機能強化） ---
def predict_engine_ultra(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data(ticker, period, interval)
    if df is None or len(df) < 50: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # 特徴量エンジニアリング
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / df['Close']
        df['EMA200_Dist'] = (df['Close'] - ta.ema(df['Close'], length=200)) / df['Close']
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去の比較対象レートを特定
        # 日足は当日を含めないよう調整、分足は指定offset分戻る
        idx = -(offset + 1) if is_daily else -offset
        if abs(idx) > len(df): idx = -len(df)
        
        past_price = float(df['Close'].iloc[idx])
        past_row = df.iloc[[idx]]

        # AI学習
        df_train = df.dropna()
        features = ['RSI', 'ADX', 'BB_Width', 'EMA200_Dist']
        X = df_train[features]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        # 予測（最新または過去時点）
        eval_row = df.dropna().tail(1) if offset == 0 else past_row.fillna(method='ffill')
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. メイン表示 ---
st.title("🦅 FX-AI Dashboard Ultra")
st.caption(f"高精度マルチタイムフレーム分析中 | {jst_now.strftime('%H:%M:%S')}")

# メイン特大レート
st.markdown(f"""
    <div style="background-color: #000000; padding: 10px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 15px;">
        <p style="color: #00ff00; margin: 0; font-size: 1rem; letter-spacing: 2px;">USD/JPY LIVE</p>
        <p style="color: #00ff00; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

if st.button('🔄 データを最新に更新（AI再学習）'): st.rerun()

st.divider()

# 各時間軸の設定
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
        
        # --- 実績比較セクション ---
        st.markdown(f'<p class="section-label">これまでの動き</p>', unsafe_allow_html=True)
        p_val, p_dir, _, _ = predict_engine_ultra("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
        
        if p_val > 0:
            diff = current_price - p_val
            st.metric("", "📈 上昇中" if diff > 0 else "📉 下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
        else:
            st.metric("", "取得中", "")

        # --- 最新予測セクション ---
        st.markdown(f'<p class="section-label">AIの最新予測</p>', unsafe_allow_html=True)
        _, f_dir, f_prob, f_rsi = predict_engine_ultra("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
        
        if max(f_prob) < 0.54:
            st.metric("", "⚖️ 中立", "迷い")
        else:
            st.metric("", "📈 上昇" if f_dir == 1 else "📉 下落", f"{max(f_prob)*100:.1f}%")
        
        st.markdown(f'<p class="tech-subtext">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)

st.divider()
st.link_button("🌐 重要経済指標カレンダーを確認", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
