import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dash Pro", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 8px; min-height: 90px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: bold !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
    .time-header { font-size: 1.1rem; font-weight: bold; text-align: center; margin-bottom: 5px; color: #00ff00; border-bottom: 2px solid #333; }
    .section-label { font-size: 0.85rem; color: #aaaaaa; margin-top: 10px; margin-bottom: 2px; text-align: center; }
    .price-subtext { font-size: 0.8rem; color: #888888; text-align: center; margin-top: -5px; margin-bottom: 5px; }
    .prediction-caption { font-size: 0.75rem; color: #cccccc; text-align: center; margin-top: -5px; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. データ取得 ---
def get_latest_price():
    try:
        data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 0.0

current_price = get_latest_price()
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 強化されたAI予測ロジック ---
def predict_at_point(ticker, interval, period, future_steps, offset=0):
    try:
        # 学習用に少し多めにデータを取得
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        df = raw.copy()
        if isinstance(df['Close'], pd.DataFrame): 
            df['Close'] = df['Close'].iloc[:, 0]
            df['Open'] = df['Open'].iloc[:, 0]
            df['High'] = df['High'].iloc[:, 0]
            df['Low'] = df['Low'].iloc[:, 0]

        # 過去時点の再現
        if offset > 0: df = df.iloc[:-offset]

        # --- 特徴量の追加（精度向上の鍵） ---
        # 1. RSI (買われすぎ・売られすぎ)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        # 2. EMA (指数平滑移動平均線)
        df['EMA_diff'] = df['Close'] - ta.ema(df['Close'], length=20)
        # 3. ボリンジャーバンド (ボラティリティ)
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_upper_diff'] = bbands.iloc[:, 2] - df['Close']
        # 4. モメンタム
        df['ROC'] = ta.roc(df['Close'], length=10)

        # 目的変数：future_steps後に価格が上がっているか
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        df = df.dropna()
        
        # 学習用特徴量
        features = ['RSI', 'EMA_diff', 'BB_upper_diff', 'ROC']
        X = df[features]
        y = df['Target']

        # モデルの強化 (決定木を200本に増やし、学習を深化)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        last_price = float(df['Close'].iloc[-1])
        pred = model.predict(X.tail(1))[0]
        prob = model.predict_proba(X.tail(1))[0]
        
        return last_price, pred, prob
    except Exception as e:
        return 0.0, 0, [0.5, 0.5]

# --- 4. メイン表示 ---
st.title("🦅 FX-AI 診断 Pro")
st.caption(f"最終更新: {jst_now.strftime('%H:%M')}")

st.markdown(f"""
    <div style="background-color: #000000; padding: 10px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 10px;">
        <p style="color: #00ff00; margin: 0; font-size: 0.9rem;">USD/JPY リアルタイム</p>
        <p style="color: #00ff00; margin: 0; font-size: 2.8rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

if st.button('🔄 AI再学習・更新'): st.rerun()

st.divider()

timeframes = {
    "10分": {"params": ("1m","1d",10), "offset": 10},
    "1時間": {"params": ("5m","5d",12), "offset": 12},
    "4時間": {"params": ("15m","15d",16), "offset": 16},
    "1日": {"params": ("1d","2y",1), "offset": 1}
}

cols = st.columns(4)

for i, (label, cfg) in enumerate(timeframes.items()):
    with cols[i]:
        st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-label">実績</p>', unsafe_allow_html=True)
        p_val, p_dir, _ = predict_at_point("JPY=X", cfg["params"][0], cfg["params"][1], cfg["params"][2], offset=cfg["offset"])
        diff = current_price - p_val
        status_text = "📈上昇中" if diff > 0 else "📉下落中"
        st.metric("", status_text, f"{diff:+.2f}")
        st.markdown(f'<p class="price-subtext">{p_val:.2f}→{current_price:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-caption">予測:{"上" if p_dir==1 else "下"}</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p class="section-label">AI予測</p>', unsafe_allow_html=True)
        _, f_dir, f_prob = predict_at_point("JPY=X", cfg["params"][0], cfg["params"][1], cfg["params"][2], offset=0)
        st.metric("", "📈上昇" if f_dir == 1 else "📉下落", f"{max(f_prob)*100:.1f}%")

st.divider()
st.subheader("📅 経済指標リンク")
st.link_button("🌐 GMO外貨 指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
