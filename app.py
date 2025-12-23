import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dash", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    
    /* カードの高さとデザインを統一 */
    [data-testid="stMetric"] {
        background-color: #1e2128 !important;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
        min-height: 110px;
    }
    
    .time-header {
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        color: #00ff00;
        border-bottom: 1px solid #00ff00;
    }

    .section-label {
        font-size: 0.75rem;
        color: #aaaaaa;
        margin-top: 10px;
        margin-bottom: 5px;
        text-align: center;
        font-weight: bold;
    }
    
    .price-subtext {
        font-size: 0.8rem;
        color: #dddddd;
        text-align: center;
        margin: 0;
    }
    
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

# --- 3. 共通予測ロジック ---
def predict_at_point(ticker, interval, period, future_steps, offset=0):
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        df_close = raw['Close']
        if isinstance(df_close, pd.DataFrame): df_close = df_close.iloc[:, 0]
        df = pd.DataFrame({"Price": df_close})
        if offset > 0: df = df.iloc[:-offset]
        df['Ret'] = df['Price'].pct_change()
        df['MA'] = df['Price'].rolling(5).mean()
        df['Dist'] = df['Price'] - df['MA']
        df['Target'] = (df['Price'].shift(-future_steps) > df['Price']).astype(int)
        df = df.dropna()
        X = df[['Ret', 'Dist']]
        y = df['Target']
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        return float(df['Price'].iloc[-1]), model.predict(X.tail(1))[0], model.predict_proba(X.tail(1))[0]
    except: return 0.0, 0, [0.5, 0.5]

# --- 4. 画面表示 ---
st.title("🦅 FX-AI 診断パネル")
st.caption(f"最終更新: {jst_now.strftime('%H:%M')}")

# メイン現在価格表示
st.markdown(f"""
    <div style="background-color: #000000; padding: 15px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 10px;">
        <p style="color: #00ff00; margin: 0; font-size: 1rem;">USD/JPY リアルタイム価格</p>
        <p style="color: #00ff00; margin: 0; font-size: 3.2rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

st.link_button("📈 XE.com チャートを確認", "https://www.xe.com/ja/currencycharts/?from=USD&to=JPY", use_container_width=True)
if st.button('🔄 情報を更新'): st.rerun()

st.divider()

# 【メインレイアウト：4つの時間軸】
timeframes = {
    "10分軸": {"params": ("1m","1d",10), "offset": 10, "past_label": "10分前"},
    "1時間軸": {"params": ("5m","5d",12), "offset": 12, "past_label": "1時間前"},
    "4時間軸": {"params": ("15m","15d",16), "offset": 16, "past_label": "4時間前"},
    "1日軸": {"params": ("1d","2y",1), "offset": 1, "past_label": "1日前"}
}

cols = st.columns(4)

for i, (label, cfg) in enumerate(timeframes.items()):
    with cols[i]:
        st.markdown(f'<p class="time-header">{label}</p>', unsafe_allow_html=True)
        
        # --- 過去実績セクション ---
        st.markdown(f'<p class="section-label">過去実績と現在</p>', unsafe_allow_html=True)
        p_val, p_dir, _ = predict_at_point("JPY=X", cfg["params"][0], cfg["params"][1], cfg["params"][2], offset=cfg["offset"])
        diff = current_price - p_val
        
        # 現在値と過去値を併記するためのカスタム表示
        st.metric("", f"現:{current_price:.2f}", f"{diff:+.2f}")
        st.markdown(f'<p class="price-subtext">始点: {p_val:.2f}</p>', unsafe_allow_html=True)
        st.caption("📈上昇予測済" if p_dir == 1 else "📉下落予測済")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- 現在予測セクション ---
        st.markdown(f'<p class="section-label">現在のAI予測</p>', unsafe_allow_html=True)
        _, f_dir, f_prob = predict_at_point("JPY=X", cfg["params"][0], cfg["params"][1], cfg["params"][2], offset=0)
        st.metric("", "📈上昇" if f_dir == 1 else "📉下落", f"{max(f_prob)*100:.1f}%")

# --- 5. 外部リンク ---
st.divider()
st.subheader("📅 経済指標リンク")
st.link_button("🌐 GMO外貨 指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
c1, c2 = st.columns(2)
with c1: st.link_button("📊 Yahoo!", "https://finance.yahoo.co.jp/fx/center/calendar/", use_container_width=True)
with c2: st.link_button("🔍 みんかぶ", "https://fx.minkabu.jp/indicators", use_container_width=True)
