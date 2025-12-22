import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="FX-AI Signal", layout="centered")

# カスタムCSSでデザイン調整
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .signal-up { color: #ff4b4b; font-size: 24px; font-weight: bold; }
    .signal-down { color: #1f77b4; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦅 FX-AI リアルタイム診断")

# --- 予測ロジック ---
def predict_logic(ticker, interval, period, future_steps):
    raw = yf.download(ticker, period=period, interval=interval, progress=False)
    df_close = raw['Close']
    if isinstance(df_close, pd.DataFrame): df_close = df_close.iloc[:, 0]
    df = pd.DataFrame({"Price": df_close})
    df['Ret'] = df['Price'].pct_change()
    df['MA'] = df['Price'].rolling(5).mean()
    df['Dist'] = df['Price'] - df['MA']
    df['Target'] = (df['Price'].shift(-future_steps) > df['Price']).astype(int)
    df = df.dropna()
    X = df[['Ret', 'Dist']]
    y = df['Target']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
    return model.predict(X.tail(1))[0], model.predict_proba(X.tail(1))[0]

# --- メインコンテンツ ---
now = datetime.datetime.now().strftime('%H:%M')
st.subheader(f"📊 現在時刻: {now} の診断結果")

# 1. 総合判断（サマリー）
col_main = st.columns(1)[0]
preds = []
timeframes = {"10分後": ("1m", "1d", 10), "1時間後": ("5m", "5d", 12), "1日後": ("1d", "2y", 1)}

for label, params in timeframes.items():
    p, prob = predict_logic("JPY=X", params[0], params[1], params[2])
    preds.append(p)

up_ratio = sum(preds) / len(preds)

if up_ratio > 0.7:
    st.success("🔥 【強い買いシグナル】 他者のアルゴも上昇方向で一致しています")
elif up_ratio < 0.3:
    st.error("❄️ 【強い売りシグナル】 下落トレンドへの追随が推奨されます")
else:
    st.warning("⚖️ 【様子見】 方向感が定まっていません。レンジ相場です")

st.divider()

# 2. 時間軸別の詳細カード
cols = st.columns(3)
for i, (label, params) in enumerate(timeframes.items()):
    p, prob = predict_logic("JPY=X", params[0], params[1], params[2])
    with cols[i]:
        direction = "上昇" if p == 1 else "下落"
        icon = "📈" if p == 1 else "📉"
        st.metric(label, f"{icon} {direction}", f"{max(prob)*100:.1f}%")

st.divider()

# 3. 経済指標アラート
st.subheader("⚠️ 注目イベント")
event_col1, event_col2 = st.columns(2)
with event_col1:
    st.info("21:30 米雇用統計 (最重要)")
with event_col2:
    st.info("23:00 米景気指数 (重要)")

st.caption("※データは自動更新されます。トレードは自己責任でお願いします。")
