import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定（ダークモード対策済） ---
st.set_page_config(page_title="FX-AI Dashboard", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6 !important; }
    h1, h2, h3, p, span, label { color: #1f1f1f !important; }
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTable td, .stTable th { color: #1f1f1f !important; background-color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 日本時間と現在価格の取得 ---
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
current_time_str = jst_now.strftime('%Y-%m-%d %H:%M')

st.title("🦅 FX-AI リアルタイム診断")
st.caption(f"最終更新 (日本時間): {current_time_str}")

# 更新ボタン
if st.button('🔄 データを更新'):
    st.rerun()

# 価格取得
raw_data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
current_price = raw_data['Close'].iloc[-1]
if isinstance(current_price, pd.Series):
    current_price = current_price.iloc[0]

# 【重要】現在価格表示（ダークモードでも絶対見える設定）
st.markdown(f"""
    <div style="background-color: #1a1a1a !important; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
        <p style="color: #aaaaaa !important; margin: 0; font-size: 1rem;">現在のドル円 (USD/JPY)</p>
        <p style="color: #00ff00 !important; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f} <span style="font-size: 1.5rem;">円</span></p>
    </div>
""", unsafe_allow_html=True)

# --- 3. 予測ロジック ---
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

# 診断実行
timeframes = {"10分後": ("1m", "1d", 10), "1時間後": ("5m", "5d", 12), "1日後": ("1d", "2y", 1)}
preds = []
results = []

for label, params in timeframes.items():
    p, prob = predict_logic("JPY=X", params[0], params[1], params[2])
    preds.append(p)
    results.append((label, p, prob))

# 総合判断表示
up_ratio = sum(preds) / len(preds)
if up_ratio > 0.7:
    st.success("🔥 【強い買い】 上昇トレンドの可能性が高いです")
elif up_ratio < 0.3:
    st.error("❄️ 【強い売り】 下落に警戒が必要です")
else:
    st.warning("⚖️ 【様子見】 方向感が定まっていません")

# 各時間軸のカード
cols = st.columns(3)
for i, (label, p, prob) in enumerate(results):
    with cols[i]:
        direction = "上昇" if p == 1 else "下落"
        icon = "📈" if p == 1 else "📉"
        st.metric(label, f"{icon} {direction}", f"{max(prob)*100:.1f}%")

# --- 4. 経済指標スケジュール ---
st.divider()
st.subheader("📅 本日の重要指標 (日本時間)")
events = [
    {"時間": "21:30", "重要度": "🔥🔥🔥", "指標名": "米・雇用統計 / CPI"},
    {"時間": "23:00", "重要度": "🔥🔥", "指標名": "米・景気指数"},
    {"時間": "04:00", "重要度": "🔥🔥🔥", "指標名": "FOMC政策金利"},
]
st.table(pd.DataFrame(events))

# 指標アラート
current_hour = jst_now.hour
for e in events:
    event_hour = int(e["時間"].split(":")[0])
    if 0 <= (event_hour - current_hour) <= 1:
        st.warning(f"⚠️ まもなく {e['時間']} に重要指標があります！")
