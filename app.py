import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# --- 【ここからCSS設定】 ---
st.set_page_config(page_title="FX-AI Signal", layout="centered")

st.markdown("""
    <style>
    /* 画面全体の背景色と文字色を強制的に固定（ダークモード対策） */
    .stApp {
        background-color: #f0f2f6 !important;
    }
    
    /* 全ての文字色を黒に近い色に固定 */
    h1, h2, h3, p, span, div, label {
        color: #1f1f1f !important;
    }

    /* 指標などのテーブルの中身も見やすく固定 */
    .stTable td, .stTable th {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }

    /* 予測カード（Metric）の背景を白に固定 */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
# --- 【ここまでCSS設定】 ---

# この下に、以前作った「日本時間の取得」や「現在価格の表示」を続けて書きます

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
if st.button('🔄 今すぐ最新データで再計算'):
    st.rerun()

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
# サーバーの時間に9時間を足して日本時間にする
now_jst = datetime.datetime.now() + datetime.timedelta(hours=9)
now = now_jst.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📊 最終更新 (日本時間): {now}")
st.subheader(f"📊 現在時刻: {now} の診断結果")

# --- 現在価格の取得と表示 ---
# 直近の価格データを取得
raw_data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
current_price = raw_data['Close'].iloc[-1]

# MultiIndex（2層構造）になっている場合の対策
if isinstance(current_price, pd.Series):
    current_price = current_price.iloc[0]

# 大きく表示
st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 1.2rem;">現在のドル円 (USD/JPY)</h2>
        <h1 style="color: #00ff00; margin: 0; font-size: 3.5rem;">{current_price:.2f} <span style="font-size: 1.5rem;">円</span></h1>
    </div>
""", unsafe_allow_html=True)
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
# --- 経済指標スケジュール ---
st.subheader("📅 本日の重要経済指標 (日本時間)")

# 指標データをリスト化（ここに今後の注目指標を追記できます）
events = [
    {"時間": "21:30", "重要度": "🔥🔥🔥", "指標名": "米・雇用統計 / CPI"},
    {"時間": "23:00", "重要度": "🔥🔥", "指標名": "米・景気指数 / 住宅販売"},
    {"時間": "24:00", "重要度": "🔥", "指標名": "米・週間石油在庫"},
    {"時間": "04:00", "重要度": "🔥🔥🔥", "指標名": "FOMC政策金利発表"},
]

# テーブル形式で分かりやすく表示
event_df = pd.DataFrame(events)
st.table(event_df)

# 指標直前の注意喚起
now_hour = int(datetime.datetime.now().strftime('%H'))
for e in events:
    event_hour = int(e["時間"].split(":")[0])
    if abs(now_hour - event_hour) <= 1:
        st.warning(f"⚠️ まもなく {e['時間']} に重要指標があります！急変に注意。")

st.caption("※データは自動更新されます。トレードは自己責任でお願いします。")
