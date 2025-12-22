import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定（ダークモードで統一） ---
st.set_page_config(page_title="FX-AI Dashboard", layout="centered")

st.markdown("""
    <style>
    /* 全体を黒背景、文字を白に固定 */
    .stApp { background-color: #0e1117 !important; }
    
    /* あらゆる場所の文字を白にする */
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    
    /* 予測カードの背景を濃いグレー、文字を白に */
    [data-testid="stMetric"] {
        background-color: #1e2128 !important;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }

    /* テーブルの設定 */
    .stTable { background-color: #1e2128 !important; color: #ffffff !important; }
    .stTable td, .stTable th { color: #ffffff !important; border-bottom: 1px solid #333 !important; }
    
    /* ボタンの文字色 */
    .stButton>button { color: #ffffff !important; border: 1px solid #444; background-color: #262730; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 日本時間と現在価格の取得 ---
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
current_time_str = jst_now.strftime('%Y-%m-%d %H:%M')

st.title("🦅 FX-AI リアルタイム診断")
st.caption(f"最終更新 (日本時間): {current_time_str}")

if st.button('🔄 データを更新'):
    st.rerun()

# 価格取得
raw_data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
current_price = raw_data['Close'].iloc[-1]
if isinstance(current_price, pd.Series):
    current_price = current_price.iloc[0]

# 現在価格表示
st.markdown(f"""
    <div style="background-color: #000000 !important; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; border: 1px solid #00ff00;">
        <p style="color: #00ff00 !important; margin: 0; font-size: 1rem;">USD/JPY 現在価格</p>
        <p style="color: #00ff00 !important; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

# --- 3. 予測ロジック ---
def predict_logic(ticker, interval, period, future_steps):
    try:
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
    except:
        return 0, [0.5, 0.5]

# 診断とカード表示
timeframes = {"10分後": ("1m", "1d", 10), "1時間後": ("5m", "5d", 12), "1日後": ("1d", "2y", 1)}
preds, results = [], []
for label, params in timeframes.items():
    p, prob = predict_logic("JPY=X", params[0], params[1], params[2])
    preds.append(p)
    results.append((label, p, prob))

up_ratio = sum(preds) / len(preds)
if up_ratio > 0.7:
    st.success("🔥 【強い買い】")
elif up_ratio < 0.3:
    st.error("❄️ 【強い売り】")
else:
    st.warning("⚖️ 【様子見】")

cols = st.columns(3)
for i, (label, p, prob) in enumerate(results):
    with cols[i]:
        st.metric(label, "📈 上昇" if p == 1 else "📉 下落", f"{max(prob)*100:.1f}%")

# --- 4. 経済指標 (カレンダーへのリンク) ---
st.divider()
st.subheader("📅 経済指標を確認")

st.info("信頼できる外部サイトで最新のスケジュールをチェックしましょう。")

# ボタンを配置
st.link_button("🌐 GMO外貨 経済指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)

col_link1, col_link2 = st.columns(2)
with col_link1:
    st.link_button("📊 Yahoo!指標", "https://finance.yahoo.co.jp/fx/center/calendar/", use_container_width=True)
with col_link2:
    st.link_button("🔍 みんかぶ指標", "https://fx.minkabu.jp/indicators", use_container_width=True)

st.caption("※GMO外貨は重要度や通貨別の絞り込みがしやすくおすすめです。")
