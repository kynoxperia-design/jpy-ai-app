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
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    [data-testid="stMetric"] {
        background-color: #1e2128 !important;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. データ取得・時間設定 ---
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
current_time_str = jst_now.strftime('%Y-%m-%d %H:%M')

def get_latest_price():
    try:
        data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 0.0

current_price = get_latest_price()

# --- 3. 共通予測ロジック ---
def predict_at_point(ticker, interval, period, future_steps, offset=0):
    try:
        # 指定された期間のデータを取得
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        df_close = raw['Close']
        if isinstance(df_close, pd.DataFrame): df_close = df_close.iloc[:, 0]
        df = pd.DataFrame({"Price": df_close})
        
        # offsetが指定されている場合、過去の時点までのデータに絞る
        if offset > 0:
            df = df.iloc[:-offset]
            
        df['Ret'] = df['Price'].pct_change()
        df['MA'] = df['Price'].rolling(5).mean()
        df['Dist'] = df['Price'] - df['MA']
        df['Target'] = (df['Price'].shift(-future_steps) > df['Price']).astype(int)
        df = df.dropna()
        
        X = df[['Ret', 'Dist']]
        y = df['Target']
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        pred = model.predict(X.tail(1))[0]
        prob = model.predict_proba(X.tail(1))[0]
        return df['Price'].iloc[-1], pred, prob
    except:
        return 0.0, 0, [0.5, 0.5]

# --- 4. 画面レイアウト ---
st.title("🦅 FX-AI リアルタイム診断")
st.caption(f"最終更新 (日本時間): {current_time_str}")

# 【最上段：現在価格】
st.markdown(f"""
    <div style="background-color: #000000 !important; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 10px; border: 2px solid #00ff00;">
        <p style="color: #00ff00 !important; margin: 0; font-size: 1rem; font-weight: bold;">USD/JPY 現在価格</p>
        <p style="color: #00ff00 !important; margin: 0; font-size: 3.8rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

st.link_button("📈 XE.com リアルタイムチャートを確認", "https://www.xe.com/ja/currencycharts/?from=USD&to=JPY", use_container_width=True)

if st.button('🔄 情報を更新'):
    st.rerun()

# 【中段：過去時点の答え合わせ】
st.divider()
st.subheader("🕰️ 過去時点でのAI予測結果")
st.caption("その時点のデータのみを使ってAIがどう判断していたかを表示します")

# 過去の予測シミュレーション（offsetで過去に遡る）
past_sim = {
    "10分前": predict_at_point("JPY=X", "1m", "1d", 10, offset=10),
    "1時間前": predict_at_point("JPY=X", "5m", "5d", 12, offset=12),
    "4時間前": predict_at_point("JPY=X", "15m", "15d", 16, offset=16),
    "1日前": predict_at_point("JPY=X", "1d", "2y", 1, offset=1)
}

cols1 = st.columns(4)
for i, (label, (p_price, p_dir, p_prob)) in enumerate(past_sim.items()):
    with cols1[i]:
        direction = "📈上昇" if p_dir == 1 else "📉下落"
        st.metric(label, f"{p_price:.2f}", direction)
        st.caption(f"確信度: {max(p_prob)*100:.1f}%")

# 【下段：現在のAI未来予測】
st.divider()
st.subheader("🔮 最新のAI未来予測")

timeframes = {"10分後": ("1m","1d",10), "1時間後": ("5m","5d",12), "4時間後": ("15m","15d",16), "1日後": ("1d","2y",1)}
preds, results = [], []
for label, params in timeframes.items():
    _, p, prob = predict_at_point("JPY=X", params[0], params[1], params[2], offset=0)
    preds.append(p)
    results.append((label, p, prob))

up_ratio = sum(preds) / len(preds)
if up_ratio > 0.7: st.success("🔥 【強い買い】上昇トレンドの可能性が高い")
elif up_ratio < 0.3: st.error("❄️ 【強い売り】下落に注意が必要")
else: st.warning("⚖️ 【様子見】方向感が定まっていません")

cols2 = st.columns(4)
for i, (label, p, prob) in enumerate(results):
    with cols2[i]:
        st.metric(label, "📈 上昇" if p == 1 else "📉 下落", f"{max(prob)*100:.1f}%")

# 【最下段：外部リンク】
st.divider()
st.subheader("📅 経済指標リンク")
st.link_button("🌐 GMO外貨 経済指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
c1, c2 = st.columns(2)
with c1: st.link_button("📊 Yahoo!指標", "https://finance.yahoo.co.jp/fx/center/calendar/", use_container_width=True)
with c2: st.link_button("🔍 みんかぶ指標", "https://fx.minkabu.jp/indicators", use_container_width=True)
