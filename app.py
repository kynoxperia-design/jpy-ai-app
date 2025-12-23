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
    .stTable { background-color: #1e2128 !important; }
    .stTable td, .stTable th { color: #ffffff !important; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. データ取得・時間設定 ---
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
current_time_str = jst_now.strftime('%Y-%m-%d %H:%M')

# 現在価格の取得（リトライ機能付き）
def get_latest_price():
    try:
        data = yf.download("JPY=X", period="1d", interval="1m", progress=False)
        return data['Close'].iloc[-1]
    except:
        return 0.0

current_price = get_latest_price()

# --- 3. 画面レイアウト ---
st.title("🦅 FX-AI リアルタイム診断")
st.caption(f"最終更新 (日本時間): {current_time_str}")

# 【最上段：現在価格】
st.markdown(f"""
    <div style="background-color: #000000 !important; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 10px; border: 2px solid #00ff00;">
        <p style="color: #00ff00 !important; margin: 0; font-size: 1rem; font-weight: bold;">USD/JPY 現在価格</p>
        <p style="color: #00ff00 !important; margin: 0; font-size: 3.8rem; font-weight: bold;">{current_price:.2f}</p>
    </div>
""", unsafe_allow_html=True)

st.link_button("📈 XE.com リアルタイムチャートを確認", 
               "https://www.xe.com/ja/currencycharts/?from=USD&to=JPY", 
               use_container_width=True)

if st.button('🔄 情報を更新'):
    st.rerun()

# 【中段：過去の振り返り】
st.divider()
st.subheader("🕰️ 過去レートと比較 (勢いの確認)")

def get_past_price_v2(period, interval):
    try:
        # 少し長めにデータを取って、その一番古いデータを「過去」とする
        p_data = yf.download("JPY=X", period=period, interval=interval, progress=False)
        if len(p_data) > 0:
            return p_data['Close'].iloc[0] # 期間内の最初の価格
        return current_price
    except:
        return current_price

# 正確な比較のために期間を調整
past_list = {
    "10分前": get_past_price_v2("30m", "1m"), # 30分間のデータの最初 = 約30分前
    "1時間前": get_past_price_v2("2d", "1h"), # 2日間の1時間足の最初 = 約1日前になってしまうのを防ぐため細かく調整
    "4時間前": get_past_price_v2("5d", "1h"),
    "1日前": get_past_price_v2("5d", "1d")
}

# 1時間前と4時間前をより正確にするための再調整
past_list["1時間前"] = get_past_price_v2("2h", "5m") 
past_list["4時間前"] = get_past_price_v2("8h", "15m")

cols1 = st.columns(4)
for i, (label, p_val) in enumerate(past_list.items()):
    # 取得した値がSeriesだった場合の対策
    display_p = float(p_val)
    diff = current_price - display_p
    with cols1[i]:
        st.metric(label, f"{display_p:.2f}", f"{diff:+.2f}")

# 【下段：AI未来予測】
st.divider()
st.subheader("🔮 AI未来予測 (これからの診断)")

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
    except: return 0, [0.5, 0.5]

timeframes = {"10分後": ("1m","1d",10), "1時間後": ("5m","5d",12), "4時間後": ("15m","15d",16), "1日後": ("1d","2y",1)}
preds, results = [], []
for label, params in timeframes.items():
    p, prob = predict_logic("JPY=X", params[0], params[1], params[2])
    preds.append(p)
    results.append((label, p, prob))

# 総合判断
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
