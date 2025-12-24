import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dashboard Pro+", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 8px; min-height: 90px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: bold !important; }
    .time-header { font-size: 1.1rem; font-weight: bold; text-align: center; margin-bottom: 5px; color: #00ff00; border-bottom: 2px solid #333; padding-bottom: 5px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; margin-top: 10px; text-align: center; font-weight: bold; }
    .price-subtext { font-size: 0.75rem; color: #888888; text-align: center; margin-top: -5px; }
    .tech-subtext { font-size: 0.7rem; color: #55aaff; text-align: center; margin-top: 2px; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安定したデータ取得ロジック ---
def fetch_fx_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# 最新価格の取得
data_latest = fetch_fx_data("JPY=X", "1d", "1m")
current_price = float(data_latest['Close'].iloc[-1]) if data_latest is not None else 0.0
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 予測エンジン（1日前対応版） ---
def predict_engine(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data(ticker, period, interval)
    if df is None or len(df) < 20: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # 特徴量計算
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_Sig'] = macd.iloc[:, 2]
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA_Diff'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去実績の算出
        # 日足の場合、最新行は「今日」なので、1日前を取得するには1行前を見る
        if offset > 0:
            past_idx = -(offset + 1) if is_daily else -offset
            if abs(past_idx) > len(df): past_idx = 0
            past_price = float(df['Close'].iloc[past_idx])
            past_row = df.iloc[[past_idx]]
        else:
            past_price = float(df['Close'].iloc[-1])
            past_row = df.tail(1)

        # 学習と予測
        df_train = df.dropna()
        X = df_train[['RSI', 'MACD', 'MACD_Sig', 'ATR', 'MA_Diff']]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        feat_val = past_row[['RSI', 'MACD', 'MACD_Sig', 'ATR', 'MA_Diff']]
        pred = model.predict(feat_val)[0]
        prob = model.predict_proba(feat_val)[0]
        rsi_val = float(past_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. メイン表示 ---
st.title("🦅 FX-AI Dashboard Pro+")
st.caption(f"最終更新: {jst_now.strftime('%H:%M:%S')}")

if current_price == 0:
    st.error("⚠️ レート取得エラー。市場が閉まっているか、API制限の可能性があります。")
else:
    st.markdown(f"""
        <div style="background-color: #000000; padding: 10px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 10px;">
            <p style="color: #00ff00; margin: 0; font-size: 0.9rem;">USD/JPY リアルタイム</p>
            <p style="color: #00ff00; margin: 0; font-size: 2.8rem; font-weight: bold;">{current_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

if st.button('🔄 データを更新'): st.rerun()

st.divider()

# 時間軸設定：1日軸に 'is_daily': True を追加
timeframes = {
    "10分": {"p": ("1d","1m",10), "o": 10, "d": False},
    "1時間": {"p": ("5d","5m",12), "o": 12, "d": False},
    "4時間": {"p": ("7d","15m",16), "o": 16, "d": False},
    "1日": {"p": ("1mo","1d",1), "o": 1, "d": True}
}

cols = st.columns(4)

for i, (label, cfg) in enumerate(timeframes.items()):
    with cols[i]:
        st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
        
        # 実績計算（過去の時点）
        p_val, p_dir, _, _ = predict_engine("JPY=X", cfg["p"][1], cfg["p"][0], cfg["p"][2], offset=cfg["o"], is_daily=cfg["d"])
        
        # 予測計算（現在の時点）
        _, f_dir, f_prob, f_rsi = predict_engine("JPY=X", cfg["p"][1], cfg["p"][0], cfg["p"][2], offset=0, is_daily=cfg["d"])
        
        # 実績表示
        st.markdown(f'<p class="section-label">これまでの動き</p>', unsafe_allow_html=True)
        if p_val > 0:
            diff = current_price - p_val
            st.metric("", "📈上昇中" if diff > 0 else "📉下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
        else:
            st.metric("", "取得中", "")

        # 予測表示
        st.markdown(f'<p class="section-label">最新予測</p>', unsafe_allow_html=True)
        if max(f_prob) < 0.53:
            st.metric("", "⚖️中立", "迷い")
        else:
            st.metric("", "📈上昇" if f_dir == 1 else "📉下落", f"{max(f_prob)*100:.1f}%")
        st.markdown(f'<p class="tech-subtext">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)
