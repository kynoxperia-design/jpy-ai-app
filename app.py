import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. デザイン設定 ---
st.set_page_config(page_title="FX-AI Dashboard Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 10px; text-align: center; }
    .time-header { font-size: 1.2rem; font-weight: bold; text-align: center; color: #00ff00; border-bottom: 2px solid #00ff00; margin-bottom: 10px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; text-align: center; font-weight: bold; }
    .price-subtext { font-size: 0.8rem; color: #ffffff; text-align: center; background: #262730; border-radius: 5px; padding: 3px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安定データ取得 ---
def fetch_fx_data(ticker, period, interval):
    try:
        # 修正：列名の階層を壊さないよう取得
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        
        # MultiIndex(2重構造)を解消
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 重複列の削除と必要列の確保
        df = df.loc[:, ~df.columns.duplicated()]
        return df[['Open', 'High', 'Low', 'Close']].copy()
    except:
        return None

# --- 3. 予測ロジック（エラー回避重視） ---
def predict_logic(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data(ticker, period, interval)
    if df is None or len(df) < 30: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # 指標計算
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0]
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA_Diff'] = (df['Close'] - df['MA20'])
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去価格
        idx = -(offset + 1) if is_daily else -offset
        past_price = float(df['Close'].iloc[idx])
        
        # 学習
        df_train = df.dropna()
        features = ['RSI', 'MACD', 'MA_Diff']
        X = df_train[features]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        # 予測
        eval_row = df.dropna().tail(1) if offset == 0 else df.iloc[[idx]].ffill()
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. 画面表示 ---
st.title("🦅 FX-AI Dashboard Lite")
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
st.caption(f"最終取得: {jst_now.strftime('%H:%M:%S')}")

# メインレートの取得と表示
data_main = fetch_fx_data("JPY=X", "2d", "1m")
if data_main is not None:
    current_price = float(data_main['Close'].iloc[-1])
    st.markdown(f"""
        <div style="background-color: #000000; padding: 15px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 20px;">
            <p style="color: #00ff00; margin: 0; font-size: 1rem;">USD/JPY リアルタイム</p>
            <p style="color: #00ff00; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("現在、Yahoo Financeからデータを取得できません。1分後に再読み込みしてください。")
    current_price = 0.0

if st.button('🔄 画面を更新する'): st.rerun()

st.divider()

if current_price > 0:
    timeframes = {
        "10分": {"p": ("2d","1m",10), "o": 10, "d": False},
        "1時間": {"p": ("7d","5m",12), "o": 12, "d": False},
        "4時間": {"p": ("14d","15m",16), "o": 16, "d": False},
        "1日": {"p": ("1y","1d",1), "o": 1, "d": True}
    }
    cols = st.columns(4)
    for i, (label, cfg) in enumerate(timeframes.items()):
        with cols[i]:
            st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
            p_val, _, _, _ = predict_logic("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
            _, f_dir, f_prob, f_rsi = predict_logic("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
            
            st.markdown('<p class="section-label">実績</p>', unsafe_allow_html=True)
            diff = current_price - p_val if p_val > 0 else 0
            st.metric("", "📈上昇中" if diff > 0 else "📉下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
            
            st.markdown('<p class="section-label">AI予測</p>', unsafe_allow_html=True)
            st.metric("", "📈上昇" if f_dir == 1 else "📉下落", f"{max(f_prob)*100:.1f}%")
            st.markdown(f'<p style="color:#55aaff; font-size:0.7rem; text-align:center;">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)
