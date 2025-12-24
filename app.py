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
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: bold !important; color: #00ff00 !important; }
    .time-header { font-size: 1.2rem; font-weight: bold; text-align: center; color: #00ff00; border-bottom: 2px solid #00ff00; padding-bottom: 5px; margin-bottom: 10px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; margin-top: 10px; text-align: center; font-weight: bold; }
    .price-subtext { font-size: 0.8rem; color: #ffffff; text-align: center; background: #262730; border-radius: 5px; padding: 3px; margin-top: 5px; }
    .tech-subtext { font-size: 0.75rem; color: #55aaff; text-align: center; margin-top: 4px; border-top: 1px solid #333; padding-top: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 究極の安定データ取得関数 ---
@st.cache_data(ttl=30) # 30秒間キャッシュしてエラーを防ぐ
def fetch_fx_data_safe(ticker, period, interval):
    try:
        # auto_adjustをFalseにして確実な列名を取得
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty or len(df) < 5:
            return None
        
        # 【重要】MultiIndex(2層構造)を平坦化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 重複する列を削除し、必要な列だけ抽出
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        
        # 欠損値補完
        df = df.ffill().dropna()
        return df
    except:
        return None

# 現在のUSD/JPYリアルタイム価格
data_now = fetch_fx_data_safe("JPY=X", "1d", "1m")
current_price = float(data_now['Close'].iloc[-1]) if data_now is not None else 0.0
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 予測エンジン ---
def predict_engine(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data_safe(ticker, period, interval)
    if df is None or len(df) < 30:
        return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # 指標計算
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_S'] = macd.iloc[:, 2]
        df['EMA200'] = ta.ema(df['Close'], length=min(200, len(df)-1))
        df['EMA_Dist'] = (df['Close'] - df['EMA200']) / df['Close']
        
        # ターゲット（未来の判定）
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 実績の比較用レート特定
        idx = -(offset + 1) if is_daily else -offset
        past_price = float(df['Close'].iloc[idx])
        past_row = df.iloc[[idx]]

        # 学習
        df_train = df.dropna()
        features = ['RSI', 'MACD', 'MACD_S', 'EMA_Dist']
        X = df_train[features]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        # 最新の予測
        eval_row = df.dropna().tail(1) if offset == 0 else past_row.fillna(method='ffill')
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. 画面表示 ---
st.title("🦅 FX-AI Dashboard Pro")
st.caption(f"最終更新 (JST): {jst_now.strftime('%H:%M:%S')}")

if current_price == 0:
    st.error("⚠️ レートが取得できません。市場閉場中か、Yahoo Finance側の制限の可能性があります。数分後に再度お試しください。")
else:
    st.markdown(f"""
        <div style="background-color: #000000; padding: 15px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 20px;">
            <p style="color: #00ff00; margin: 0; font-size: 1rem; letter-spacing: 1px;">USD/JPY リアルタイム</p>
            <p style="color: #00ff00; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

if st.button('🔄 情報を更新'): st.rerun()

st.divider()

timeframes = {
    "10分": {"p": ("2d","1m",10), "o": 10, "d": False},
    "1時間": {"p": ("7d","5m",12), "o": 12, "d": False},
    "4時間": {"p": ("30d","15m",16), "o": 16, "d": False},
    "1日": {"p": ("2y","1d",1), "o": 1, "d": True}
}

cols = st.columns(4)

for i, (label, cfg) in enumerate(timeframes.items()):
    with cols[i]:
        st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
        
        # 過去価格と最新予測を取得
        p_val, _, _, _ = predict_engine("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
        _, f_dir, f_prob, f_rsi = predict_engine("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
        
        # --- 実績 ---
        st.markdown(f'<p class="section-label">実績</p>', unsafe_allow_html=True)
        if p_val > 0:
            diff = current_price - p_val
            st.metric("", "📈 上昇中" if diff > 0 else "📉 下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
        else:
            st.metric("", "取得中", "")

        # --- AI予測 ---
        st.markdown(f'<p class="section-label">AI予測</p>', unsafe_allow_html=True)
        if max(f_prob) < 0.53:
            st.metric("", "⚖️ 中立", "迷い")
        else:
            st.metric("", "📈 上昇" if f_dir == 1 else "📉 下落", f"{max(f_prob)*100:.1f}%")
        st.markdown(f'<p class="tech-subtext">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)

st.divider()
st.link_button("📅 経済指標カレンダーを確認", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
