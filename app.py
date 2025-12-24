import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. ページ・デザイン設定 ---
st.set_page_config(page_title="FX-AI Dashboard Ultra", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 10px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: bold !important; color: #00ff00 !important; }
    .time-header { font-size: 1.2rem; font-weight: bold; text-align: center; color: #00ff00; border-bottom: 2px solid #00ff00; padding-bottom: 5px; margin-bottom: 10px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; text-align: center; font-weight: bold; margin-top: 10px; text-transform: uppercase; }
    .price-subtext { font-size: 0.85rem; color: #ffffff; text-align: center; background: #262730; border-radius: 5px; padding: 4px; margin-top: 5px; border: 1px solid #444; }
    .tech-subtext { font-size: 0.75rem; color: #55aaff; text-align: center; margin-top: 6px; border-top: 1px solid #333; padding-top: 4px; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 安定データ取得エンジン ---
@st.cache_data(ttl=60)
def fetch_fx_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty: return None
        
        # MultiIndex(二重列名)の解消
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 列名のクレンジング
        df.columns = [str(col).strip() for col in df.columns]
        target_cols = ['Open', 'High', 'Low', 'Close']
        df = df[target_cols].copy()
        
        # 数値型へ強制変換
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
    except:
        return None

# 現在のリアルタイムレートを取得
data_main = fetch_fx_data("JPY=X", "5d", "1m")
current_price = float(data_main['Close'].iloc[-1]) if data_main is not None else 0.0
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)

# --- 3. 高精度AI予測エンジン（マルチ指標版） ---
def predict_engine_full(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data(ticker, period, interval)
    if df is None or len(df) < 50: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        # 指標計算（RSI, MACD, ADX, EMA乖離）
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0]
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']
        df['EMA200'] = ta.ema(df['Close'], length=min(200, len(df)-1))
        df['EMA_Dist'] = (df['Close'] - df['EMA200']) / df['Close']
        
        # 未来の判定（Target）
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去価格の特定
        idx = -(offset + 1) if is_daily else -offset
        if abs(idx) > len(df): idx = -1
        past_price = float(df['Close'].iloc[idx])
        past_row = df.iloc[[idx]]

        # AI学習
        df_train = df.dropna()
        features = ['RSI', 'MACD', 'ADX', 'EMA_Dist']
        X = df_train[features]
        y = df_train['Target']
        
        # 学習モデルの構築（300本の決定木で高密度学習）
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        # 最新の予測実行
        eval_row = df.dropna().tail(1) if offset == 0 else past_row.fillna(method='ffill')
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. 画面表示メイン ---
st.title("🦅 FX-AI Dashboard Ultra")
st.caption(f"全時間軸・高精度予測モデル | 更新: {jst_now.strftime('%H:%M:%S')} (JST)")

# メイン特大レート
if current_price > 0:
    st.markdown(f"""
        <div style="background-color: #000000; padding: 15px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 20px;">
            <p style="color: #00ff00; margin: 0; font-size: 1rem; letter-spacing: 2px;">USD/JPY リアルタイム</p>
            <p style="color: #00ff00; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("データの取得に失敗しました。市場が閉まっていないか確認してください。")

if st.button('🔄 市場データを再学習して更新'): st.rerun()

st.divider()

# 各時間軸の設定
# 1日軸（1日）は1日前を確実にとるために期間を2年(2y)に設定
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
        
        # 過去レートと最新予測の計算
        p_val, _, _, _ = predict_engine_full("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
        _, f_dir, f_prob, f_rsi = predict_engine_full("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
        
        # 【実績比較】表示
        st.markdown('<p class="section-label">これまでの動き</p>', unsafe_allow_html=True)
        if p_val > 0:
            diff = current_price - p_val
            st.metric("", "📈 上昇中" if diff > 0 else "📉 下落中", f"{diff:+.2f}")
            st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
        else:
            st.metric("", "取得中", "")
        
        # 【AI予測】表示
        st.markdown('<p class="section-label">AI最新予測</p>', unsafe_allow_html=True)
        # 判定の厳格化（確信度53%未満は中立）
        if max(f_prob) < 0.53:
            st.metric("", "⚖️ 中立", "迷い")
        else:
            st.metric("", "📈 上昇" if f_dir == 1 else "📉 下落", f"{max(f_prob)*100:.1f}%")
        
        st.markdown(f'<p class="tech-subtext">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)

st.divider()
st.link_button("🌐 重要経済指標カレンダーを確認", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)
