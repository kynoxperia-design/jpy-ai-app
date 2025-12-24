import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 1. ページ設定 ---
st.set_page_config(page_title="FX-AI Dashboard Ultra", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] { background-color: #1e2128 !important; border: 1px solid #333; border-radius: 10px; padding: 10px; text-align: center; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: bold !important; color: #00ff00 !important; }
    .time-header { font-size: 1.2rem; font-weight: bold; text-align: center; color: #00ff00; border-bottom: 2px solid #00ff00; padding-bottom: 5px; margin-bottom: 10px; }
    .section-label { font-size: 0.8rem; color: #aaaaaa; text-align: center; font-weight: bold; margin-top: 10px; }
    .price-subtext { font-size: 0.85rem; color: #ffffff; text-align: center; background: #262730; border-radius: 5px; padding: 4px; margin-top: 5px; border: 1px solid #444; }
    .stButton>button { width: 100%; color: #ffffff !important; background-color: #262730; border: 1px solid #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. データ取得ロジック（最新yfinance対応） ---
def fetch_fx_data_fixed(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close', 'Open', 'High', 'Low']].copy()
        df = df.apply(pd.to_numeric, errors='coerce').ffill()
        return df
    except:
        return None

# --- 3. 予測・比較エンジン ---
def predict_engine_full(ticker, interval, period, future_steps, offset=0, is_daily=False):
    df = fetch_fx_data_fixed(ticker, period, interval)
    if df is None or len(df) < 20: return 0.0, 0, [0.5, 0.5], 50.0
    
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd.iloc[:, 0]
        df['EMA200'] = ta.ema(df['Close'], length=min(200, len(df)-1))
        df['EMA_Dist'] = (df['Close'] - df['EMA200']) / df['Close']
        df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
        
        # 過去レート取得
        idx = -2 if is_daily else -offset
        if abs(idx) > len(df): idx = -1
        past_price = float(df['Close'].iloc[idx])
        
        # 学習
        df_train = df.dropna()
        features = ['RSI', 'MACD', 'EMA_Dist']
        X = df_train[features]
        y = df_train['Target']
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X.iloc[:-future_steps], y.iloc[:-future_steps])
        
        eval_row = df.dropna().tail(1)
        pred = model.predict(eval_row[features])[0]
        prob = model.predict_proba(eval_row[features])[0]
        rsi_val = float(eval_row['RSI'].iloc[0])
        
        return past_price, pred, prob, rsi_val
    except:
        return 0.0, 0, [0.5, 0.5], 50.0

# --- 4. 画面表示 ---
st.title("🦅 FX-AI Dashboard Ultra")
jst_now = datetime.datetime.now() + datetime.timedelta(hours=9)
st.caption(f"最終更新: {jst_now.strftime('%H:%M:%S')} (JST)")

# メインレート表示
data_main = fetch_fx_data_fixed("JPY=X", "2d", "1m")
if data_main is not None:
    current_price = float(data_main['Close'].iloc[-1])
    st.markdown(f"""
        <div style="background-color: #000000; padding: 15px; border-radius: 15px; text-align: center; border: 2px solid #00ff00; margin-bottom: 20px;">
            <p style="color: #00ff00; margin: 0; font-size: 1rem; letter-spacing: 2px;">USD/JPY リアルタイム</p>
            <p style="color: #00ff00; margin: 0; font-size: 3.5rem; font-weight: bold;">{current_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("レート取得中...")
    current_price = 0.0

if st.button('🔄 データを更新・再学習'): st.rerun()

st.divider()

# 各時間軸の比較
if current_price > 0:
    timeframes = {
        "10分": {"p": ("2d","1m",10), "o": 10, "d": False},
        "1時間": {"p": ("7d","5m",12), "o": 12, "d": False},
        "4時間": {"p": ("14d","15m",16), "o": 16, "d": False},
        "1日": {"p": ("2y","1d",1), "o": 1, "d": True}
    }
    
    cols = st.columns(4)
    for i, (label, cfg) in enumerate(timeframes.items()):
        with cols[i]:
            st.markdown(f'<p class="time-header">{label}軸</p>', unsafe_allow_html=True)
            p_val, _, _, _ = predict_engine_full("JPY=X", *cfg["p"], offset=cfg["o"], is_daily=cfg["d"])
            _, f_dir, f_prob, f_rsi = predict_engine_full("JPY=X", *cfg["p"], offset=0, is_daily=cfg["d"])
            
            st.markdown('<p class="section-label">これまでの動き</p>', unsafe_allow_html=True)
            if p_val > 0:
                diff = current_price - p_val
                st.metric("", "📈 上昇" if diff > 0 else "📉 下落", f"{diff:+.2f}")
                st.markdown(f'<p class="price-subtext">{p_val:.2f} → {current_price:.2f}</p>', unsafe_allow_html=True)
            
            st.markdown('<p class="section-label">AI予測</p>', unsafe_allow_html=True)
            st.metric("", "🚀 上昇" if f_dir == 1 else "💧 下落", f"{max(f_prob)*100:.1f}%")
            st.markdown(f'<p style="color:#55aaff; font-size:0.75rem; text-align:center;">RSI: {f_rsi:.1f}</p>', unsafe_allow_html=True)

st.divider()

# --- 5. 外部リンク集（復活） ---
st.subheader("🔗 トレーダー用リンク集")
link_cols = st.columns(3)

with link_cols[0]:
    st.markdown("### 📅 経済指標")
    st.link_button("🌐 みんかぶ 経済指標カレンダー", "https://fx.minkabu.jp/indicators", use_container_width=True)
    st.link_button("🌐 GMO外貨 指標カレンダー", "https://www.gaikaex.com/gaikaex/mark/calendar/", use_container_width=True)

with link_cols[1]:
    st.markdown("### 📰 ニュース・速報")
    st.link_button("📺 Yahoo!ファイナンス(FXニュース)", "https://finance.yahoo.co.jp/news/fx", use_container_width=True)
    st.link_button("📺 ロイター 為替情報", "https://jp.reuters.com/markets/currencies", use_container_width=True)

with link_cols[2]:
    st.markdown("### 📊 詳細チャート")
    st.link_button("📈 TradingView (USDJPY)", "https://jp.tradingview.com/symbols/USDJPY/", use_container_width=True)
    st.link_button("📈 OANDA オーダーブック", "https://www.oanda.jp/labo/oandachart/", use_container_width=True)

st.info("※ AIの予測はあくまで計算結果であり、将来の利益を保証するものではありません。投資は自己責任でお願いいたします。")
