import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import warnings
from datetime import datetime, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN VIBE CODING (BLOOMBERG TERMINAL THEME)
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Entropy Terminal", page_icon="📈")

def _t(vi, en):
    return en if st.session_state.get('lang', 'Tiếng Việt') == 'English' else vi

st.markdown("""
<style>
    .reportview-container { background: #000000; color: #00FF41; }
    .stApp { background-color: #0E1117; color: #C0C0C0; }
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New', Courier, monospace; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #ffffff; }
    .metric-label { color: #aaaaaa; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA ENGINE
# ==============================================================================

@st.cache_data(ttl=3600)
def fetch_market_data(ticker="VNINDEX", start_date="2020-01-01", end_date=None):
    from vnstock import Vnstock
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.history(start=start_date, end=end_date)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def fetch_vn30_components(start_date="2020-01-01", end_date=None):
    vn30_tickers = [
        "ACB.VN", "BCM.VN", "BID.VN", "BVH.VN", "CTG.VN", "FPT.VN", 
        "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN", "MBB.VN", "MSN.VN", 
        "MWG.VN", "PLX.VN", "POW.VN", "SAB.VN", "SHB.VN", "STB.VN", 
        "TCB.VN", "TPB.VN", "VCB.VN", "VHM.VN", "VIB.VN", "VIC.VN", 
        "VJC.VN", "VNM.VN", "VPB.VN", "VRE.VN"
    ]
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(vn30_tickers, start=start_date, end=end_date)['Close']
    return df.ffill().pct_change().dropna(how='all')

# ==============================================================================
# PHYSICS ENGINE: WPE & STATISTICAL COMPLEXITY
# ==============================================================================

def get_ordinal_patterns_wpe(x, m, tau=1):
    N = len(x)
    n_patterns = N - (m - 1) * tau
    if n_patterns <= 0: return [], []
    cols = [x[i*tau : i*tau + n_patterns] for i in range(m)]
    matrix = np.vstack(cols).T 
    
    patterns = np.argsort(matrix, axis=1)
    weights = np.var(matrix, axis=1)
    return patterns, weights

def compute_wpe_complexity(x, m=3, tau=1):
    patterns, weights = get_ordinal_patterns_wpe(x, m, tau)
    if len(patterns) == 0: return np.nan, np.nan
    
    hashable = [tuple(p) for p in patterns]
    w_dict = {}
    for h, w in zip(hashable, weights):
        w_dict[h] = w_dict.get(h, 0) + w
        
    total_w = sum(w_dict.values())
    if total_w <= 0: return np.nan, np.nan
        
    p_dist = np.array(list(w_dict.values())) / total_w
    N_states = math.factorial(m)
    
    P = p_dist[p_dist > 0]
    S_P = -np.sum(P * np.log(P))
    H_P = S_P / np.log(N_states)
    
    U = 1.0 / N_states
    P_mid_present = (P + U) / 2.0
    num_missing = N_states - len(P)
    
    S_mid = -np.sum(P_mid_present * np.log(P_mid_present)) 
    if num_missing > 0:
        P_mid_missing = U / 2.0
        S_mid -= num_missing * (P_mid_missing * np.log(P_mid_missing))
        
    S_U = np.log(N_states)
    JSD = S_mid - 0.5 * S_P - 0.5 * S_U
    
    P_star_mid_0 = (1.0 + U) / 2.0
    P_star_mid_rest = U / 2.0
    S_star_mid = -( P_star_mid_0 * np.log(P_star_mid_0) ) - (N_states - 1) * ( P_star_mid_rest * np.log(P_star_mid_rest) )
    D_max = S_star_mid - 0.5 * S_U 
    Q0 = 1.0 / D_max if D_max > 0 else 0
    
    C_JS = Q0 * JSD * H_P
    return H_P, C_JS

def calculate_advanced_entropy(df, m=3, tau=1, window=22):
    np.seterr(divide='ignore', invalid='ignore')
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).values
    vol_ret = np.log(df['Volume'] / df['Volume'].shift(1)).values
    
    n = len(df)
    wpe_p = np.full(n, np.nan); c_p = np.full(n, np.nan)
    wpe_v = np.full(n, np.nan); c_v = np.full(n, np.nan)
    
    for i in range(window, n):
        p_slice = log_ret[i-window:i]
        p_valid = p_slice[np.isfinite(p_slice)]
        if len(p_valid) >= m:
            h, c = compute_wpe_complexity(p_valid, m, tau)
            wpe_p[i] = h; c_p[i] = c
            
        v_slice = vol_ret[i-window:i]
        v_valid = v_slice[np.isfinite(v_slice)]
        if len(v_valid) >= m:
            h, c = compute_wpe_complexity(v_valid, m, tau)
            wpe_v[i] = h; c_v[i] = c
            
    df['WPE_Price_Daily'] = wpe_p
    df['Complexity_Price'] = c_p
    df['MFI_Price'] = wpe_p * (1 - c_p)
    
    df['WPE_Volume_Daily'] = wpe_v
    df['Complexity_Volume'] = c_v
    df['MFI_Volume'] = wpe_v * (1 - c_v)

    # Weekly
    ret_w_series = np.log(df['Close'].resample('W').last() / df['Close'].resample('W').last().shift(1))
    w_vals = ret_w_series.values
    n_w = len(w_vals)
    wpe_w = np.full(n_w, np.nan)
    for i in range(12, n_w):
        valid = w_vals[i-12:i]
        valid = valid[np.isfinite(valid)]
        if len(valid) >= m:
            wpe_w[i], _ = compute_wpe_complexity(valid, m, tau)
    df['WPE_Price_Weekly'] = pd.Series(wpe_w, index=ret_w_series.index).reindex(df.index, method='ffill')
    
    # Monthly
    ret_m_series = np.log(df['Close'].resample('M').last() / df['Close'].resample('M').last().shift(1))
    m_vals = ret_m_series.values
    n_m = len(m_vals)
    wpe_m = np.full(n_m, np.nan)
    for i in range(6, n_m):
        valid = m_vals[i-6:i]
        valid = valid[np.isfinite(valid)]
        if len(valid) >= m:
            wpe_m[i], _ = compute_wpe_complexity(valid, m, tau)
    df['WPE_Price_Monthly'] = pd.Series(wpe_m, index=ret_m_series.index).reindex(df.index, method='ffill')
    
    return df

@st.cache_data
def generate_ce_boundary(m):
    N = math.factorial(m)
    U = 1.0 / N
    S_U = np.log(N)
    
    P_star_mid_0 = (1.0 + U) / 2.0
    P_star_mid_rest = U / 2.0
    S_star_mid = -(P_star_mid_0 * np.log(P_star_mid_0)) - (N - 1) * (P_star_mid_rest * np.log(P_star_mid_rest))
    D_max = S_star_mid - 0.5 * S_U
    Q0 = 1.0 / D_max if D_max > 0 else 0
    
    def calc_H_C(P_dist):
        P = P_dist[P_dist > 0]
        if len(P) == 0: return 0.0, 0.0
        S_P = -np.sum(P * np.log(P))
        H_P = S_P / S_U
        P_mid = (P_dist + U) / 2.0
        S_mid = -np.sum(P_mid * np.log(P_mid))
        JSD = S_mid - 0.5 * S_P - 0.5 * S_U
        C_JS = Q0 * JSD * H_P
        return H_P, C_JS

    H_max_curve, C_max_curve = [], []
    for p_max in np.linspace(1/N, 1.0, 200):
        P_dist = np.full(N, (1.0 - p_max) / (N - 1))
        P_dist[0] = p_max
        h, c = calc_H_C(P_dist)
        H_max_curve.append(h)
        C_max_curve.append(c)
        
    H_min_curve, C_min_curve = [], []
    for k in range(1, N):
        for p in np.linspace(1.0/k, 1.0/(k+1), 50):
            P_dist = np.zeros(N)
            P_dist[:k] = p
            P_dist[k] = 1.0 - k*p
            h, c = calc_H_C(P_dist)
            H_min_curve.append(h)
            C_min_curve.append(c)
            
    h, c = calc_H_C(np.full(N, 1.0/N))
    H_min_curve.append(h)
    C_min_curve.append(c)

    return H_max_curve, C_max_curve, H_min_curve, C_min_curve

def calculate_correlation_entropy(df_returns, window=22):
    n_days = len(df_returns)
    corr_entropy = pd.Series(index=df_returns.index, dtype='float64')
    
    for i in range(window, n_days):
        window_rets = df_returns.iloc[i-window:i]
        corr_matrix = window_rets.corr().values
        corr_matrix = np.nan_to_num(corr_matrix)
        
        eigenvalues, _ = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        total_variance = np.sum(eigenvalues)
        if total_variance == 0: continue
            
        p_i = eigenvalues / total_variance
        p_i = p_i[p_i > 0]
        
        max_entropy = np.log(len(eigenvalues))
        corr_entropy.iloc[i] = (-np.sum(p_i * np.log(p_i)) / max_entropy) * 100
        
    return corr_entropy

def detect_market_regime(df):
    df['Regime'] = 'Neutral'
    df['Color'] = 'rgba(128, 128, 128, 0.2)'
    df['Norm_Entropy'] = df['WPE_Price_Daily'] * 100  
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Entropy_Diff'] = df['Norm_Entropy'].diff().rolling(3).mean()
    
    for i in range(len(df)):
        c = df['Close'].iloc[i]
        ma = df['MA20'].iloc[i]
        ent = df['MFI_Price'].iloc[i] * 100
        ent_d = df['Entropy_Diff'].iloc[i]
        
        if pd.isna(ent) or pd.isna(ma):
            continue
            
        if c >= ma and ent < 50:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Stable Growth'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(0, 255, 65, 0.8)'
        elif c >= ma and ent >= 50:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Fragile Growth'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(255, 215, 0, 0.8)'
        elif c < ma and ent > 65:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Chaos/Panic'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(255, 0, 0, 0.8)'
        elif c < ma and ent_d < 0:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Structural Recomposition'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(138, 43, 226, 0.8)'

    return df

def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal_len=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_len, adjust=False).mean()
    return macd, signal

# ==============================================================================
# GIAO DIỆN (STREAMLIT DASHBOARD)
# ==============================================================================
def main():
    st.sidebar.markdown("### 🌐 " + _t("Ngôn ngữ", "Language"))
    st.session_state['lang'] = st.sidebar.radio("", ["Tiếng Việt", "English"], horizontal=True, label_visibility="collapsed")
    
    st.markdown("<h1>" + _t("SYSTEM ARCHITECT: ĐỨT GÃY CẤU TRÚC", "SYSTEM ARCHITECT: STRUCTURAL ENTROPY & REGIME DETECTION") + "</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888;'>" + _t("Khám phá sự đứt gãy cấu trúc trong hành vi thị trường thông qua lăng kính Vật lý Hệ thống Phức tạp.", "Discovering Structural Breaks in market behavior through the lens of Complex Systems Physics.") + "</p>", unsafe_allow_html=True)
    
    st.sidebar.header(_t("⚙️ Cấu hình Dữ liệu", "⚙️ Data Configuration"))
    data_source = st.sidebar.radio(_t("Nguồn Dữ liệu:", "Data Engine:"), [_t("API Đám mây (vnstock/yfinance)", "API Cloud (vnstock/yfinance)"), _t("Tải lên Local File", "Upload Local File")])
    
    start_date = st.sidebar.date_input(_t("Ngày Bắt đầu", "Start Date"), datetime(2020, 1, 1))
    end_date = st.sidebar.date_input(_t("Ngày Kết thúc", "End Date"), datetime.now())
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧮 " + _t("Tham số Vật lý", "Physics Parameters"))
    embed_dim = st.sidebar.slider(_t("Kích thước Nhúng (m)", "Embedded Dimension (m)"), 3, 7, 3)
    rolling_window = st.sidebar.slider(_t("Cửa sổ Dữ liệu (days)", "Rolling Window (days)"), 20, 252, 22)

    vni_df = None
    vn30_returns = None
    
    if data_source == _t("API Đám mây (vnstock/yfinance)", "API Cloud (vnstock/yfinance)"):
        with st.spinner(_t('Đang xây dựng Kiến trúc... (Lấy Dữ liệu API)', 'Building System Architectures... (Fetching Market Data API)')):
            try:
                s_date = start_date.strftime('%Y-%m-%d')
                e_date = end_date.strftime('%Y-%m-%d')
                vni_df = fetch_market_data(ticker="VNINDEX", start_date=s_date, end_date=e_date)
                vn30_returns = fetch_vn30_components(start_date=s_date, end_date=e_date)
            except Exception as e:
                st.error(_t("❌ Lỗi truy cập API. (Có thể do giới hạn IP/chặn Cloud).", "❌ API Access Error. (Possible IP limit or Cloud block)."))
                st.info(_t("💡 Hãy chuyển sang tính năng 'Tải lên Local File' trên thanh Sidebar để tiếp tục.", "💡 Please switch to 'Upload Local File' on the Sidebar to continue."))
                return
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 " + _t("Tải lên Excel/CSV", "Upload Your Data"))
        uploaded_file = st.sidebar.file_uploader(_t("Tải Dữ liệu (.csv / .xlsx)", "Upload Data (.csv / .xlsx)"), type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    vni_df = pd.read_csv(uploaded_file)
                else:
                    vni_df = pd.read_excel(uploaded_file)
                
                date_cols = [c for c in vni_df.columns if str(c).lower().strip() in ['date', 'time', 'ngày']]
                if len(date_cols) > 0:
                    vni_df['Date_Index'] = pd.to_datetime(vni_df[date_cols[0]])
                    vni_df.set_index('Date_Index', inplace=True)
                    vni_df.sort_index(inplace=True)
                
                col_mapping = {}
                for c in vni_df.columns:
                    c_low = str(c).lower().strip()
                    if c_low == 'open': col_mapping[c] = 'Open'
                    elif c_low == 'high': col_mapping[c] = 'High'
                    elif c_low == 'low': col_mapping[c] = 'Low'
                    elif c_low == 'close': col_mapping[c] = 'Close'
                    elif c_low == 'volume': col_mapping[c] = 'Volume'
                
                vni_df.rename(columns=col_mapping, inplace=True)
                
                start_ts = pd.to_datetime(start_date)
                end_ts = pd.to_datetime(end_date)
                if vni_df.index.tzinfo is not None:
                    start_ts = start_ts.tz_localize(vni_df.index.tzinfo)
                    end_ts = end_ts.tz_localize(vni_df.index.tzinfo)
                vni_df = vni_df.loc[start_ts:end_ts]
            except Exception as e:
                st.error(f"❌ Upload Error: {e}")
                return
        else:
            st.info(_t("⬅️ Vui lòng upload File dữ liệu ở Sidebar bên trái.", "⬅️ Please upload data file on the Left Sidebar."))
            return

    if vni_df is not None and not vni_df.empty:
        with st.spinner(_t('Đang tính toán Bánh răng Vật lý (WPE, MFI & Causality)...', 'Calculating Physics Engines (WPE, MFI & Causality)...')):
            vni_df = calculate_advanced_entropy(vni_df, m=embed_dim, tau=1, window=rolling_window)
            vni_df = detect_market_regime(vni_df)
            
            vni_df['RSI'] = calculate_rsi(vni_df['Close'])
            vni_df['MACD'], vni_df['MACD_Signal'] = calculate_macd(vni_df['Close'])
            
            if vn30_returns is not None:
                sys_entropy = calculate_correlation_entropy(vn30_returns, window=rolling_window)
                vni_df['System_Entropy'] = sys_entropy.reindex(vni_df.index).ffill()
            else:
                vni_df['System_Entropy'] = np.nan
        
    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    last_idx = vni_df.dropna(subset=['Close', 'Regime', 'MFI_Price']).index[-1]
    curr_data = vni_df.loc[last_idx]
    
    with col1:
        st.markdown(f"<div class='metric-label'>VN-INDEX</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{curr_data['Close']:.2f}</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"<div class='metric-label'>{'System Market Regime' if _t('V','') == 'E' else 'Trạng thái Cấu trúc'}</div>", unsafe_allow_html=True)
        regime_color = curr_data['Color'].replace('0.8', '1.0') 
        st.markdown(f"<div class='metric-value' style='color:{regime_color}'>{curr_data['Regime']}</div>", unsafe_allow_html=True)

    with col3:
        sys_ent_val = curr_data['System_Entropy']
        st.markdown(f"<div class='metric-label'>{'Cross-Sectional Entropy' if _t('V','') == 'E' else 'Entropy Phân luồng'}</div>", unsafe_allow_html=True)
        if pd.isna(sys_ent_val):
            st.markdown(f"<div class='metric-value'>N/A</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-value'>{sys_ent_val:.1f} / 100</div>", unsafe_allow_html=True)

    with col4:
        mfi = curr_data['MFI_Price'] * 100
        st.markdown(f"<div class='metric-label'>{'Market Fragility Index (MFI)' if _t('V','') == 'E' else 'Chỉ số Phân rã (MFI)'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{mfi:.1f} / 100</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- MAIN CHART: MẢNG MÀU THỂ HIỆN REGIME ---
    st.subheader(_t("1. Cấu trúc Trạng thái VN-Index (MFI-Driven)", "1. VN-Index Structural Regimes (MFI-Driven)"))
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(
        x=vni_df.index, open=vni_df['Open'], high=vni_df['High'],
        low=vni_df['Low'], close=vni_df['Close'], name='VN-Index',
        increasing_line_color='#00FF41', decreasing_line_color='#FF0000'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df['MA20'], line=dict(color='white', width=1, dash='dash'), name='MA20'
    ), row=1, col=1)

    regime_colors = {
        'Stable Growth': 'rgba(0, 255, 65, 0.8)',
        'Fragile Growth': 'rgba(255, 215, 0, 0.8)',
        'Chaos/Panic': 'rgba(255, 0, 0, 0.8)',
        'Structural Recomposition': 'rgba(138, 43, 226, 0.8)'
    }
    
    for regime, color in regime_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='square', size=8, color=color), name=regime
        ), row=1, col=1)

    vni_df['Regime_Shift'] = vni_df['Regime'] != vni_df['Regime'].shift(1)
    shift_indices = vni_df.index[vni_df['Regime_Shift']].tolist()
    if len(shift_indices) > 0 and shift_indices[0] != vni_df.index[0]:
        shift_indices.insert(0, vni_df.index[0])
    if len(shift_indices) == 0:
        shift_indices = [vni_df.index[0]]
    shift_indices.append(vni_df.index[-1])

    for i in range(len(shift_indices) - 1):
        start_idx = shift_indices[i]
        end_idx = shift_indices[i+1]
        regime = vni_df.loc[start_idx, 'Regime']
        color = regime_colors.get(regime)
        if color:
            fig.add_shape(
                type="rect", x0=start_idx, x1=end_idx, y0=0.97, y1=1.0,
                yref="y domain", fillcolor=color, line_width=0, layer="above", row=1, col=1
            )

    # WPE Lines
    fig.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df['WPE_Price_Daily'] * 100, 
        line=dict(color='#00FF41', width=2), name='WPE Price'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df['WPE_Volume_Daily'] * 100, 
        line=dict(color='orange', width=1, dash='dot'), name='WPE Vol'
    ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark", height=700, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='#0E1117', paper_bgcolor='#000000',
        xaxis_rangeslider_visible=False, dragmode='pan',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, fixedrange=False, side="right")
    fig.update_yaxes(title_text="WPE (0-100)", row=2, col=1, fixedrange=False, side="right")
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
    
    st.info(_t(
        "**Vật lý của Trạng thái Động lượng (Theo MFI):**\n"
        "- 🟩 **Stable Growth**: Giá tăng, dòng tiền có kiến trúc trật tự.\n"
        "- 🟨 **Fragile Growth**: Giá tăng định danh nhưng MFI đạt đỉnh -> Kiến trúc gãy, rủi ro sụp đổ cao.\n"
        "- 🟥 **Chaos / Panic**: Chu kỳ sụp đổ cấu trúc (White Noise).\n"
        "- 🟪 **Structural Recomposition**: Hệ thống thoát khỏi vùng hỗn loạn và bước vào pha bình ổn cục bộ. Quá trình này đại diện cho sự sắp xếp lại các cấu phần thị trường chứ chưa hẳn là một đáy giá.",
        "**Physics of Market Regimes (MFI-Driven):**\n"
        "- 🟩 **Stable Growth**: Prices trace above MAs in a structurally ordered regime.\n"
        "- 🟨 **Fragile Growth**: Prices rise nominally, but MFI balloons meaning structural breakdown -> Extreme collapse risk.\n"
        "- 🟥 **Chaos / Panic**: Mathematical execution of the price breakdown (White Noise).\n"
        "- 🟪 **Structural Recomposition**: The system exits the chaos zone and enters a phase of local stability. This represents a re-ordering of market components rather than a confirmed price bottom."
    ))

    st.markdown("---")

    # --- CROSS-SECTIONAL ENTROPY & MULTI-SCALE ---
    st.subheader(_t("2. Phân tích Cấu trúc (MSE & VN30)", "2. MSE & VN30 Components Analysis"))
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown(_t("**Đa khung thời gian WPE - Khử Nhiễu Lagging**", "**Multi-scale WPE (Price) - Noise Reduction**"))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['WPE_Price_Daily'], name='Daily WPE', line=dict(color='rgba(255, 255, 255, 0.4)')))
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['WPE_Price_Weekly'], name='Weekly WPE', line=dict(color='rgba(0, 255, 65, 0.8)', width=2)))
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['WPE_Price_Monthly'], name='Monthly WPE', line=dict(color='rgba(255, 0, 255, 0.8)', width=3)))
        fig2.update_layout(dragmode='pan', template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#000000', margin=dict(l=0, r=0, t=30, b=0), height=350)
        fig2.update_yaxes(fixedrange=False, side="right")
        st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        st.caption(_t("WPE Daily nhiễu mạnh nhưng Weekly/Monthly thấp -> Biến động chỉ là hỗn loạn ngắn hạn, không ảnh hưởng cấu trúc.","High short-term WPE Noise combined with stable structural WPE implies non-threatening short term fluctuations."))

    with colB:
        st.markdown(_t("**Phân luồng Cấu trúc (EVD VN30)**", "**Cross-Sectional Entropy (Eigenvalue Decomposition VN30)**"))
        if vni_df['System_Entropy'].isna().all():
            st.info(_t("⚠️ Không có dữ liệu rổ VN30 (Đang dùng Upload Data). Biểu đồ bị ẩn.", "⚠️ VN30 data unavailable in Upload mode. Correlation chart disabled."))
        else:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=vni_df.index, y=vni_df['System_Entropy'], fill='tozeroy', name='System Entropy', line=dict(color='#00BFFF')))
            fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Chaos Threshold")
            fig3.add_hline(y=40, line_dash="dash", line_color="#00FF41", annotation_text="Consensus Threshold")
            fig3.update_layout(dragmode='pan', template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#000000', margin=dict(l=0, r=0, t=30, b=0), height=350)
            fig3.update_yaxes(fixedrange=False, side="right")
            st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
            st.caption(_t("Mức độ phân mảnh đồng thuận của dòng tiền vào 30 cổ phiếu trụ.", "Capital consensus fragmentation via Correlation eigenvalues."))

    st.markdown("---")

    # --- CECP ---
    st.subheader(_t("3. Mặt phẳng Nhân quả Cấu trúc - Hỗn loạn (CECP)", "3. Complexity-Entropy Causality Plane (CECP)"))
    col_ce, col_mfi = st.columns([2, 1])
    with col_ce:
        st.markdown(_t("Phân ranh giới **López-Ruiz**: VN-Index đang dịch chuyển về ngẫu nhiên hay có cấu trúc (Structural).", "**López-Ruiz Boundaries**: Determining VN-Index shifts between randomness (White noise) and deterministic structures."))
        H_max, C_max, H_min, C_min = generate_ce_boundary(embed_dim)
        
        fig_ce = go.Figure()
        fig_ce.add_trace(go.Scatter(x=H_max, y=C_max, mode='lines', line=dict(color='gray', dash='dash'), name='Upper Bound', hoverinfo='none'))
        fig_ce.add_trace(go.Scatter(x=H_min, y=C_min, mode='lines', line=dict(color='gray', dash='dash'), name='Lower Bound', hoverinfo='none'))
        
        N_traj = min(120, len(vni_df.dropna(subset=['WPE_Price_Daily', 'Complexity_Price'])))
        last_n = vni_df.dropna(subset=['WPE_Price_Daily', 'Complexity_Price']).tail(N_traj)
        
        colors = ["rgba({}, 255, {}, {})".format(int(255*(1 - i/N_traj)), int(255*(i/N_traj)), 0.2 + 0.8*(i/N_traj)) for i in range(len(last_n))]
        sizes = [4 + 12*(i/N_traj) for i in range(len(last_n))]
        
        fig_ce.add_trace(go.Scatter(
            x=last_n['WPE_Price_Daily'], y=last_n['Complexity_Price'],
            mode='markers+lines',
            marker=dict(color=colors, size=sizes, line=dict(width=1, color='white')),
            line=dict(color='rgba(255,255,255,0.1)', width=1),
            name='VNI Trajectory',
            text=last_n.index.strftime('%Y-%m-%d'),
            hoverinfo='text+x+y'
        ))
        
        fig_ce.add_trace(go.Scatter(
            x=[last_n['WPE_Price_Daily'].iloc[-1]], y=[last_n['Complexity_Price'].iloc[-1]],
            mode='markers', marker=dict(color='red', size=16, symbol='star'), name='Present Market'
        ))
        
        fig_ce.update_layout(template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#000000', margin=dict(l=0, r=0, t=30, b=0), dragmode='pan')
        fig_ce.update_xaxes(title="Permutation Entropy (H)", range=[0.2, 1.0])
        fig_ce.update_yaxes(title="Statistical Complexity (C)")
        st.plotly_chart(fig_ce, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
    with col_mfi:
        st.markdown(_t("**Cơ cấu Fragility Index & Chẩn đoán CECP**", "**Fragility Index & CECP Diagnostics**"))
        curr_mfi = last_n['MFI_Price'].iloc[-1]
        prev_mfi = last_n['MFI_Price'].iloc[-2]
        curr_h = last_n['WPE_Price_Daily'].iloc[-1]
        prev_h = last_n['WPE_Price_Daily'].iloc[-2]
        curr_c = last_n['Complexity_Price'].iloc[-1]
        
        st.metric("MFI (Price)", f"{curr_mfi:.4f}", delta=f"{curr_mfi - prev_mfi:.4f}", delta_color="inverse")
        st.metric("Statistical Complexity", f"{curr_c:.4f}", delta=f"{curr_c - last_n['Complexity_Price'].iloc[-2]:.4f}")
        
        st.markdown("---")
        
        if curr_h >= 0.9 and curr_c <= 0.05:
            state_name = _t("🔴 Chaos/Crash (Hỗn loạn/Sụp đổ)", "🔴 Chaos/Crash")
            state_prompt = _t(
                "Thị trường đang ghi nhận mức Entropy biến động giá đạt cực đại ($H \\approx 1.0$) và Complexity biến mất ($C \\approx 0$). Điểm Present Market nằm ở góc dưới cùng bên phải của đồ thị CECP. Hãy mô tả trạng thái hỗn loạn này dưới góc nhìn của sự hoảng loạn bầy đàn (Herding Behavior). Nhà đầu tư nên làm gì khi hệ thống hoàn toàn mất đi tính tự tổ chức?",
                "The market is recording maximum price fluctuation Entropy ($H \\approx 1.0$) and vanished Complexity ($C \\approx 0$). The Present Market point resides at the bottom-right corner of the CECP plot. Describe this chaotic state from the perspective of Herding Behavior. What should investors do when the system completely loses its self-organization?"
            )
        elif curr_h > 0.8 and curr_c < 0.1:
            state_name = _t("🟡 Fragile Growth (Tăng trưởng dễ vỡ)", "🟡 Fragile Growth")
            state_prompt = _t(
                "Dựa trên dữ liệu VN-Index, chỉ số MFI đang ở mức cao (>0.8) với Complexity ($C$) cực thấp và Entropy ($H$) tiệm cận vùng Chaos. Điểm trên đồ thị CECP đang bám sát đường Lower Bound. Hãy phân tích trạng thái này dưới góc độ dòng tiền nóng và rủi ro sụp đổ cấu trúc. Tại sao mức tăng điểm hiện tại lại được coi là thiếu bền vững và dễ vỡ trước các cú sốc thông tin?",
                "Based on VN-Index data, the MFI is exceptionally high (>0.8) while Complexity ($C$) is extremely low and Entropy ($H$) is approaching the Chaos zone. The point on the CECP graph is clinging to the Lower Bound. Analyze this state from the perspective of 'hot money' and structural collapse risk. Why is the current price surge considered unsustainable and fragile to information shocks?"
            )
        elif 0.6 <= curr_h <= 0.75 and curr_h < prev_h and curr_c <= 0.15:
            state_name = _t("👽 Dead Cat Bounce (Hồi phục thiếu NL)", "👽 Dead Cat Bounce")
            state_prompt = _t(
                "VN-Index vừa trải qua nhịp giảm mạnh và đang có dấu hiệu hồi phục kỹ thuật. Chỉ số Entropy ($H$) đang giảm dần nhưng Complexity ($C$) không có sự bứt phá đáng kể, điểm vẫn nằm ở vùng biên dưới của mặt phẳng CECP. Với giả định khối lượng giao dịch (Volume) đang suy yếu như một 'quả bóng tennis mất năng lượng', hãy đưa ra nhận định về khả năng đây chỉ là một bẫy hồi phục ngắn hạn thay vì một sự khởi đầu của xu hướng Structural Growth.",
                "VN-Index has just gone through a sharp decline and shows signs of technical recovery. Entropy ($H$) is gradually decreasing but Complexity ($C$) has no significant breakthrough, the point remains in the lower boundary region of the CECP plane. Assuming trading volume is weakening like an 'energy-depleted tennis ball', evaluate the probability of this being just a short-term recovery trap rather than the genesis of a Structural Growth trend."
            )
        elif 0.4 <= curr_h <= 0.6 and curr_c > 0.15:
            state_name = _t("🟢 Structural Growth (Bền vững)", "🟢 Structural Growth")
            state_prompt = _t(
                "Chỉ số MFI của VN-Index đang giảm về vùng an toàn (<0.5) nhờ sự gia tăng đáng kể của Statistical Complexity ($C$). Điểm trên đồ thị CECP đã rời xa Lower Bound và di chuyển về vùng trung tâm của các hệ thống phức tạp. Hãy giải thích tại sao trạng thái này cho thấy thị trường đang có sự đồng thuận về cấu trúc, dòng tiền thông minh bắt đầu dẫn dắt và các quy luật kỹ thuật có độ tin cậy cao hơn.",
                "The VN-Index's MFI is returning to a safe zone (<0.5) thanks to a significant increase in Statistical Complexity ($C$). The point on the CECP graph has departed from the Lower Bound towards the center of complex systems. Explain why this indicates that the market has structural consensus, smart money is taking the lead, and technical rules boast higher reliability."
            )
        else:
            state_name = _t("⚪ Transitioning (Chuyển pha)", "⚪ Transitioning Phase")
            state_prompt = _t(
                "Thị trường đang nằm ở vùng đệm của mặt phẳng CECP, Entropy và Complexity đang giằng co để xác lập cấu trúc dòng tiền mới. Cần quan sát thêm quỹ đạo (Trajectory) để xác định rõ xu hướng.",
                "The market lies in the buffer zone of the CECP plane. Entropy and Complexity are wrestling to establish a new capital flow structure. Further observation of the Trajectory is needed to define a clear trend."
            )
            
        st.markdown(f"**{_t('Trạng thái CECP hiện hành:', 'Current CECP State:')}**<br> <span style='color: white; font-size: 1.1rem'>{state_name}</span>", unsafe_allow_html=True)
        st.info(f"💡 **AI Prompt Context:**\n\n_{state_prompt}_")
        
    st.markdown("---")
    
    # --- COMPARISON LAYER: ENTROPY VS TRADITIONAL LAG INDICATORS ---
    st.subheader(_t("4. Chỉ báo Vật lý vs Chỉ báo Độ trễ (RSI/MACD)", "4. Comparison Layer: Fragility vs Traditional (RSI & MACD)"))
    st.markdown(_t("Độ tương quan giữa MFI thời thời gian thực và sự phản ứng chậm chạp của RSI/MACD.", "Evaluating Market Fragility Index lead signal vs lagging signals."))
    
    fig_comp = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.33, 0.33, 0.33])
    
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['RSI'], name='RSI (14)', line=dict(color='yellow')), row=1, col=1)
    fig_comp.add_hline(y=70, row=1, col=1, line_dash="dash", line_color="gray")
    fig_comp.add_hline(y=30, row=1, col=1, line_dash="dash", line_color="gray")
    
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['MACD_Signal'], name='Signal', line=dict(color='magenta')), row=2, col=1)
    
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['MFI_Price'] * 100, name='MFI Price Dynamics', line=dict(color='#00FF41')), row=3, col=1)
    fig_comp.add_hline(y=65, row=3, col=1, line_dash="dash", line_color="red", annotation_text="Tín hiệu đứt gãy" if _t("V","") == "" else "Breakdown Signal")
    
    fig_comp.update_layout(dragmode='pan', template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#000000', height=600, margin=dict(l=20, r=20, t=20, b=20))
    fig_comp.update_yaxes(title_text="RSI", row=1, col=1, fixedrange=False, side="right")
    fig_comp.update_yaxes(title_text="MACD", row=2, col=1, fixedrange=False, side="right")
    fig_comp.update_yaxes(title_text="MFI", row=3, col=1, fixedrange=False, side="right")
    
    st.plotly_chart(fig_comp, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
    
    st.download_button(
        label=_t("Tải xuống DataFrame (.csv)", "Download Full Physics DataFrame (.csv)"),
        data=vni_df.to_csv().encode('utf-8'),
        file_name='vni_entropy_physics.csv',
        mime='text/csv'
    )
    
    st.markdown("<p style='text-align:center; color:#888'>Terminal Framework by Systems Physics Quant. Code with 💚 (Vibe Coding).</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
