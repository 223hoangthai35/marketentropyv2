import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import warnings
from datetime import datetime, timedelta
import yfinance as yf

# Ignore warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN VIBE CODING (BLOOMBERG TERMINAL THEME)
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Entropy Terminal", page_icon="📈")

st.markdown("""
<style>
    .reportview-container {
        background: #000000;
        color: #00FF41;
    }
    .stApp {
        background-color: #0E1117;
        color: #C0C0C0;
    }
    h1, h2, h3 {
        color: #00FF41 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        color: #aaaaaa;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# QUANTS ENGINE: STRUCTURAL ENTROPY & REGIME DETECTION
# ==============================================================================

@st.cache_data(ttl=3600)
def fetch_market_data(ticker="VNINDEX", period="2y"):
    """Lấy dữ liệu vĩ mô hoặc chỉ số (Vnstock)"""
    from vnstock import Vnstock
    end_date = datetime.now().strftime('%Y-%m-%d')
    days = 365
    if period == "2y":
        days = 2 * 365
    elif period == "3y":
        days = 3 * 365
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.history(start=start_date, end=end_date)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def fetch_vn30_components():
    """Lấy dữ liệu 30 mã cổ phiếu lớn bằng YF để tính ma trận tương quan"""
    vn30_tickers = [
        "ACB.VN", "BCM.VN", "BID.VN", "BVH.VN", "CTG.VN", "FPT.VN", 
        "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN", "MBB.VN", "MSN.VN", 
        "MWG.VN", "PLX.VN", "POW.VN", "SAB.VN", "SHB.VN", "STB.VN", 
        "TCB.VN", "TPB.VN", "VCB.VN", "VHM.VN", "VIB.VN", "VIC.VN", 
        "VJC.VN", "VNM.VN", "VPB.VN", "VRE.VN"
    ]
    df = yf.download(vn30_tickers, period="2y")['Close']
    return df.ffill().pct_change().dropna(how='all')

def shannon_entropy(time_series, window=20, bins=10):
    """
    Tính Shannon Entropy trên một khung thời gian cụ thể (Daily, Weekly...).
    Ý nghĩa: Đo mức độ phân mảnh của phân phối lợi suất.
    Nhiều nhiễu -> phân phối đều -> Entropy cao.
    """
    entropy_vals = np.zeros(len(time_series))
    entropy_vals[:] = np.nan
    
    for i in range(window, len(time_series)):
        window_data = time_series.iloc[i-window:i].dropna()
        if len(window_data) < 2 or window_data.std() == 0:
            entropy_vals[i] = 0
            continue
        # Histogram để phân cụm xác suất
        hist, _ = np.histogram(window_data, bins=bins, density=False)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy_vals[i] = -np.sum(prob * np.log2(prob))
        
    return pd.Series(entropy_vals, index=time_series.index)

def calculate_mse(df):
    """
    Multi-scale Entropy (MSE) - Khử nhiễu & Lagging
    Tạo 3 khung thời gian (Daily, Weekly, Monthly)
    """
    # Tính lợi suất Daily
    ret_d = df['Close'].pct_change()
    
    # Resample sang khung Weekly và Monthly
    ret_w = df['Close'].resample('W').last().pct_change()
    ret_m = df['Close'].resample('M').last().pct_change()
    
    # Tính Entropy
    ent_d = shannon_entropy(ret_d, window=22) # Khoảng 1 tháng giao dịch
    ent_w = shannon_entropy(ret_w, window=12) # Khoảng 1 quý (12 tuần)
    ent_m = shannon_entropy(ret_m, window=6)  # Nửa năm
    
    df['Entropy_Daily'] = ent_d
    # Căn chỉnh lại chuỗi thời gian cho Weekly và Monthly về Daily index để dễ vẽ chart
    df['Entropy_Weekly'] = ent_w.reindex(df.index, method='ffill')
    df['Entropy_Monthly'] = ent_m.reindex(df.index, method='ffill')
    
    return df

def calculate_correlation_entropy(df_returns, window=22):
    """
    Correlation Entropy: Dùng Phân rã Trị riêng (EVD) trên Ma trận Tương quan.
    Ý nghĩa: Sự đồng thuận của VN30 cao -> Ít trị riêng nổi bật -> Entropy thấp.
    Nếu phân hóa mạnh -> Các trị riêng san đều -> Entropy đạt cực đại.
    """
    n_days = len(df_returns)
    corr_entropy = pd.Series(index=df_returns.index, dtype='float64')
    
    for i in range(window, n_days):
        window_rets = df_returns.iloc[i-window:i]
        corr_matrix = window_rets.corr().values
        corr_matrix = np.nan_to_num(corr_matrix) # Khử nhiễu NaN
        
        # Eigenvalue Decomposition
        eigenvalues, _ = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        total_variance = np.sum(eigenvalues)
        if total_variance == 0: continue
            
        p_i = eigenvalues / total_variance
        p_i = p_i[p_i > 0]
        
        max_entropy = np.log(len(eigenvalues))
        # Chuẩn hóa Entropy System về biên độ 0-100 (100 là cực kỳ hỗn loạn)
        corr_entropy.iloc[i] = (-np.sum(p_i * np.log(p_i)) / max_entropy) * 100
        
    return corr_entropy

def detect_market_regime(df):
    """
    Xác định vùng trạng thái thị trường học thuật.
    """
    # Khởi tạo Regimes
    df['Regime'] = 'Neutral'
    df['Color'] = 'rgba(128, 128, 128, 0.2)'
    
    # Giá trị Entropy Daily chuẩn hóa 
    max_e = df['Entropy_Daily'].max()
    min_e = df['Entropy_Daily'].min()
    df['Norm_Entropy'] = (df['Entropy_Daily'] - min_e) / (max_e - min_e) * 100
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Entropy_Diff'] = df['Norm_Entropy'].diff().rolling(3).mean()
    
    # Các điều kiện phân vùng
    for i in range(len(df)):
        c = df['Close'].iloc[i]
        ma = df['MA20'].iloc[i]
        ent = df['Norm_Entropy'].iloc[i]
        ent_d = df['Entropy_Diff'].iloc[i]
        
        if pd.isna(ent) or pd.isna(ma):
            continue
            
        if c >= ma and ent < 50:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Stable Growth'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(0, 255, 65, 0.3)' # Lục
        elif c >= ma and ent >= 50:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Fragile Growth'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(255, 215, 0, 0.3)' # Vàng
        elif c < ma and ent > 65:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Chaos/Panic'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(255, 0, 0, 0.3)' # Đỏ
        elif c < ma and ent_d < 0:
            df.iloc[i, df.columns.get_loc('Regime')] = 'Bottoming'
            df.iloc[i, df.columns.get_loc('Color')] = 'rgba(138, 43, 226, 0.3)' # Tím

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
    st.markdown("<h1>SYSTEM ARCHITECT: STRUCTURAL ENTROPY & REGIME DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888;'>Khám phá sự đứt gãy cấu trúc (Structural Break) trong hành vi thị trường thông qua lăng kính Vật lý Hệ thống Phức tạp.</p>", unsafe_allow_html=True)
    
    # Dữ liệu chính
    with st.spinner('Building System Architectures... (Fetching Market Data)'):
        vni_df = fetch_market_data(ticker="VNINDEX", period="3y")
        vni_df = calculate_mse(vni_df)
        vni_df = detect_market_regime(vni_df)
        
        vni_df['RSI'] = calculate_rsi(vni_df['Close'])
        vni_df['MACD'], vni_df['MACD_Signal'] = calculate_macd(vni_df['Close'])
        
        vn30_returns = fetch_vn30_components()
        sys_entropy = calculate_correlation_entropy(vn30_returns)
        vni_df['System_Entropy'] = sys_entropy.reindex(vni_df.index).ffill()
        
    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    last_idx = vni_df.dropna().index[-1]
    curr_data = vni_df.loc[last_idx]
    
    with col1:
        st.markdown(f"<div class='metric-label'>VN-INDEX</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{curr_data['Close']:.2f}</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"<div class='metric-label'>Current Market Regime</div>", unsafe_allow_html=True)
        regime_color = curr_data['Color'].replace('0.3', '1.0') # Làm đậm để text rõ
        st.markdown(f"<div class='metric-value' style='color:{regime_color}'>{curr_data['Regime']}</div>", unsafe_allow_html=True)

    with col3:
        sys_ent_val = curr_data['System_Entropy']
        st.markdown(f"<div class='metric-label'>Cross-Sectional Entropy</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{sys_ent_val:.1f} / 100</div>", unsafe_allow_html=True)

    with col4:
        norm_ent = curr_data['Norm_Entropy']
        st.markdown(f"<div class='metric-label'>Time-Series Entropy (D)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{norm_ent:.1f} / 100</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- MAIN CHART: MẢNG MÀU THỂ HIỆN REGIME ---
    st.subheader("1. VN-Index Structural Regimes (Entropy-Driven)")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Nến giá VNI
    fig.add_trace(go.Candlestick(
        x=vni_df.index, open=vni_df['Open'], high=vni_df['High'],
        low=vni_df['Low'], close=vni_df['Close'], name='VN-Index',
        increasing_line_color='#00FF41', decreasing_line_color='#FF0000'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df['MA20'], line=dict(color='white', width=1, dash='dash'), name='MA20'
    ), row=1, col=1)

    # Tô màu Regime phía sau
    # Để tối ưu rendering, ta dùng fill_tozero hoặc các dải hcn
    for regime, color in zip(['Stable Growth', 'Fragile Growth', 'Chaos/Panic', 'Bottoming'], 
                             ['rgba(0, 255, 65, 0.3)', 'rgba(255, 215, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(138, 43, 226, 0.3)']):
        regime_mask = vni_df['Regime'] == regime
        if regime_mask.any():
            fig.add_trace(go.Scatter(
                x=vni_df.index[regime_mask],
                y=[vni_df['High'].max()] * len(vni_df[regime_mask]),
                mode='markers',
                marker=dict(symbol='square', size=8, color=color),
                name=regime
            ), row=1, col=1)

    # Entropy Line
    fig.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df['Norm_Entropy'], 
        line=dict(color='#00FF41', width=2), name='Daily Entropy'
    ), row=2, col=1)
    
    # Layout tinh chỉnh như Terminal
    fig.update_layout(
        template="plotly_dark", height=700, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='#0E1117', paper_bgcolor='#000000',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Entropy (0-100)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    
    # Ý nghĩa Regimes
    st.info("""
    **Vật lý của Market Regimes:**
    - 🟩 **Stable Growth (Nền Xanh Lục)**: Giá tăng, Entropy cấu trúc thấp. Hệ thống duy trì ma sát thấp, dòng tiền tập trung cao độ vào xu hướng chính.
    - 🟨 **Fragile Growth (Nền Vàng)**: Quá trình phân kỳ cấu trúc. Giá vẫn tăng nhờ quán tính nhưng Entropy gia tăng (Dòng tiền phân tách ra các nhóm nhỏ lẻ). Cảnh báo rủi ro gãy đổ.
    - 🟥 **Chaos / Panic (Nền Đỏ)**: Pha phá vỡ cấu trúc (Structural Break). Tổ chức bán ra, sự hoảng loạn đẩy mức cực đại của nhiễu tín hiệu.
    - 🟪 **Bottoming (Nền Tím)**: Quá trình hạ nhiệt Entropy (Damping). Dù giá vẫn giảm, cấu trúc đang dần có sự "Tự tổ chức" (Self-Organization) lại. Dòng tiền lớn bắt đầu tái lập trật tự.
    """)

    st.markdown("---")

    # --- CROSS-SECTIONAL ENTROPY ---
    st.subheader("2. MSE & VN30 Components Analysis")
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**Multi-scale Entropy (MSE) - Khử Nhiễu Lagging**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['Entropy_Daily'], name='Daily Entropy (Ngắn hạn)', line=dict(color='rgba(255, 255, 255, 0.4)')))
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['Entropy_Weekly'], name='Weekly Entropy (Trung hạn)', line=dict(color='rgba(0, 255, 65, 0.8)', width=2)))
        fig2.add_trace(go.Scatter(x=vni_df.index, y=vni_df['Entropy_Monthly'], name='Monthly Entropy (Dài hạn)', line=dict(color='rgba(255, 0, 255, 0.8)', width=3)))
        fig2.update_layout(template="plotly_dark", plot_bgcolor='#000000', paper_bgcolor='#000000', margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True})
        st.caption("Logic: Entropy Daily cao nhưng Weekly/Monthly thấp -> Biến động chỉ là NHIỄU NGẮN HẠN, xu hướng vĩ mô dài hạn vẫn bền vững. Người dùng không nên hoảng loạn.")

    with colB:
        st.markdown("**Cross-Sectional Entropy (Eigenvalue Decomposition VN30)**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=vni_df.index, y=vni_df['System_Entropy'], fill='tozeroy', name='System Entropy', line=dict(color='#00BFFF')))
        fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Chaos Threshold (Đỉnh Phân Hóa)")
        fig3.add_hline(y=40, line_dash="dash", line_color="#00FF41", annotation_text="High Consensus (Trend Mạnh)")
        fig3.update_layout(template="plotly_dark", plot_bgcolor='#000000', paper_bgcolor='#000000', margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': True})
        st.caption("Đo lường sự phân mảnh (Fractal) dòng tiền vào 30 mã vốn hóa cao nhất.")

    st.markdown("---")
    
    # --- COMPARISON LAYER: ENTROPY VS TRADITIONAL LAG INDICATORS ---
    st.subheader("3. Comparison Layer: Entropy vs Traditional (RSI & MACD)")
    st.markdown("So sánh hiệu suất Entropy (Leading) với các chỉ báo truyền thống thường bị trễ (Lagging).")
    
    fig_comp = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.33, 0.33, 0.33])
    
    # RSI
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['RSI'], name='RSI (14)', line=dict(color='yellow')), row=1, col=1)
    fig_comp.add_hline(y=70, row=1, col=1, line_dash="dash", line_color="gray")
    fig_comp.add_hline(y=30, row=1, col=1, line_dash="dash", line_color="gray")
    
    # MACD
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['MACD_Signal'], name='Signal', line=dict(color='magenta')), row=2, col=1)
    
    # Entropy
    fig_comp.add_trace(go.Scatter(x=vni_df.index, y=vni_df['Norm_Entropy'], name='Structural Entropy', line=dict(color='#00FF41')), row=3, col=1)
    fig_comp.add_hline(y=65, row=3, col=1, line_dash="dash", line_color="red", annotation_text="Tín hiệu sớm đứt gãy")
    
    fig_comp.update_layout(template="plotly_dark", plot_bgcolor='#000000', paper_bgcolor='#000000', height=600, margin=dict(l=20, r=20, t=20, b=20))
    fig_comp.update_yaxes(title_text="RSI", row=1, col=1)
    fig_comp.update_yaxes(title_text="MACD", row=2, col=1)
    fig_comp.update_yaxes(title_text="Entropy", row=3, col=1)
    
    st.plotly_chart(fig_comp, use_container_width=True, config={'scrollZoom': True})
    
    st.markdown("""
    > **Sự tinh hoa của Entropy Filter (Khử nhiễu cho RSI):** 
    > Ví dụ kinh điển: Khi RSI vào vùng siêu Quá Mua (> 80) nhưng Entropy Hệ thống giảm (duy trì mức thấp), điều đó chứng tỏ Khối Lượng và Chu kỳ vẫn đang tự tổ chức và củng cố Trend hoàn hảo -> **CHƯA BÁN**. Ngược lại, nếu RSI chỉ mới 60 nhưng Entropy bật tăng thẳng đứng, chứng tỏ "nội bộ đã đứt gãy (dòng tiền rời rạc)" -> **BÁN SỚM**.
    """)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#888'>Terminal Framework by Systems Physics Quant. Code with 💚 (Vibe Coding).</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
