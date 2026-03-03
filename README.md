# Market Entropy: Vietnam Stock Exchange Chaos Dashboard

A streamlined financial dashboard built upon the theories of **Complex Systems Physics**, designed to detect turning points in the Vietnam Stock Index (VN-Index) by applying Information Theory principles to measure capital dispersion and structural chaos within the financial system.

This project introduces a **“Vibe Coding” System Architect Dashboard**, offering a Bloomberg-style Dark Mode visual experience built purely with Streamlit.

## 🌟 The Physics of Financial Chaos
Traditional indicators (RSI, MACD) are notoriously *lagging*, tracking price alone rather than the systemic strength of the price movement. This dashboard introduces an **Entropy-driven mathematical foundation** to decouple the "Market Noise" from the "Regimes." In adaptive systems theory, high entropy indicates energy dissipation — capital fragments and market consensuses break down before the formal structure (the index price) collapses.

### 1. 📈 Multi-Scale Weighted Permutation Entropy (WPE)
The system calculates **Weighted Permutation Entropy (WPE)** for both the Index Price and Trading Volume independently, measured across three time horizons (Daily, Weekly, Monthly) to separate lagging noise from long-term structure.

**Formula:**
Given an embedding dimension $m$ (default = 3, up to 7) and time lag $\tau=1$, we extract embedded ordinal patterns. Traditional Permutation Entropy is enhanced by WPE, which incorporates the variance of the amplitude vectors as weights.

$$ H(WPE) = - \frac{1}{\ln(m!)} \sum p^{(w)}_i \ln(p^{(w)}_i) $$

Where $p^{(w)}_i$ represents the variance-weighted frequency of the $i$-th ordinal pattern. If the market forms predictable repeating shapes, Entropy sits near zero. If price paths are random jumps, Entropy approaches 1.

### 2. 🧩 Cross-Sectional System Correlation Entropy (VN30)
Evaluates the internal agreement of the 30 largest capitalized stocks in the Vietnam market. The core calculation computes the Correlation Matrix of returns across these tickers and performs **Eigenvalue Decomposition (EVD)**.

**Formula:**
Let $C$ be the Pearson Correlation Matrix of returns for $M$ stocks (e.g., VN30 components, $M=30$). EVD factorizes the matrix to extract $M$ eigenvalues $\lambda_i$.
We define the normalized variance contribution of each principal component as:

$$ p_i = \frac{\lambda_i}{\sum_{j=1}^{M} \lambda_j} $$

The System Correlation Entropy $S_{corr}$, normalized to a $0-100$ scale (measuring maximum vs minimum chaos), is:

$$ S_{corr} = - \left( \frac{\sum_{i=1}^{M} p_i \ln(p_i)}{\ln(M)} \right) \times 100 $$

- **Low Entropy (< 40)**: A few prominent eigenvalues represent high market consensus; capital moves forcefully entirely in one directional trend.
- **High Entropy (> 70)**: The eigenvalues become equally dispersed. The capital flow is fragmented (Chaos State), often marking imminent structural breakdown and offering early warning warnings relative to lagging price indicators.

### 3. 🌀 Jensen-Shannon Statistical Complexity ($C$) & Market Fragility Index (MFI)
By applying Jensen-Shannon divergence between the empirical ordinal pattern distribution $P=\{p_i\}$ and the normalized uniform random probability distribution $U$, we compute the **Statistical Complexity ($C$)**:

$$ C = Q_0 \cdot JSD(P, U) \cdot H $$

The marriage of Entropy and Statistical Complexity results in the ultimate proxy for imminent structural breakdown—the **Market Fragility Index (MFI)**:

$$ MFI = WPE \times (1 - C) $$

- **Logic Context**: Rapid uptrends driven by "Hot Money" fundamentally erode structural integrity. When a system becomes aggressively random ($WPE$ peaks) and completely loses its internal deterministic complexity ($C$ plummets), $MFI$ spikes drastically, mathematically flagging 'Fragile Growth'.

### 4. 🎨 Complexity-Entropy Causality Plane & MFI-Driven Regimes 
- **CECP Plot**: A physical phase-space mapping of Entropy (X-axis) and Complexity (Y-axis). Theoretical Upper and Lower boundary curves formulated by *López-Ruiz* encase the possible outcomes of a complex state. The index trajectory is recorded across the CECP to determine if it flows toward pure randomness (White Noise) or deterministic structure. 
- **MFI Regime Engine**: Replaces traditional lagging lines.
  - 🟩 **Stable Growth**: Prices trace above MAs in a Low-MFI (structurally ordered) regime.
  - 🟨 **Fragile Growth**: Prices rise nominally, but $MFI$ balloons—meaning the fundamental market structure has completely fractured. Extreme collapse risk.
  - 🟥 **Chaos / Panic**: Mathematical execution of the price breakdown.
  - 🟪 **Bottoming**: Prices are depressed, but MFI successfully "Damps / Cools off." Formally reorganizing.

---

## 🚀 Features & Usability

- **API Integrations:** Out-of-the-box integration directly with native Vietnamese Market Data (`Vnstock`) & Yahoo Finance integrations. 
- **Offline Reliability / Graceful Degradation:** A complete suite of localized fallback settings. Fully supports importing Custom CSV / Excel data if remote APIs rate limit or block server IP access from Cloud Hosting.
- **Full Physics Engine Export:** The comprehensive calculation block dynamically generates a complete DataFrame of the calculated metrics ($WPE, Complexity, MFI, Regimes$) downloadable directly to your local workstation.
- **Trading-Native Interactive Design:** Implemented fully responsive UI with dual-synchronized TradingView-style interactive zooming, fluid chart pannings, and dynamic Y-axis scale dragging.

## 🛠 Prerequisites and Installation

**Dependencies**
Requires Python `3.9` or higher.
Libraries used: `streamlit`, `pandas`, `numpy`, `scipy`, `plotly`, `vnstock`, `yfinance`, `openpyxl`.

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/hoangthai35/marketentropy.git
   cd marketentropy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the App:
   ```bash
   streamlit run app.py
   ```

## 🧠 Credits & Theory
Authored by the mindset of a **Complex Systems Quant Designer**.
For theoretical background, users are encouraged to explore literature discussing *Shannon Information Theory*, *Multi-Scale Entropy in Econophysics*, and *Random Matrix Theory (RMT)* applications into market correlations.
