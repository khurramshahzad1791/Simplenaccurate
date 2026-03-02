"""
MEXC ULTIMATE MULTI-TIMEFRAME SCANNER (FIXED)
- Scans ALL MEXC coins (up to 100+)
- Multi-timeframe analysis
- Position sizing with leverage
- Fixed f-string errors
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
from datetime import datetime, timedelta
import time
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Technical analysis
import ta

# Page config
st.set_page_config(
    page_title="MEXC ULTIMATE SCANNER",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .signal-card {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid;
        margin: 10px 0;
        transition: transform 0.3s;
        color: white;
    }
    .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .long-card { border-left-color: #00ff00; }
    .short-card { border-left-color: #ff4444; }
    .neutral-card { border-left-color: #888888; }
    .metric-box {
        background: #2d2d2d;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .filter-section {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .timeframe-badge {
        background: #4a4a4a;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📊 MEXC ULTIMATE MULTI-TF SCANNER</h1>', unsafe_allow_html=True)

# ============================================================================
# DATA FETCHER (with multi-timeframe support)
# ============================================================================

class MEXCDataFetcher:
    def __init__(self):
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1200
        })
        self.timeframes = {
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
    
    def get_all_symbols(self):
        """Fetch ALL active USDT pairs from MEXC"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets if '/USDT' in s and markets[s]['active']]
            return symbols
        except Exception as e:
            st.error(f"Error loading symbols: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        """Fetch OHLCV for a given symbol and timeframe"""
        try:
            tf = self.timeframes.get(timeframe, '1h')
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except Exception as e:
            return None
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        data = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit=200)
            if df is not None and len(df) >= 50:  # Ensure enough data
                data[tf] = df
        return data

# ============================================================================
# TECHNICAL INDICATORS CALCULATOR
# ============================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to dataframe"""
        if df is None or len(df) < 50:
            return df
        
        # Moving averages
        for period in [9, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['c'].rolling(period).mean()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['c'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['c'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Volume
        df['volume_sma'] = df['v'].rolling(20).mean()
        df['volume_ratio'] = df['v'] / df['volume_sma']
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(df['h'], df['l'], df['c'], window=14)
        
        return df

# ============================================================================
# SUPPORT/RESISTANCE DETECTOR (swing points + clustering)
# ============================================================================

class SupportResistanceDetector:
    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int = 5):
        highs = df['h'].values
        lows = df['l'].values
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df)-window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append((i, lows[i]))
        return swing_highs, swing_lows
    
    @staticmethod
    def detect_levels(df: pd.DataFrame, tolerance: float = 0.01) -> Dict:
        swing_highs, swing_lows = SupportResistanceDetector.find_swing_points(df)
        high_prices = [p[1] for p in swing_highs]
        low_prices = [p[1] for p in swing_lows]
        
        # Simple clustering
        def cluster(prices, tolerance):
            if not prices:
                return []
            prices = sorted(prices)
            clusters = []
            current_cluster = [prices[0]]
            for p in prices[1:]:
                if abs(p - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                    current_cluster.append(p)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [p]
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance_levels = cluster(high_prices, tolerance)
        support_levels = cluster(low_prices, tolerance)
        
        current_price = df['c'].iloc[-1]
        
        # Find nearest levels
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'current_price': current_price
        }

# ============================================================================
# TRENDLINE DETECTOR (simplified)
# ============================================================================

class TrendlineDetector:
    @staticmethod
    def detect_trend(df: pd.DataFrame) -> str:
        """Simple trend detection using linear regression slope on last 20 periods"""
        if len(df) < 20:
            return "neutral"
        y = df['c'].values[-20:]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.001:
            return "uptrend"
        elif slope < -0.001:
            return "downtrend"
        else:
            return "sideways"

# ============================================================================
# HIGH MOVER DETECTOR
# ============================================================================

class HighMoverDetector:
    @staticmethod
    def detect(df: pd.DataFrame, lookback: int = 30, move_threshold: float = 0.5) -> bool:
        """Detect if a coin had a large move (e.g., 50% up then 50% down)"""
        if len(df) < lookback:
            return False
        recent = df.tail(lookback)
        high = recent['h'].max()
        low = recent['l'].min()
        current = df['c'].iloc[-1]
        # Check if price moved significantly and then retraced
        if high / low > 1.5:  # 50% range
            # Check if current price is between low and high (retraced)
            if low * 1.1 < current < high * 0.9:
                return True
        return False

# ============================================================================
# POSITION SIZING ENGINE
# ============================================================================

class PositionSizer:
    @staticmethod
    def calculate(account_balance: float, risk_percent: float, entry: float, stop: float, leverage: float = 1.0) -> Dict:
        """Calculate position size and potential profit"""
        risk_amount = account_balance * (risk_percent / 100)
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            return {}
        position_size = risk_amount / stop_distance
        position_value = position_size * entry
        # Adjust for leverage
        required_margin = position_value / leverage
        return {
            'position_size': position_size,
            'position_value': position_value,
            'required_margin': required_margin,
            'risk_amount': risk_amount,
            'leverage': leverage
        }

# ============================================================================
# MULTI-TIMEFRAME ANALYZER
# ============================================================================

class MultiTimeframeAnalyzer:
    def __init__(self, timeframes: List[str] = ['1M', '1w', '1d', '4h', '1h', '15m']):
        self.timeframes = timeframes
    
    def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze multi-timeframe data and produce a signal"""
        if not data:
            return {}
        
        # Hierarchical analysis: higher timeframes set the bias
        bias = "neutral"
        for tf in ['1M', '1w', '1d', '4h', '1h']:
            if tf in data and data[tf] is not None:
                df = data[tf]
                if len(df) > 20:
                    trend = TrendlineDetector.detect_trend(df)
                    if trend != "neutral":
                        bias = trend
                        break
        
        # Now analyze the entry timeframe (e.g., 15m)
        entry_tf = '15m'
        if entry_tf not in data:
            entry_tf = '1h'  # fallback
        if entry_tf not in data:
            return {}
        
        df_entry = data.get(entry_tf)
        if df_entry is None or len(df_entry) < 20:
            return {}
        
        df_entry = TechnicalIndicators.calculate_all(df_entry)
        current = df_entry.iloc[-1]
        prev = df_entry.iloc[-2]
        
        # Technical conditions
        ma_alignment = (
            current['ma_9'] > current['ma_20'] > current['ma_50'] if bias == "uptrend"
            else current['ma_9'] < current['ma_20'] < current['ma_50'] if bias == "downtrend"
            else True
        )
        
        rsi_ok = (30 < current['rsi'] < 70)
        
        volume_ok = current['volume_ratio'] > 1.2
        
        macd_bullish = current['macd'] > current['macd_signal'] and current['macd_hist'] > 0
        macd_bearish = current['macd'] < current['macd_signal'] and current['macd_hist'] < 0
        
        # Support/Resistance
        sr = SupportResistanceDetector.detect_levels(df_entry)
        near_support = sr['nearest_support'] and current['c'] <= sr['nearest_support'] * 1.02
        near_resistance = sr['nearest_resistance'] and current['c'] >= sr['nearest_resistance'] * 0.98
        
        # Determine signal direction
        signal = "neutral"
        confidence = 0
        reasons = []
        
        if bias == "uptrend" and near_support and ma_alignment and rsi_ok and volume_ok and macd_bullish:
            signal = "long"
            confidence = 80
            reasons = ["Higher TF uptrend", "Near support", "MA aligned", "RSI healthy", "Volume spike", "MACD bullish"]
        elif bias == "downtrend" and near_resistance and ma_alignment and rsi_ok and volume_ok and macd_bearish:
            signal = "short"
            confidence = 80
            reasons = ["Higher TF downtrend", "Near resistance", "MA aligned", "RSI healthy", "Volume spike", "MACD bearish"]
        elif bias == "uptrend" and near_support:
            signal = "long"
            confidence = 60
            reasons = ["Higher TF uptrend", "Near support"]
        elif bias == "downtrend" and near_resistance:
            signal = "short"
            confidence = 60
            reasons = ["Higher TF downtrend", "Near resistance"]
        
        return {
            'symbol': symbol.replace('/USDT', ''),
            'price': current['c'],
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'bias': bias,
            'entry_tf': entry_tf,
            'near_support': sr['nearest_support'],
            'near_resistance': sr['nearest_resistance'],
            'atr': current['atr'],
            'rsi': current['rsi'],
            'volume_ratio': current['volume_ratio'],
            'ma_9': current['ma_9'],
            'ma_20': current['ma_20'],
            'ma_50': current['ma_50'],
            'ma_100': current.get('ma_100', 0),
            'ma_200': current.get('ma_200', 0),
            'trendline': TrendlineDetector.detect_trend(df_entry)
        }

# ============================================================================
# SCANNER ENGINE
# ============================================================================

class Scanner:
    def __init__(self, fetcher: MEXCDataFetcher, analyzer: MultiTimeframeAnalyzer):
        self.fetcher = fetcher
        self.analyzer = analyzer
    
    def scan(self, symbols: List[str], filters: Dict) -> List[Dict]:
        results = []
        total = len(symbols)
        progress_bar = st.progress(0)
        status = st.empty()
        start_time = time.time()
        
        for i, symbol in enumerate(symbols):
            elapsed = time.time() - start_time
            est_remaining = (elapsed/(i+1))*(total-i-1) if i>0 else 0
            status.text(f"Scanning {i+1}/{total}: {symbol} | Elapsed: {elapsed:.0f}s | Est remaining: {est_remaining:.0f}s")
            
            # Fetch multi-timeframe data
            data = self.fetcher.get_multi_timeframe_data(symbol, self.analyzer.timeframes)
            if not data:
                progress_bar.progress((i+1)/total)
                continue
            
            # Analyze
            signal = self.analyzer.analyze(symbol, data)
            if not signal:
                progress_bar.progress((i+1)/total)
                continue
            
            # Apply filters
            if filters.get('near_support_only') and not signal.get('near_support'):
                progress_bar.progress((i+1)/total)
                continue
            if filters.get('near_resistance_only') and not signal.get('near_resistance'):
                progress_bar.progress((i+1)/total)
                continue
            if filters.get('high_movers_only'):
                # Need to check high mover on any timeframe
                is_mover = False
                for tf, df in data.items():
                    if HighMoverDetector.detect(df):
                        is_mover = True
                        break
                if not is_mover:
                    progress_bar.progress((i+1)/total)
                    continue
            
            results.append(signal)
            progress_bar.progress((i+1)/total)
        
        status.text(f"Scan completed in {time.time()-start_time:.1f}s")
        progress_bar.empty()
        return results

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Initialize
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = MEXCDataFetcher()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MultiTimeframeAnalyzer()
if 'scanner' not in st.session_state:
    st.session_state.scanner = Scanner(st.session_state.fetcher, st.session_state.analyzer)
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

# Sidebar
with st.sidebar:
    st.header("🔍 SCAN SETTINGS")
    
    # Timeframe selection
    st.subheader("⏰ Timeframes")
    available_tfs = ['15m', '1h', '4h', '1d', '1w', '1M']
    selected_tfs = st.multiselect(
        "Select timeframes to analyze",
        available_tfs,
        default=['1d', '4h', '1h', '15m']
    )
    if selected_tfs:
        st.session_state.analyzer.timeframes = selected_tfs
    
    st.divider()
    
    # Filters
    st.subheader("🎯 Filters")
    filter_near_support = st.checkbox("Near Support only")
    filter_near_resistance = st.checkbox("Near Resistance only")
    filter_high_movers = st.checkbox("High Movers only")
    
    filters = {
        'near_support_only': filter_near_support,
        'near_resistance_only': filter_near_resistance,
        'high_movers_only': filter_high_movers
    }
    
    st.divider()
    
    # Position sizing
    st.subheader("💰 Position Sizing")
    account_balance = st.number_input("Account Balance (USDT)", min_value=10, value=1000, step=100)
    risk_percent = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    leverage = st.number_input("Leverage", min_value=1, value=1, step=1)
    
    st.divider()
    
    # Scan button
    if st.button("🚀 START SCAN (ALL COINS)", use_container_width=True, type="primary"):
        st.session_state.scanning = True
    
    # Manual refresh
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.session_state.scanning = True
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    
    st.divider()
    st.caption("Note: Scanning all MEXC coins may take 2-5 minutes. Please be patient.")

# Main area
if st.session_state.scanning:
    symbols = st.session_state.fetcher.get_all_symbols()
    if not symbols:
        st.error("Failed to fetch symbols. Please try again.")
        st.session_state.scanning = False
    else:
        st.info(f"Found {len(symbols)} coins. Starting scan... This may take several minutes.")
        with st.spinner("Scanning..."):
            results = st.session_state.scanner.scan(symbols, filters)
            st.session_state.scan_results = results
            st.session_state.scanning = False

# Display results
if st.session_state.scan_results:
    st.success(f"✅ Found {len(st.session_state.scan_results)} trading opportunities")
    
    # Separate by signal
    long_signals = [r for r in st.session_state.scan_results if r['signal'] == 'long']
    short_signals = [r for r in st.session_state.scan_results if r['signal'] == 'short']
    neutral_signals = [r for r in st.session_state.scan_results if r['signal'] == 'neutral']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("LONG", len(long_signals))
    col2.metric("SHORT", len(short_signals))
    col3.metric("NEUTRAL", len(neutral_signals))
    
    # Tabs for long/short
    tab_long, tab_short = st.tabs(["📈 LONG SIGNALS", "📉 SHORT SIGNALS"])
    
    with tab_long:
        if long_signals:
            for sig in long_signals:
                # Calculate position
                if sig['atr'] and not np.isnan(sig['atr']):
                    stop_price = sig['price'] - 2 * sig['atr']
                else:
                    stop_price = sig['price'] * 0.98
                pos = PositionSizer.calculate(
                    account_balance, risk_percent, sig['price'], stop_price, leverage
                )
                with st.container():
                    # Format support/resistance strings safely
                    near_support_str = f"${sig['near_support']:.4f}" if sig['near_support'] else "N/A"
                    near_resistance_str = f"${sig['near_resistance']:.4f}" if sig['near_resistance'] else "N/A"
                    
                    st.markdown(f"""
                    <div class='signal-card long-card'>
                        <div style='display: flex; justify-content: space-between;'>
                            <h2>{sig['symbol']}</h2>
                            <span class='timeframe-badge'>Entry TF: {sig['entry_tf']}</span>
                        </div>
                        <h3>LONG at ${sig['price']:.4f}</h3>
                        <p>Confidence: {sig['confidence']}% | Bias: {sig['bias'].upper()}</p>
                        <p>{' | '.join(sig['reasons'])}</p>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0;'>
                            <div class='metric-box'>RSI: {sig['rsi']:.1f}</div>
                            <div class='metric-box'>Vol: {sig['volume_ratio']:.1f}x</div>
                            <div class='metric-box'>ATR: ${sig['atr']:.4f}</div>
                            <div class='metric-box'>Trend: {sig['trendline']}</div>
                        </div>
                        <p><strong>Near support:</strong> {near_support_str} | <strong>Near resistance:</strong> {near_resistance_str}</p>
                    """, unsafe_allow_html=True)
                    
                    if pos:
                        target1 = sig['price'] + 3 * sig['atr'] if sig['atr'] else sig['price'] * 1.03
                        target2 = sig['price'] + 5 * sig['atr'] if sig['atr'] else sig['price'] * 1.05
                        st.markdown(f"""
                        <div style='background: #2d2d2d; padding: 10px; border-radius: 8px; margin-top: 10px;'>
                            <h4>Position Plan</h4>
                            <p>Entry: ${sig['price']:.4f} | Stop: ${stop_price:.4f} | Target 1: ${target1:.4f} | Target 2: ${target2:.4f}</p>
                            <p>Position size: {pos['position_size']:.4f} units | Value: ${pos['position_value']:.2f} | Margin: ${pos['required_margin']:.2f}</p>
                            <p>Risk: ${pos['risk_amount']:.2f} ({risk_percent}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No long signals found")
    
    with tab_short:
        if short_signals:
            for sig in short_signals:
                if sig['atr'] and not np.isnan(sig['atr']):
                    stop_price = sig['price'] + 2 * sig['atr']
                else:
                    stop_price = sig['price'] * 1.02
                pos = PositionSizer.calculate(
                    account_balance, risk_percent, sig['price'], stop_price, leverage
                )
                with st.container():
                    near_support_str = f"${sig['near_support']:.4f}" if sig['near_support'] else "N/A"
                    near_resistance_str = f"${sig['near_resistance']:.4f}" if sig['near_resistance'] else "N/A"
                    
                    st.markdown(f"""
                    <div class='signal-card short-card'>
                        <div style='display: flex; justify-content: space-between;'>
                            <h2>{sig['symbol']}</h2>
                            <span class='timeframe-badge'>Entry TF: {sig['entry_tf']}</span>
                        </div>
                        <h3>SHORT at ${sig['price']:.4f}</h3>
                        <p>Confidence: {sig['confidence']}% | Bias: {sig['bias'].upper()}</p>
                        <p>{' | '.join(sig['reasons'])}</p>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0;'>
                            <div class='metric-box'>RSI: {sig['rsi']:.1f}</div>
                            <div class='metric-box'>Vol: {sig['volume_ratio']:.1f}x</div>
                            <div class='metric-box'>ATR: ${sig['atr']:.4f}</div>
                            <div class='metric-box'>Trend: {sig['trendline']}</div>
                        </div>
                        <p><strong>Near support:</strong> {near_support_str} | <strong>Near resistance:</strong> {near_resistance_str}</p>
                    """, unsafe_allow_html=True)
                    
                    if pos:
                        target1 = sig['price'] - 3 * sig['atr'] if sig['atr'] else sig['price'] * 0.97
                        target2 = sig['price'] - 5 * sig['atr'] if sig['atr'] else sig['price'] * 0.95
                        st.markdown(f"""
                        <div style='background: #2d2d2d; padding: 10px; border-radius: 8px; margin-top: 10px;'>
                            <h4>Position Plan</h4>
                            <p>Entry: ${sig['price']:.4f} | Stop: ${stop_price:.4f} | Target 1: ${target1:.4f} | Target 2: ${target2:.4f}</p>
                            <p>Position size: {pos['position_size']:.4f} units | Value: ${pos['position_value']:.2f} | Margin: ${pos['required_margin']:.2f}</p>
                            <p>Risk: ${pos['risk_amount']:.2f} ({risk_percent}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No short signals found")

else:
    st.info("👈 Configure filters and click START SCAN to begin scanning all MEXC coins")

# Footer
st.divider()
if st.session_state.scan_results:
    st.caption(f"🔄 Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total signals: {len(st.session_state.scan_results)}")
else:
    st.caption(f"🔄 Ready to scan | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(30)
    st.rerun()
