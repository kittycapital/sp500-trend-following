#!/usr/bin/env python3
"""
S&P 500 Trend Following Dashboard - Data Fetcher
=================================================
Fetches all S&P 500 stocks, calculates trend following indicators,
composite trend score, profit-taking signals, and backtest statistics.

Designed to run daily via GitHub Actions.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ──────────────────────────────────────────────────────────
# 1. FETCH S&P 500 TICKERS
# ──────────────────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 tickers from hardcoded list (sp500_tickers.py)."""
    from sp500_tickers import SP500
    tickers = list(SP500.keys())
    sectors = {t: info["sector"] for t, info in SP500.items()}
    names = {t: info["name"] for t, info in SP500.items()}
    return tickers, sectors, names

# ──────────────────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────

def calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    atr = calc_atr(high, low, close, period)
    plus_di = 100 * calc_ema(plus_dm, period) / atr
    minus_di = 100 * calc_ema(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = calc_ema(dx, period)
    return adx

def calc_supertrend(high, low, close, period=10, multiplier=3):
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1]) if direction.iloc[i-1] == 1 else lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1]) if direction.iloc[i-1] == -1 else upper_band.iloc[i]
            direction.iloc[i] = -1

    return supertrend, direction

# ──────────────────────────────────────────────────────────
# 3. COMPOSITE TREND SCORE
# ──────────────────────────────────────────────────────────

def calc_trend_score(row):
    """Calculate composite trend score (0-100)."""
    score = 0

    # Price > 200 SMA (+15)
    if row['close'] > row['sma_200']:
        score += 15

    # Price > 50 SMA (+10)
    if row['close'] > row['sma_50']:
        score += 10

    # 50 SMA > 200 SMA - Golden Cross (+15)
    if row['sma_50'] > row['sma_200']:
        score += 15

    # 10 EMA > 30 EMA (+10)
    if row['ema_10'] > row['ema_30']:
        score += 10

    # MACD > Signal (+10)
    if row['macd'] > row['macd_signal']:
        score += 10

    # RSI > 50 (+10)
    if row['rsi'] > 50:
        score += 10

    # ADX > 25 - Strong trend (+15)
    if row['adx'] > 25:
        score += 15

    # ADX > 40 - Very strong trend bonus (+5)
    if row['adx'] > 40:
        score += 5

    # Supertrend = Uptrend (+10)
    if row['supertrend_dir'] == 1:
        score += 10

    return score

# ──────────────────────────────────────────────────────────
# 4. PROFIT-TAKING SIGNALS
# ──────────────────────────────────────────────────────────

def calc_profit_signals(df):
    """Calculate profit-taking and exit signals."""
    latest = df.iloc[-1]
    close = latest['close']
    atr = latest['atr']

    # Find entry point (when score crossed above 60)
    scores = df['trend_score']
    entry_price = None
    entry_date = None

    for i in range(len(scores) - 1, 0, -1):
        if scores.iloc[i] >= 60 and scores.iloc[i-1] < 60:
            entry_price = df['close'].iloc[i]
            entry_date = df.index[i]
            break

    if entry_price is None:
        # If always above or always below, use first day above 60
        above = df[df['trend_score'] >= 60]
        if len(above) > 0:
            entry_price = above['close'].iloc[0]
            entry_date = above.index[0]
        else:
            entry_price = close
            entry_date = df.index[-1]

    # ATR Trailing Stop
    recent_high = df['high'].rolling(20).max().iloc[-1]
    atr_stop = recent_high - 2 * atr
    stop_distance_pct = ((atr_stop - close) / close) * 100

    # Gain since signal
    gain_since_signal = ((close - entry_price) / entry_price) * 100

    # Days in trend
    days_in_trend = (df.index[-1] - entry_date).days if entry_date else 0

    # Profit zones
    profit_10 = entry_price * 1.10
    profit_20 = entry_price * 1.20
    profit_30 = entry_price * 1.30

    # EMA exit warning
    ema10_exit = close < latest['ema_10']

    # Trend score drop warning
    score_warning = latest['trend_score'] < 40

    return {
        'entry_price': round(entry_price, 2),
        'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
        'atr_trailing_stop': round(atr_stop, 2),
        'stop_distance_pct': round(stop_distance_pct, 2),
        'gain_since_signal': round(gain_since_signal, 2),
        'days_in_trend': days_in_trend,
        'profit_zone_10': round(profit_10, 2),
        'profit_zone_20': round(profit_20, 2),
        'profit_zone_30': round(profit_30, 2),
        'reached_10': close >= profit_10,
        'reached_20': close >= profit_20,
        'reached_30': close >= profit_30,
        'ema10_exit_warning': ema10_exit,
        'score_drop_warning': score_warning
    }

# ──────────────────────────────────────────────────────────
# 5. BACKTEST
# ──────────────────────────────────────────────────────────

def run_backtest(df):
    """Simple trend following backtest based on composite score."""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None

    for i in range(1, len(df)):
        score = df['trend_score'].iloc[i]
        close = df['close'].iloc[i]
        date = df.index[i]

        if not in_position and score >= 60:
            in_position = True
            entry_price = close
            entry_date = date

        elif in_position and score < 40:
            in_position = False
            exit_price = close
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
                'exit_date': str(date.date()) if hasattr(date, 'date') else str(date)[:10],
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'pnl_pct': round(pnl_pct, 2)
            })

    # Close open position at last price
    if in_position:
        close = df['close'].iloc[-1]
        pnl_pct = ((close - entry_price) / entry_price) * 100
        trades.append({
            'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
            'exit_date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])[:10],
            'entry_price': round(entry_price, 2),
            'exit_price': round(close, 2),
            'pnl_pct': round(pnl_pct, 2),
            'open': True
        })

    # Stats
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_winner': 0,
            'avg_loser': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'buy_hold_return': 0,
            'trades': []
        }

    winners = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losers = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]

    # Buy & hold return
    buy_hold = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

    # Max drawdown
    cumulative = (1 + df['close'].pct_change()).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    return {
        'total_trades': len(trades),
        'win_rate': round(len(winners) / len(trades) * 100, 1) if trades else 0,
        'avg_winner': round(np.mean(winners), 2) if winners else 0,
        'avg_loser': round(np.mean(losers), 2) if losers else 0,
        'total_return': round(sum(t['pnl_pct'] for t in trades), 2),
        'max_drawdown': round(max_dd, 2),
        'buy_hold_return': round(buy_hold, 2),
        'trades': trades[-5:]  # Last 5 trades only for JSON size
    }

# ──────────────────────────────────────────────────────────
# 6. PROCESS SINGLE STOCK
# ──────────────────────────────────────────────────────────

def process_stock(ticker, hist, sector, name):
    """Process a single stock and return all data."""
    try:
        if hist is None or len(hist) < 200:
            return None

        df = hist.copy()
        df.columns = [c.lower() for c in df.columns]

        if 'adj close' in df.columns:
            df['close'] = df['adj close']

        # Calculate indicators
        df['sma_50'] = calc_sma(df['close'], 50)
        df['sma_200'] = calc_sma(df['close'], 200)
        df['ema_10'] = calc_ema(df['close'], 10)
        df['ema_30'] = calc_ema(df['close'], 30)
        df['rsi'] = calc_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(df['close'])
        df['atr'] = calc_atr(df['high'], df['low'], df['close'])
        df['adx'] = calc_adx(df['high'], df['low'], df['close'])
        df['supertrend'], df['supertrend_dir'] = calc_supertrend(df['high'], df['low'], df['close'])

        # Drop NaN rows
        df = df.dropna(subset=['sma_200', 'adx'])
        if len(df) < 50:
            return None

        # Trend score for all rows
        df['trend_score'] = df.apply(calc_trend_score, axis=1)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Signal
        score = int(latest['trend_score'])
        if score >= 75:
            signal = "STRONG_BUY"
        elif score >= 60:
            signal = "BUY"
        elif score >= 40:
            signal = "HOLD"
        else:
            signal = "SELL"

        # Score change
        score_change = int(latest['trend_score'] - prev['trend_score'])

        # Profit signals
        profit = calc_profit_signals(df)

        # Backtest
        backtest = run_backtest(df)

        # 52-week high/low
        high_52w = df['high'].tail(252).max()
        low_52w = df['low'].tail(252).min()
        pct_from_high = ((latest['close'] - high_52w) / high_52w) * 100

        return {
            'ticker': ticker,
            'name': name,
            'sector': sector,
            'price': round(float(latest['close']), 2),
            'change_pct': round(float((latest['close'] - prev['close']) / prev['close'] * 100), 2),
            'volume': int(latest['volume']),
            'trend_score': score,
            'score_change': score_change,
            'signal': signal,
            'high_52w': round(float(high_52w), 2),
            'low_52w': round(float(low_52w), 2),
            'pct_from_high': round(float(pct_from_high), 2),
            'indicators': {
                'sma_50': round(float(latest['sma_50']), 2),
                'sma_200': round(float(latest['sma_200']), 2),
                'ema_10': round(float(latest['ema_10']), 2),
                'ema_30': round(float(latest['ema_30']), 2),
                'rsi': round(float(latest['rsi']), 1),
                'adx': round(float(latest['adx']), 1),
                'macd': round(float(latest['macd']), 3),
                'macd_signal': round(float(latest['macd_signal']), 3),
                'supertrend_dir': 'UP' if latest['supertrend_dir'] == 1 else 'DOWN',
                'atr': round(float(latest['atr']), 2)
            },
            'profit_taking': profit,
            'backtest': backtest
        }

    except Exception as e:
        print(f"  Error processing {ticker}: {e}")
        return None

# ──────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("S&P 500 Trend Following Dashboard - Data Fetcher")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get tickers
    print("\n[1/4] Fetching S&P 500 ticker list...")
    tickers, sectors, names = get_sp500_tickers()
    print(f"  Found {len(tickers)} tickers")

    if not tickers:
        print("ERROR: No tickers found. Exiting.")
        sys.exit(1)

    # Download data in batch (much faster)
    print("\n[2/4] Downloading price data (1 year)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # Extra buffer for 200 SMA

    # Download in batches of 100
    all_data = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch)} tickers)...")
        try:
            data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', progress=False, threads=True)
            for t in batch:
                try:
                    if len(batch) == 1:
                        stock_df = data
                    else:
                        stock_df = data[t]
                    if stock_df is not None and len(stock_df.dropna()) > 200:
                        all_data[t] = stock_df.dropna()
                except:
                    pass
        except Exception as e:
            print(f"  Batch error: {e}")
        time.sleep(1)

    print(f"  Successfully downloaded {len(all_data)} stocks")

    # Process all stocks
    print("\n[3/4] Calculating indicators and signals...")
    results = []
    for idx, (ticker, hist) in enumerate(all_data.items()):
        if (idx + 1) % 50 == 0:
            print(f"  Processing {idx+1}/{len(all_data)}...")
        result = process_stock(
            ticker, hist,
            sectors.get(ticker, 'Unknown'),
            names.get(ticker, ticker)
        )
        if result:
            results.append(result)

    print(f"  Successfully processed {len(results)} stocks")

    # Sort by trend score
    results.sort(key=lambda x: x['trend_score'], reverse=True)

    # Sector summary
    sector_scores = {}
    for r in results:
        s = r['sector']
        if s not in sector_scores:
            sector_scores[s] = []
        sector_scores[s].append(r['trend_score'])

    sector_summary = {}
    for s, scores in sector_scores.items():
        sector_summary[s] = {
            'avg_score': round(np.mean(scores), 1),
            'count': len(scores),
            'strong_buy': len([x for x in scores if x >= 75]),
            'buy': len([x for x in scores if 60 <= x < 75]),
            'hold': len([x for x in scores if 40 <= x < 60]),
            'sell': len([x for x in scores if x < 40])
        }

    # Overall stats
    all_scores = [r['trend_score'] for r in results]
    market_summary = {
        'total_stocks': len(results),
        'avg_score': round(np.mean(all_scores), 1),
        'strong_buy_count': len([s for s in all_scores if s >= 75]),
        'buy_count': len([s for s in all_scores if 60 <= s < 75]),
        'hold_count': len([s for s in all_scores if 40 <= s < 60]),
        'sell_count': len([s for s in all_scores if s < 40]),
        'median_score': round(np.median(all_scores), 1)
    }

    # Build output
    output = {
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'market_summary': market_summary,
        'sector_summary': sector_summary,
        'stocks': results
    }

    # Save
    print("\n[4/4] Saving data...")
    os.makedirs('data', exist_ok=True)
    with open('data/trend_data.json', 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n✅ Done! Saved {len(results)} stocks to data/trend_data.json")
    print(f"   Market Average Score: {market_summary['avg_score']}")
    print(f"   Strong Buy: {market_summary['strong_buy_count']} | Buy: {market_summary['buy_count']} | Hold: {market_summary['hold_count']} | Sell: {market_summary['sell_count']}")
    print(f"   Top 5: {', '.join(r['ticker'] for r in results[:5])}")

if __name__ == '__main__':
    main()
