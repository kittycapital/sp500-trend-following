#!/usr/bin/env python3
"""
S&P 500 Trend Following Dashboard - Data Fetcher v2
====================================================
Includes equity curve, trade markers, price history for top 100,
and daily market history accumulation.
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
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── 1. TICKERS ──
def get_sp500_tickers():
    from sp500_tickers import SP500
    tickers = list(SP500.keys())
    sectors = {t: info["sector"] for t, info in SP500.items()}
    names = {t: info["name"] for t, info in SP500.items()}
    return tickers, sectors, names

# ── 2. INDICATORS ──
def calc_sma(s, p): return s.rolling(window=p, min_periods=p).mean()
def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_rsi(series, period=14):
    d = series.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, min_periods=period).mean()
    al = l.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al))

def calc_macd(series, fast=12, slow=26, sig=9):
    ml = calc_ema(series, fast) - calc_ema(series, slow)
    sl = calc_ema(ml, sig)
    return ml, sl, ml - sl

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=p, min_periods=p).mean()

def calc_adx(h, l, c, p=14):
    pdm = h.diff(); mdm = -l.diff()
    pdm[pdm < 0] = 0; mdm[mdm < 0] = 0
    pdm[pdm < mdm] = 0; mdm[mdm < pdm] = 0
    atr = calc_atr(h, l, c, p)
    pdi = 100 * calc_ema(pdm, p) / atr
    mdi = 100 * calc_ema(mdm, p) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return calc_ema(dx, p)

def calc_supertrend(h, l, c, p=10, m=3):
    atr = calc_atr(h, l, c, p); hl2 = (h + l) / 2
    ub = hl2 + m * atr; lb = hl2 - m * atr
    st = pd.Series(index=c.index, dtype=float)
    dr = pd.Series(index=c.index, dtype=float)
    st.iloc[0] = ub.iloc[0]; dr.iloc[0] = -1
    for i in range(1, len(c)):
        if c.iloc[i] > st.iloc[i-1]:
            st.iloc[i] = max(lb.iloc[i], st.iloc[i-1]) if dr.iloc[i-1] == 1 else lb.iloc[i]
            dr.iloc[i] = 1
        else:
            st.iloc[i] = min(ub.iloc[i], st.iloc[i-1]) if dr.iloc[i-1] == -1 else ub.iloc[i]
            dr.iloc[i] = -1
    return st, dr

# ── 3. TREND SCORE ──
def calc_trend_score(row):
    s = 0
    if row['close'] > row['sma_200']: s += 15
    if row['close'] > row['sma_50']: s += 10
    if row['sma_50'] > row['sma_200']: s += 15
    if row['ema_10'] > row['ema_30']: s += 10
    if row['macd'] > row['macd_signal']: s += 10
    if row['rsi'] > 50: s += 10
    if row['adx'] > 25: s += 15
    if row['adx'] > 40: s += 5
    if row['supertrend_dir'] == 1: s += 10
    return s

# ── 4. PROFIT SIGNALS ──
def calc_profit_signals(df):
    latest = df.iloc[-1]; close = latest['close']; atr = latest['atr']
    scores = df['trend_score']; entry_price = None; entry_date = None
    for i in range(len(scores) - 1, 0, -1):
        if scores.iloc[i] >= 60 and scores.iloc[i-1] < 60:
            entry_price = df['close'].iloc[i]; entry_date = df.index[i]; break
    if entry_price is None:
        above = df[df['trend_score'] >= 60]
        if len(above) > 0: entry_price = above['close'].iloc[0]; entry_date = above.index[0]
        else: entry_price = close; entry_date = df.index[-1]
    rh = df['high'].rolling(20).max().iloc[-1]; atr_stop = rh - 2 * atr
    return {
        'entry_price': round(entry_price, 2),
        'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
        'atr_trailing_stop': round(atr_stop, 2),
        'stop_distance_pct': round(((atr_stop - close) / close) * 100, 2),
        'gain_since_signal': round(((close - entry_price) / entry_price) * 100, 2),
        'days_in_trend': (df.index[-1] - entry_date).days if entry_date else 0,
        'profit_zone_10': round(entry_price * 1.10, 2),
        'profit_zone_20': round(entry_price * 1.20, 2),
        'profit_zone_30': round(entry_price * 1.30, 2),
        'reached_10': close >= entry_price * 1.10,
        'reached_20': close >= entry_price * 1.20,
        'reached_30': close >= entry_price * 1.30,
        'ema10_exit_warning': close < latest['ema_10'],
        'score_drop_warning': latest['trend_score'] < 40
    }

# ── 5. BACKTEST WITH EQUITY CURVE + TRADE MARKERS ──
def run_backtest(df):
    trades = []; markers = []; in_pos = False; ep = 0; ed = None
    cap = 100.0; equity = []; bh_start = float(df['close'].iloc[0])

    for i in range(len(df)):
        sc = df['trend_score'].iloc[i]; cl = float(df['close'].iloc[i])
        dt = df.index[i]
        ds = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]

        if not in_pos and sc >= 60 and i > 0:
            in_pos = True; ep = cl; ed = dt
            markers.append({'type':'BUY','date':ds,'price':round(cl,2),'score':int(sc)})
        elif in_pos and sc < 40:
            in_pos = False; pnl = ((cl - ep) / ep) * 100; cap *= (1 + pnl / 100)
            markers.append({'type':'SELL','date':ds,'price':round(cl,2),'score':int(sc),'pnl_pct':round(pnl,2),'days_held':(dt-ed).days})
            trades.append({'entry_date':str(ed.date()) if hasattr(ed,'date') else str(ed)[:10],'exit_date':ds,'entry_price':round(ep,2),'exit_price':round(cl,2),'pnl_pct':round(pnl,2)})

        mtm = cap * (cl / ep) if in_pos and ep > 0 else cap
        bh = (cl / bh_start) * 100
        if i % 5 == 0 or i == len(df) - 1:
            equity.append({'d':ds,'s':round(mtm,2),'b':round(bh,2)})

    if in_pos:
        cl = float(df['close'].iloc[-1]); pnl = ((cl - ep) / ep) * 100
        trades.append({'entry_date':str(ed.date()) if hasattr(ed,'date') else str(ed)[:10],'exit_date':str(df.index[-1].date()) if hasattr(df.index[-1],'date') else str(df.index[-1])[:10],'entry_price':round(ep,2),'exit_price':round(cl,2),'pnl_pct':round(pnl,2),'open':True})

    if not trades:
        return {'total_trades':0,'win_rate':0,'avg_winner':0,'avg_loser':0,'total_return':0,'max_drawdown':0,'buy_hold_return':0,'trades':[],'trade_markers':[],'equity_curve':equity}

    w = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    l = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]
    bhr = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    cum = (1 + df['close'].pct_change()).cumprod(); mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

    return {
        'total_trades':len(trades),'win_rate':round(len(w)/len(trades)*100,1) if trades else 0,
        'avg_winner':round(float(np.mean(w)),2) if w else 0,'avg_loser':round(float(np.mean(l)),2) if l else 0,
        'total_return':round(float(sum(t['pnl_pct'] for t in trades)),2),'max_drawdown':round(float(mdd),2),
        'buy_hold_return':round(float(bhr),2),'trades':trades[-5:],'trade_markers':markers,'equity_curve':equity
    }

# ── 6. PRICE HISTORY (60 days) ──
def get_price_history(df, n=60):
    tail = df.tail(n); h = []
    for i in range(len(tail)):
        r = tail.iloc[i]; ds = str(tail.index[i].date()) if hasattr(tail.index[i],'date') else str(tail.index[i])[:10]
        h.append({'d':ds,'c':round(float(r['close']),2),'s50':round(float(r['sma_50']),2),'s200':round(float(r['sma_200']),2),'st':1 if r['supertrend_dir']==1 else 0})
    return h

# ── 7. PROCESS STOCK ──
def process_stock(ticker, hist, sector, name, include_ph=False):
    try:
        if hist is None or len(hist) < 200: return None
        df = hist.copy(); df.columns = [c.lower() for c in df.columns]
        if 'adj close' in df.columns: df['close'] = df['adj close']

        df['sma_50'] = calc_sma(df['close'], 50); df['sma_200'] = calc_sma(df['close'], 200)
        df['ema_10'] = calc_ema(df['close'], 10); df['ema_30'] = calc_ema(df['close'], 30)
        df['rsi'] = calc_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(df['close'])
        df['atr'] = calc_atr(df['high'], df['low'], df['close'])
        df['adx'] = calc_adx(df['high'], df['low'], df['close'])
        df['supertrend'], df['supertrend_dir'] = calc_supertrend(df['high'], df['low'], df['close'])
        df = df.dropna(subset=['sma_200', 'adx'])
        if len(df) < 50: return None

        df['trend_score'] = df.apply(calc_trend_score, axis=1)
        latest = df.iloc[-1]; prev = df.iloc[-2] if len(df) > 1 else latest
        score = int(latest['trend_score'])
        signal = "STRONG_BUY" if score >= 75 else "BUY" if score >= 60 else "HOLD" if score >= 40 else "SELL"

        result = {
            'ticker': ticker, 'name': name, 'sector': sector,
            'price': round(float(latest['close']), 2),
            'change_pct': round(float((latest['close'] - prev['close']) / prev['close'] * 100), 2),
            'volume': int(latest['volume']), 'trend_score': score,
            'score_change': int(latest['trend_score'] - prev['trend_score']),
            'signal': signal,
            'high_52w': round(float(df['high'].tail(252).max()), 2),
            'low_52w': round(float(df['low'].tail(252).min()), 2),
            'pct_from_high': round(float((latest['close'] - df['high'].tail(252).max()) / df['high'].tail(252).max() * 100), 2),
            'indicators': {
                'sma_50': round(float(latest['sma_50']),2), 'sma_200': round(float(latest['sma_200']),2),
                'ema_10': round(float(latest['ema_10']),2), 'ema_30': round(float(latest['ema_30']),2),
                'rsi': round(float(latest['rsi']),1), 'adx': round(float(latest['adx']),1),
                'macd': round(float(latest['macd']),3), 'macd_signal': round(float(latest['macd_signal']),3),
                'supertrend_dir': 'UP' if latest['supertrend_dir'] == 1 else 'DOWN',
                'atr': round(float(latest['atr']),2)
            },
            'profit_taking': calc_profit_signals(df),
            'backtest': run_backtest(df)
        }
        if include_ph: result['price_history'] = get_price_history(df)
        return result
    except Exception as e:
        print(f"  Error processing {ticker}: {e}"); return None

# ── 8. TREND HISTORY ──
def update_trend_history(ms, ss):
    hf = 'data/trend_history.json'; h = []
    if os.path.exists(hf):
        try:
            with open(hf, 'r') as f: h = json.load(f)
        except: h = []
    today = datetime.now().strftime('%Y-%m-%d')
    h = [x for x in h if x['date'] != today]
    h.append({'date':today,'avg_score':ms['avg_score'],'median_score':ms['median_score'],
              'strong_buy':ms['strong_buy_count'],'buy':ms['buy_count'],'hold':ms['hold_count'],
              'sell':ms['sell_count'],'sectors':{s:d['avg_score'] for s,d in ss.items()}})
    h = h[-365:]
    for path in ['data/trend_history.json', 'trend_history.json']:
        with open(path, 'w') as f: json.dump(h, f, indent=2, cls=NumpyEncoder)
    print(f"  Trend history: {len(h)} days")

# ── 9. MAIN ──
def main():
    print("=" * 60)
    print("S&P 500 Trend Following Dashboard v2")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/5] Loading tickers...")
    tickers, sectors, names = get_sp500_tickers()
    print(f"  {len(tickers)} tickers")

    if not tickers: print("ERROR: No tickers."); sys.exit(1)

    print("\n[2/5] Downloading price data...")
    end_date = datetime.now(); start_date = end_date - timedelta(days=400)
    all_data = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        print(f"  Batch {i//100+1}/{(len(tickers)-1)//100+1} ({len(batch)} tickers)...")
        try:
            data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', progress=False, threads=True)
            for t in batch:
                try:
                    sd = data if len(batch)==1 else data[t]
                    if sd is not None and len(sd.dropna()) > 200: all_data[t] = sd.dropna()
                except: pass
        except Exception as e: print(f"  Error: {e}")
        time.sleep(1)
    print(f"  Downloaded {len(all_data)} stocks")

    print("\n[3/5] Processing indicators...")
    results = []
    for idx, (t, hist) in enumerate(all_data.items()):
        if (idx+1) % 50 == 0: print(f"  {idx+1}/{len(all_data)}...")
        r = process_stock(t, hist, sectors.get(t,'Unknown'), names.get(t,t))
        if r: results.append(r)
    results.sort(key=lambda x: x['trend_score'], reverse=True)
    print(f"  Processed {len(results)} stocks")

    print("\n[4/5] Adding price history (top 100)...")
    top100 = set(r['ticker'] for r in results[:100])
    for r in results:
        if r['ticker'] in top100 and r['ticker'] in all_data:
            hist = all_data[r['ticker']]; df = hist.copy()
            df.columns = [c.lower() for c in df.columns]
            if 'adj close' in df.columns: df['close'] = df['adj close']
            df['sma_50'] = calc_sma(df['close'],50); df['sma_200'] = calc_sma(df['close'],200)
            df['supertrend'], df['supertrend_dir'] = calc_supertrend(df['high'],df['low'],df['close'])
            df = df.dropna(subset=['sma_200'])
            if len(df) >= 60: r['price_history'] = get_price_history(df)

    # Summaries
    sec_data = {}
    for r in results:
        sec_data.setdefault(r['sector'],[]).append(r['trend_score'])
    sector_summary = {s:{'avg_score':round(np.mean(sc),1),'count':len(sc),
        'strong_buy':len([x for x in sc if x>=75]),'buy':len([x for x in sc if 60<=x<75]),
        'hold':len([x for x in sc if 40<=x<60]),'sell':len([x for x in sc if x<40])} for s,sc in sec_data.items()}
    asc = [r['trend_score'] for r in results]
    market_summary = {'total_stocks':len(results),'avg_score':round(np.mean(asc),1),
        'strong_buy_count':len([s for s in asc if s>=75]),'buy_count':len([s for s in asc if 60<=s<75]),
        'hold_count':len([s for s in asc if 40<=s<60]),'sell_count':len([s for s in asc if s<40]),
        'median_score':round(float(np.median(asc)),1)}

    print("\n[5/5] Saving...")
    os.makedirs('data', exist_ok=True)
    output = {'updated':datetime.now().strftime('%Y-%m-%d %H:%M'),'market_summary':market_summary,'sector_summary':sector_summary,'stocks':results}
    for path in ['data/trend_data.json','trend_data.json']:
        with open(path,'w') as f: json.dump(output, f, indent=2, cls=NumpyEncoder)
    update_trend_history(market_summary, sector_summary)

    print(f"\n✅ Done! {len(results)} stocks")
    print(f"   Avg Score: {market_summary['avg_score']} | SB:{market_summary['strong_buy_count']} B:{market_summary['buy_count']} H:{market_summary['hold_count']} S:{market_summary['sell_count']}")
    print(f"   Top 5: {', '.join(r['ticker'] for r in results[:5])}")

if __name__ == '__main__': main()
