#!/usr/bin/env python3
"""
S&P 500 Trend Following Dashboard v3
=====================================
- 5-year backtest
- Strategy fitness scoring & ranking
- Yearly returns breakdown
- Equity curve (10-day sampling + trade events)
- Trade markers for chart overlay
- Daily market history accumulation
"""

import json, os, sys, time, warnings
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
    return list(SP500.keys()), {t:v["sector"] for t,v in SP500.items()}, {t:v["name"] for t,v in SP500.items()}

# ── 2. INDICATORS ──
def calc_sma(s,p): return s.rolling(window=p,min_periods=p).mean()
def calc_ema(s,p): return s.ewm(span=p,adjust=False).mean()

def calc_rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    return 100-(100/(1+g.ewm(alpha=1/p,min_periods=p).mean()/l.ewm(alpha=1/p,min_periods=p).mean()))

def calc_macd(s,f=12,sl=26,sg=9):
    ml=calc_ema(s,f)-calc_ema(s,sl); return ml,calc_ema(ml,sg),ml-calc_ema(ml,sg)

def calc_atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(window=p,min_periods=p).mean()

def calc_adx(h,l,c,p=14):
    pdm=h.diff(); mdm=-l.diff()
    pdm[pdm<0]=0; mdm[mdm<0]=0; pdm[pdm<mdm]=0; mdm[mdm<pdm]=0
    atr=calc_atr(h,l,c,p)
    pdi=100*calc_ema(pdm,p)/atr; mdi=100*calc_ema(mdm,p)/atr
    return calc_ema(100*(pdi-mdi).abs()/(pdi+mdi),p)

def calc_supertrend(h,l,c,p=10,m=3):
    atr=calc_atr(h,l,c,p); hl2=(h+l)/2; ub=hl2+m*atr; lb=hl2-m*atr
    st=pd.Series(index=c.index,dtype=float); dr=pd.Series(index=c.index,dtype=float)
    st.iloc[0]=ub.iloc[0]; dr.iloc[0]=-1
    for i in range(1,len(c)):
        if c.iloc[i]>st.iloc[i-1]:
            st.iloc[i]=max(lb.iloc[i],st.iloc[i-1]) if dr.iloc[i-1]==1 else lb.iloc[i]; dr.iloc[i]=1
        else:
            st.iloc[i]=min(ub.iloc[i],st.iloc[i-1]) if dr.iloc[i-1]==-1 else ub.iloc[i]; dr.iloc[i]=-1
    return st,dr

# ── 3. TREND SCORE ──
def calc_trend_score(row):
    s=0
    if row['close']>row['sma_200']:s+=15
    if row['close']>row['sma_50']:s+=10
    if row['sma_50']>row['sma_200']:s+=15
    if row['ema_10']>row['ema_30']:s+=10
    if row['macd']>row['macd_signal']:s+=10
    if row['rsi']>50:s+=10
    if row['adx']>25:s+=15
    if row['adx']>40:s+=5
    if row['supertrend_dir']==1:s+=10
    return s

# ── 4. PROFIT SIGNALS ──
def calc_profit_signals(df):
    latest=df.iloc[-1]; cl=latest['close']; atr=latest['atr']
    scores=df['trend_score']; ep=None; ed=None
    for i in range(len(scores)-1,0,-1):
        if scores.iloc[i]>=60 and scores.iloc[i-1]<60:
            ep=df['close'].iloc[i]; ed=df.index[i]; break
    if ep is None:
        ab=df[df['trend_score']>=60]
        if len(ab)>0: ep=ab['close'].iloc[0]; ed=ab.index[0]
        else: ep=cl; ed=df.index[-1]
    rh=df['high'].rolling(20).max().iloc[-1]; atr_stop=rh-2*atr
    return {
        'entry_price':round(ep,2),'entry_date':str(ed.date()) if hasattr(ed,'date') else str(ed)[:10],
        'atr_trailing_stop':round(atr_stop,2),'stop_distance_pct':round(((atr_stop-cl)/cl)*100,2),
        'gain_since_signal':round(((cl-ep)/ep)*100,2),'days_in_trend':(df.index[-1]-ed).days if ed else 0,
        'profit_zone_10':round(ep*1.10,2),'profit_zone_20':round(ep*1.20,2),'profit_zone_30':round(ep*1.30,2),
        'reached_10':cl>=ep*1.10,'reached_20':cl>=ep*1.20,'reached_30':cl>=ep*1.30,
        'ema10_exit_warning':cl<latest['ema_10'],'score_drop_warning':latest['trend_score']<40
    }

# ── 5. BACKTEST (5 YEAR) ──
def run_backtest(df):
    trades=[]; markers=[]; in_pos=False; ep=0; ed=None
    cap=100.0; equity=[]; bh_start=float(df['close'].iloc[0])
    trade_dates=set()

    for i in range(len(df)):
        sc=df['trend_score'].iloc[i]; cl=float(df['close'].iloc[i])
        dt=df.index[i]; ds=str(dt.date()) if hasattr(dt,'date') else str(dt)[:10]
        is_trade=False

        if not in_pos and sc>=60 and i>0:
            in_pos=True; ep=cl; ed=dt; is_trade=True
            markers.append({'type':'BUY','date':ds,'price':round(cl,2),'score':int(sc)})
            trade_dates.add(i)
        elif in_pos and sc<40:
            in_pos=False; pnl=((cl-ep)/ep)*100; cap*=(1+pnl/100); is_trade=True
            markers.append({'type':'SELL','date':ds,'price':round(cl,2),'score':int(sc),'pnl_pct':round(pnl,2),'days_held':(dt-ed).days})
            trades.append({'entry_date':str(ed.date()) if hasattr(ed,'date') else str(ed)[:10],'exit_date':ds,
                'entry_price':round(ep,2),'exit_price':round(cl,2),'pnl_pct':round(pnl,2)})
            trade_dates.add(i)

        mtm=cap*(cl/ep) if in_pos and ep>0 else cap
        bh=(cl/bh_start)*100

        # 10-day sampling + trade events + first/last
        if i%10==0 or i==len(df)-1 or is_trade:
            equity.append({'d':ds,'s':round(mtm,2),'b':round(bh,2)})

    # Close open position
    if in_pos:
        cl=float(df['close'].iloc[-1]); pnl=((cl-ep)/ep)*100
        trades.append({'entry_date':str(ed.date()) if hasattr(ed,'date') else str(ed)[:10],
            'exit_date':str(df.index[-1].date()) if hasattr(df.index[-1],'date') else str(df.index[-1])[:10],
            'entry_price':round(ep,2),'exit_price':round(cl,2),'pnl_pct':round(pnl,2),'open':True})

    if not trades:
        return {'total_trades':0,'win_rate':0,'avg_winner':0,'avg_loser':0,'total_return':0,
            'max_drawdown':0,'buy_hold_return':0,'profit_factor':0,'win_loss_ratio':0,
            'trades':[],'trade_markers':[],'equity_curve':equity,'yearly_returns':{}}

    w=[t['pnl_pct'] for t in trades if t['pnl_pct']>0]
    l=[t['pnl_pct'] for t in trades if t['pnl_pct']<=0]
    bhr=((df['close'].iloc[-1]-df['close'].iloc[0])/df['close'].iloc[0])*100
    cum=(1+df['close'].pct_change()).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()*100
    total_w=sum(w) if w else 0; total_l=abs(sum(l)) if l else 0
    pf=round(total_w/total_l,2) if total_l>0 else 999
    wlr=round(abs(np.mean(w)/np.mean(l)),2) if l and np.mean(l)!=0 else 999

    # Yearly returns
    yearly={}
    for yr in df.index.year.unique():
        yr_df=df[df.index.year==yr]
        if len(yr_df)<20: continue
        # Strategy return for this year
        yr_trades=[t for t in trades if int(t.get('entry_date','0')[:4])<=yr and int(t.get('exit_date','9999')[:4])>=yr]
        # Simple: use equity curve
        yr_eq=[e for e in equity if e['d'][:4]==str(yr)]
        if yr_eq:
            strat_ret=round((yr_eq[-1]['s']/yr_eq[0]['s']-1)*100,1) if yr_eq[0]['s']>0 else 0
            bh_ret=round((yr_eq[-1]['b']/yr_eq[0]['b']-1)*100,1) if yr_eq[0]['b']>0 else 0
            yearly[str(yr)]={'strategy':strat_ret,'buy_hold':bh_ret}

    return {
        'total_trades':len(trades),'win_rate':round(len(w)/len(trades)*100,1) if trades else 0,
        'avg_winner':round(float(np.mean(w)),2) if w else 0,'avg_loser':round(float(np.mean(l)),2) if l else 0,
        'total_return':round(float(sum(t['pnl_pct'] for t in trades)),2),'max_drawdown':round(float(mdd),2),
        'buy_hold_return':round(float(bhr),2),'profit_factor':pf,'win_loss_ratio':wlr,
        'excess_return':round(float(sum(t['pnl_pct'] for t in trades))-float(bhr),2),
        'trades':trades[-8:],'trade_markers':markers,'equity_curve':equity,'yearly_returns':yearly
    }

# ── 6. STRATEGY FITNESS ──
def calc_fitness(bt):
    """Strategy fitness score: excess_return 30% + win_loss_ratio 20% + win_rate 20% + total_return 20% + mdd 10%"""
    if bt['total_trades']<2: return 0

    # Normalize each metric to 0-100
    er=bt.get('excess_return',0)
    er_score=min(100,max(0,(er+50)/150*100))  # -50%~+100% -> 0~100

    wlr=min(bt.get('win_loss_ratio',0),5)
    wlr_score=wlr/5*100  # 0~5 -> 0~100

    wr=bt.get('win_rate',0)
    wr_score=wr  # already 0-100

    tr=bt.get('total_return',0)
    tr_score=min(100,max(0,(tr+50)/250*100))  # -50%~+200% -> 0~100

    mdd=abs(bt.get('max_drawdown',0))
    mdd_score=max(0,100-mdd*2)  # 0%=100, 50%=0

    fitness=er_score*0.30+wlr_score*0.20+wr_score*0.20+tr_score*0.20+mdd_score*0.10
    return round(fitness,1)

# ── 7. PROCESS STOCK ──
def process_stock(ticker, hist, sector, name):
    try:
        if hist is None or len(hist)<250: return None
        df=hist.copy(); df.columns=[c.lower() for c in df.columns]
        if 'adj close' in df.columns: df['close']=df['adj close']

        df['sma_50']=calc_sma(df['close'],50); df['sma_200']=calc_sma(df['close'],200)
        df['ema_10']=calc_ema(df['close'],10); df['ema_30']=calc_ema(df['close'],30)
        df['rsi']=calc_rsi(df['close'])
        df['macd'],df['macd_signal'],df['macd_hist']=calc_macd(df['close'])
        df['atr']=calc_atr(df['high'],df['low'],df['close'])
        df['adx']=calc_adx(df['high'],df['low'],df['close'])
        df['supertrend'],df['supertrend_dir']=calc_supertrend(df['high'],df['low'],df['close'])
        df=df.dropna(subset=['sma_200','adx'])
        if len(df)<50: return None

        df['trend_score']=df.apply(calc_trend_score,axis=1)
        latest=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else latest
        score=int(latest['trend_score'])
        signal="STRONG_BUY" if score>=75 else "BUY" if score>=60 else "HOLD" if score>=40 else "SELL"

        bt=run_backtest(df)
        fitness=calc_fitness(bt)

        return {
            'ticker':ticker,'name':name,'sector':sector,
            'price':round(float(latest['close']),2),
            'change_pct':round(float((latest['close']-prev['close'])/prev['close']*100),2),
            'volume':int(latest['volume']),'trend_score':score,
            'score_change':int(latest['trend_score']-prev['trend_score']),
            'signal':signal,'strategy_fitness':fitness,
            'high_52w':round(float(df['high'].tail(252).max()),2),
            'low_52w':round(float(df['low'].tail(252).min()),2),
            'pct_from_high':round(float((latest['close']-df['high'].tail(252).max())/df['high'].tail(252).max()*100),2),
            'indicators':{
                'sma_50':round(float(latest['sma_50']),2),'sma_200':round(float(latest['sma_200']),2),
                'ema_10':round(float(latest['ema_10']),2),'ema_30':round(float(latest['ema_30']),2),
                'rsi':round(float(latest['rsi']),1),'adx':round(float(latest['adx']),1),
                'macd':round(float(latest['macd']),3),'macd_signal':round(float(latest['macd_signal']),3),
                'supertrend_dir':'UP' if latest['supertrend_dir']==1 else 'DOWN',
                'atr':round(float(latest['atr']),2)
            },
            'profit_taking':calc_profit_signals(df),
            'backtest':bt
        }
    except Exception as e:
        print(f"  Error {ticker}: {e}"); return None

# ── 8. TREND HISTORY ──
def update_trend_history(ms,ss):
    hf='data/trend_history.json'; h=[]
    if os.path.exists(hf):
        try:
            with open(hf,'r') as f: h=json.load(f)
        except: h=[]
    today=datetime.now().strftime('%Y-%m-%d')
    h=[x for x in h if x['date']!=today]
    h.append({'date':today,'avg_score':ms['avg_score'],'median_score':ms['median_score'],
        'strong_buy':ms['strong_buy_count'],'buy':ms['buy_count'],'hold':ms['hold_count'],
        'sell':ms['sell_count'],'sectors':{s:d['avg_score'] for s,d in ss.items()}})
    h=h[-365:]
    for p in ['data/trend_history.json','trend_history.json']:
        with open(p,'w') as f: json.dump(h,f,indent=2,cls=NumpyEncoder)
    print(f"  History: {len(h)} days")

# ── 9. MAIN ──
def main():
    print("="*60)
    print("S&P 500 Trend Following v3 (5-Year Backtest)")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\n[1/4] Loading tickers...")
    tickers,sectors,names=get_sp500_tickers()
    print(f"  {len(tickers)} tickers")
    if not tickers: print("ERROR"); sys.exit(1)

    print("\n[2/4] Downloading 5 years of data...")
    end_date=datetime.now(); start_date=end_date-timedelta(days=1900)
    all_data={}
    for i in range(0,len(tickers),100):
        batch=tickers[i:i+100]
        print(f"  Batch {i//100+1}/{(len(tickers)-1)//100+1} ({len(batch)})...")
        try:
            data=yf.download(batch,start=start_date,end=end_date,group_by='ticker',progress=False,threads=True)
            for t in batch:
                try:
                    sd=data if len(batch)==1 else data[t]
                    if sd is not None and len(sd.dropna())>250: all_data[t]=sd.dropna()
                except: pass
        except Exception as e: print(f"  Error: {e}")
        time.sleep(1)
    print(f"  Downloaded {len(all_data)} stocks")

    print("\n[3/4] Processing (5Y backtest + fitness)...")
    results=[]
    for idx,(t,hist) in enumerate(all_data.items()):
        if (idx+1)%50==0: print(f"  {idx+1}/{len(all_data)}...")
        r=process_stock(t,hist,sectors.get(t,'Unknown'),names.get(t,t))
        if r: results.append(r)

    # Fitness ranking
    results.sort(key=lambda x:x['strategy_fitness'],reverse=True)
    for i,r in enumerate(results): r['fitness_rank']=i+1

    print(f"  Processed {len(results)} stocks")

    # Summaries
    sec_data={}
    for r in results: sec_data.setdefault(r['sector'],[]).append(r['trend_score'])
    sector_summary={s:{'avg_score':round(np.mean(sc),1),'count':len(sc),
        'strong_buy':len([x for x in sc if x>=75]),'buy':len([x for x in sc if 60<=x<75]),
        'hold':len([x for x in sc if 40<=x<60]),'sell':len([x for x in sc if x<40])} for s,sc in sec_data.items()}
    asc=[r['trend_score'] for r in results]
    market_summary={'total_stocks':len(results),'avg_score':round(np.mean(asc),1),
        'strong_buy_count':len([s for s in asc if s>=75]),'buy_count':len([s for s in asc if 60<=s<75]),
        'hold_count':len([s for s in asc if 40<=s<60]),'sell_count':len([s for s in asc if s<40]),
        'median_score':round(float(np.median(asc)),1)}

    print("\n[4/4] Saving...")
    os.makedirs('data',exist_ok=True)
    output={'updated':datetime.now().strftime('%Y-%m-%d %H:%M'),'market_summary':market_summary,
        'sector_summary':sector_summary,'stocks':results}
    for p in ['data/trend_data.json','trend_data.json']:
        with open(p,'w') as f: json.dump(output,f,cls=NumpyEncoder)
    update_trend_history(market_summary,sector_summary)

    top5=results[:5]
    print(f"\n✅ {len(results)} stocks processed")
    print(f"   Avg Score: {market_summary['avg_score']}")
    tickers = ', '.join(r['ticker'] + '(' + str(r['strategy_fitness']) + ')' for r in top5)
    print(f"   Top 5 Fitness: {tickers}")

if __name__=='__main__': main()
