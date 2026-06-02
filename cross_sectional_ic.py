# -*- coding: utf-8 -*-
"""
跨截面Rank IC验证 v2 — 一次拉取100天日线，批量计算20个截面的IC
大幅减少Tushare API调用次数
输出: output_v2/cross_sectional_ic.json
"""
import sys, os, json, time, math
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_providers.tushare_provider import _get_pro


def calc_gaussian_volume(volumes, sigma=3.0, window=20):
    """计算高斯量能因子值"""
    if len(volumes) < window:
        return None
    vols = np.array(volumes, dtype=float)
    mu = np.mean(vols)
    std = np.std(vols)
    if std < 1e-9:
        return 0.0
    z_scores = (vols - mu) / std
    end = window - 1  # 权重中心在窗口末端（最新数据点），而非中点
    weights = np.array([math.exp(-0.5 * ((i - end) / sigma) ** 2) for i in range(window)])
    weights /= weights.sum()
    return float(np.dot(z_scores, weights))


def get_stock_pool():
    """获取全市场股票池"""
    pro = _get_pro()
    df = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,symbol,name,list_date')
    if df is None or df.empty:
        return []
    
    trade_cal = pro.trade_cal(exchange='SSE', is_open=1,
                              start_date='20250801', end_date='20260101')
    if trade_cal is None or trade_cal.empty:
        return []
    today = str(trade_cal['cal_date'].iloc[-1])
    
    pool = []
    for _, row in df.iterrows():
        code = str(row['symbol']).zfill(6)
        name = str(row['name'])
        ts_code = str(row['ts_code'])
        list_date = str(row['list_date'])
        if code.startswith('8') or '.BJ' in ts_code:
            continue
        if 'ST' in name.upper() or '退' in name:
            continue
        if list_date and list_date != 'nan':
            if int(today) - int(list_date) < 60:
                continue
        pool.append({'code': code, 'name': name, 'ts_code': ts_code})
    return pool


def bulk_fetch_data(pool, lookback_days=100, top_n=500):
    """批量拉取股票日线（只拉成交额TOP N）"""
    pro = _get_pro()
    
    # 先筛选成交额TOP N（用最近一天）
    from datetime import datetime, timedelta
    end_dt = datetime(2025, 12, 25)
    start_dt = end_dt - timedelta(days=5)
    end_str = '20251225'
    start_str = start_dt.strftime('%Y%m%d')
    
    print(f"筛选成交额TOP N...")
    amount_rank = []
    batch_size = 50
    total_check = min(len(pool), 2000)
    for bi in range(0, total_check, batch_size):
        batch = pool[bi:bi+batch_size]
        for stock in batch:
            try:
                df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date=start_str, end_date=end_str,
                               fields='amount')
                if df is not None and not df.empty:
                    avg_amount = float(df['amount'].mean())
                    amount_rank.append({**stock, 'avg_amount': avg_amount})
            except Exception:
                pass
        print(f"\r  筛选: {min(bi+batch_size, total_check)}/{total_check} 上榜: {len(amount_rank)}", end="", flush=True)
        time.sleep(0.15)
    
    amount_rank.sort(key=lambda x: x['avg_amount'], reverse=True)
    top_stocks = amount_rank[:top_n]
    print(f"\n  成交额TOP{top_n}: {len(top_stocks)}只")
    
    # 批量获取日线（每只股票一次API调用，拉100天）
    start_dt_full = end_dt - timedelta(days=lookback_days + 30)
    start_str_full = start_dt_full.strftime('%Y%m%d')
    
    stock_data = {}
    for i, stock in enumerate(top_stocks):
        try:
            df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date=start_str_full, end_date=end_str,
                           fields='trade_date,close,vol,amount')
            if df is not None and not df.empty and len(df) >= 30:
                df = df.sort_values('trade_date').reset_index(drop=True)
                stock_data[stock['code']] = {
                    'ts_code': stock['ts_code'],
                    'name': stock['name'],
                    'dates': df['trade_date'].tolist(),
                    'close': df['close'].values.astype(float).tolist(),
                    'vol': df['vol'].values.astype(float).tolist(),
                    'amount': df['amount'].values.astype(float).tolist(),
                }
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"\r  拉取日线: {i+1}/{top_n} 成功: {len(stock_data)}", end="", flush=True)
            time.sleep(0.2)
    
    print(f"\n  缓存完成: {len(stock_data)}只股票日线数据")
    return stock_data


def run_cross_section_ic_fast(cross_dates=20, top_n=500):
    """快速截面IC — 从已缓存数据计算"""
    print(f"跨截面Rank IC验证 v2 — 高斯量能因子")
    print(f"{'='*60}")
    
    pool = get_stock_pool()
    print(f"全市场股票池: {len(pool)} 只")
    
    # 获取缓存数据
    stock_data = bulk_fetch_data(pool, lookback_days=100, top_n=top_n)
    
    if len(stock_data) < 100:
        print(f"错误: 缓存样本不足({len(stock_data)})")
        return None
    
    # 获取截面日期（从交易日历）
    pro = _get_pro()
    trade_cal = pro.trade_cal(exchange='SSE', is_open=1,
                              start_date='20250801', end_date='20251231')
    all_dates = sorted(trade_cal['cal_date'].tolist())
    
    # 每隔5个交易日取一个截面日期
    step = max(1, len(all_dates) // cross_dates)
    sample_dates = [all_dates[i] for i in range(0, len(all_dates), step)][-cross_dates:]
    
    print(f"截面日期: {len(sample_dates)} 个")
    
    ic_series = []
    date_details = []
    
    for di, cross_date in enumerate(sample_dates):
        print(f"\n[{di+1}/{len(sample_dates)}] {cross_date}", end="", flush=True)
        
        factors = []
        returns = []
        
        for code, data in stock_data.items():
            # 找到该日期在数据中的位置
            date_str = str(cross_date)
            if date_str not in data['dates']:
                # 找最近的前一个交易日
                for d in reversed(data['dates']):
                    if str(d) <= date_str:
                        # 用这个日期作为尾日
                        idx = data['dates'].index(d)
                        break
                else:
                    continue
            else:
                idx = data['dates'].index(date_str)
            
            # 计算因子（用当前日期前的20个成交量）
            if idx >= 20:
                vol_window = data['vol'][idx-19:idx+1]
                factor = calc_gaussian_volume(vol_window)
                if factor is None:
                    continue
                
                close = data['close'][idx]
                
                # 前视5日收益
                fwd_idx = idx + 5
                if fwd_idx < len(data['close']):
                    fwd_close = data['close'][fwd_idx]
                    fwd_return = (fwd_close - close) / close
                    
                    factors.append(factor)
                    returns.append(fwd_return)
        
        if len(factors) < 30:
            print(f"  样本不足({len(factors)})，跳过")
            continue
        
        # Rank IC (Spearman = Pearson on ranks)
        from scipy.stats import rankdata
        rank_f = rankdata(factors)
        rank_r = rankdata(returns)
        
        if np.std(rank_f) > 1e-12 and np.std(rank_r) > 1e-12:
            ic = float(np.corrcoef(rank_f, rank_r)[0, 1])
        else:
            ic = 0.0
        
        ic = 0.0 if math.isnan(ic) else ic
        ic_series.append(ic)
        
        pos_ratio = sum(1 for r in returns if r > 0) / len(returns) * 100
        print(f"  IC:{ic:+.4f} n={len(factors)} pos%={pos_ratio:.0f}%")
        
        date_details.append({
            'date': cross_date,
            'ic': round(ic, 4),
            'n_stocks': len(factors),
            'factor_mean': round(float(np.mean(factors)), 4),
            'pos_return_pct': round(pos_ratio, 1),
        })
    
    if not ic_series:
        print("错误: 无有效IC")
        return None
    
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos_ratio = sum(1 for ic in ic_series if ic > 0) / len(ic_series) * 100
    
    result = {
        "factor": "GaussianVolume",
        "method": "截面RankIC",
        "sample_size": top_n,
        "cross_section_dates": len(ic_series),
        "ic_mean": round(float(ic_mean), 4),
        "ic_std": round(float(ic_std), 4),
        "ir": round(float(ir), 4),
        "ic_positive_ratio": round(float(ic_pos_ratio), 1),
        "ic_series": [round(float(ic), 4) for ic in ic_series],
        "date_details": date_details,
    }
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cross_sectional_ic.json')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"IC均值: {ic_mean:+.4f}")
    print(f"IC标准差: {ic_std:.4f}")
    print(f"IR: {ir:+.4f}")
    print(f"IC>0胜率: {ic_pos_ratio:.1f}%")
    print(f"保存: {out_path}")
    
    print(f"\n--- 自检 ---")
    print(f"  {'✓' if len(ic_series)>=15 else '✗'} 截面日期数: {len(ic_series)}")
    print(f"  {'✓' if -2<=ir<=2 else '✗'} IR: {ir:.2f}")
    
    return result


if __name__ == "__main__":
    result = run_cross_section_ic_fast(cross_dates=20, top_n=500)
    if result is None:
        sys.exit(1)