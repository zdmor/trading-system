# -*- coding: utf-8 -*-
"""
高斯量能因子分层回测
每月第一个交易日：取全市场股票计算高斯因子 → 分5层 → 持有1个月 → 统计每层收益
预期的因子性质：反转因子 → Layer1(最低因子值)收益最高，Layer5(最高因子值)收益最低
输出: output_v2/factor_layer_results.json
"""
import sys, os, json, time, math
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_providers.tushare_provider import _get_pro


def calc_gaussian_volume(volumes, sigma=3.0, window=20):
    """计算高斯量能因子值"""
    if len(volumes) < window:
        return None
    vols = np.array(volumes[-window:], dtype=float)
    mu = np.mean(vols)
    std = np.std(vols)
    if std < 1e-9:
        return 0.0
    z_scores = (vols - mu) / std
    
    half = window // 2
    weights = np.array([math.exp(-0.5 * ((i - half) / sigma) ** 2) for i in range(window)])
    weights /= weights.sum()
    
    factor = np.dot(z_scores, weights)
    return float(factor)


def get_rebalance_dates(n_months=12):
    """获取每月第一个交易日"""
    pro = _get_pro()
    trade_cal = pro.trade_cal(exchange='SSE', is_open=1,
                              start_date='20250101', end_date='20260101')
    if trade_cal is None or trade_cal.empty:
        return []
    
    cal = trade_cal.sort_values('cal_date').reset_index(drop=True)
    months = {}
    for _, row in cal.iterrows():
        d = str(row['cal_date'])
        ym = d[:6]
        if ym not in months:
            months[ym] = d
    
    dates = sorted(months.values())
    # 取最近n_months个月（预留一个月做前视收益）
    if len(dates) > n_months + 1:
        dates = dates[-(n_months + 1):]
    
    print(f"再平衡月份: {len(dates)} 个")
    for d in dates:
        print(f"  {d}")
    return dates


def get_stock_pool():
    """获取全市场股票池（剔除北交所/ST/新股60天）"""
    pro = _get_pro()
    pool = []
    try:
        df = pro.stock_basic(exchange='', list_status='L',
                             fields='ts_code,symbol,name,list_date')
        if df is None or df.empty:
            return pool
        
        trade_cal = pro.trade_cal(exchange='SSE', is_open=1,
                                  start_date='20250801', end_date='20260101')
        today = str(trade_cal['cal_date'].iloc[-1])
        
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
    except Exception as e:
        print(f"  获取股票池失败: {e}")
    return pool


def bulk_fetch_data(pool, start_date='20250701', end_date='20251231', top_n=500):
    """批量拉取股票日线数据（一次拉取覆盖全部再平衡月份）"""
    pro = _get_pro()
    
    # 先筛选成交额TOP N
    print(f"筛选成交额TOP{top_n}...")
    amount_rank = []
    batch_size = 100
    total_check = min(len(pool), 2000)
    for bi in range(0, total_check, batch_size):
        batch = pool[bi:bi+batch_size]
        for stock in batch:
            try:
                df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date='20251220', end_date='20251231',
                               fields='amount')
                if df is not None and not df.empty:
                    avg = float(df['amount'].mean())
                    amount_rank.append({**stock, 'avg_amount': avg})
            except Exception:
                pass
        print(f"\r  筛选: {min(bi+batch_size, total_check)}/{total_check} 上榜: {len(amount_rank)}", end="", flush=True)
        time.sleep(0.2)
    
    amount_rank.sort(key=lambda x: x['avg_amount'], reverse=True)
    top_stocks = amount_rank[:top_n]
    print(f"\n  TOP{top_n}确认")
    
    # 批量获取日线
    stock_data = {}
    for i, stock in enumerate(top_stocks):
        try:
            df = pro.daily(adj='qfq', ts_code=stock['ts_code'], start_date=start_date, end_date=end_date,
                           fields='trade_date,close,vol,amount')
            if df is not None and not df.empty and len(df) >= 30:
                df = df.sort_values('trade_date').reset_index(drop=True)
                stock_data[stock['code']] = {
                    'ts_code': stock['ts_code'],
                    'name': stock['name'],
                    'dates': df['trade_date'].tolist(),
                    'close': df['close'].values.astype(float).tolist(),
                    'vol': df['vol'].values.astype(float).tolist(),
                }
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"\r  日线: {i+1}/{len(top_stocks)} 成功: {len(stock_data)}", end="", flush=True)
            time.sleep(0.3)
    
    print(f"\r  缓存: {len(stock_data)}只日线")
    return stock_data


def run_layer_backtest(n_months=12, n_layers=5, stocks_per_layer=100):
    """分层回测（一次缓存，批量计算）"""
    print(f"高斯量能因子分层回测 v2")
    print(f"{'='*60}")
    print(f"再平衡频率: 每月一次 | 分层: {n_layers} | 每层: {stocks_per_layer}")
    
    # 获取再平衡日期
    rebalance_dates = get_rebalance_dates(n_months + 1)
    if len(rebalance_dates) < 2:
        print("错误: 再平衡日期不足2个")
        return None
    
    # 获取股票池
    pool = get_stock_pool()
    if not pool:
        return None
    
    # 缓存全部数据
    cache_start = str(min(int(d) for d in rebalance_dates) - 100)[:8] + '01'
    cache_end = rebalance_dates[-1]
    stock_data = bulk_fetch_data(pool, start_date=cache_start, end_date=cache_end, top_n=500)
    
    if len(stock_data) < 100:
        print(f"缓存样本不足({len(stock_data)})")
        return None
    
    # 每个月的结果
    layer_monthly_returns = {i: [] for i in range(n_layers)}
    
    for mi in range(len(rebalance_dates) - 1):
        entry_date = rebalance_dates[mi]
        exit_date = rebalance_dates[mi + 1]
        print(f"\n[{mi+1}/{len(rebalance_dates)-1}] {entry_date} -> {exit_date}")
        
        # 从缓存计算每只股票的因子值
        stock_factors = []
        for code, data in stock_data.items():
            # 找到entry_date在数据中的位置
            try:
                dates = [str(d) for d in data['dates']]
                if entry_date in dates:
                    idx = dates.index(entry_date)
                else:
                    # 找最近的前一个交易日
                    idx = None
                    for j in range(len(dates) - 1, -1, -1):
                        if dates[j] <= entry_date:
                            idx = j
                            break
                    if idx is None or idx < 20:
                        continue
                
                if idx >= 20:
                    vol_window = data['vol'][idx-19:idx+1]
                    factor = calc_gaussian_volume(vol_window)
                    if factor is not None:
                        close = data['close'][idx]
                        stock_factors.append({
                            'code': code, 'name': data['name'],
                            'factor': factor, 'close': close,
                            'dates': dates, 'close_arr': data['close'],
                        })
            except Exception:
                continue
        
        if len(stock_factors) < n_layers * 10:
            print(f"  样本不足({len(stock_factors)})，跳过")
            continue
        
        stock_factors.sort(key=lambda x: x['factor'], reverse=True)
        per_layer = min(len(stock_factors) // n_layers, stocks_per_layer)
        layers = []
        for li in range(n_layers):
            layers.append(stock_factors[li * per_layer:(li + 1) * per_layer])
        
        for li, layer_stocks in enumerate(layers):
            avg_factor = np.mean([s['factor'] for s in layer_stocks])
            print(f"  L{li+1}({avg_factor:+.3f})", end="")
            
            # 算持有期收益(从缓存)
            layer_returns = []
            for stock in layer_stocks:
                try:
                    dates_s = [str(d) for d in stock['dates']]
                    if exit_date in dates_s:
                        exit_idx = dates_s.index(exit_date)
                    else:
                        exit_idx = None
                        for j in range(len(dates_s) - 1, -1, -1):
                            if dates_s[j] <= exit_date:
                                exit_idx = j
                                break
                        if exit_idx is None:
                            continue
                    exit_close = stock['close_arr'][exit_idx]
                    ret = (exit_close - stock['close']) / stock['close']
                    layer_returns.append(ret)
                except Exception:
                    pass
            
            if layer_returns:
                avg_ret = np.mean(layer_returns)
                layer_monthly_returns[li].append(avg_ret)
                print(f" {avg_ret:+.2%}  ")
            else:
                print(f" 无数据  ")
    
    # 汇总结果
    layer_summary = {}
    for li in range(n_layers):
        rets = layer_monthly_returns[li]
        if rets:
            layer_summary[f"Layer{li+1}"] = {
                "total_months": len(rets),
                "avg_monthly_return": round(float(np.mean(rets)), 6),
                "std_monthly_return": round(float(np.std(rets)), 6),
                "cumulative_return": round(float(np.prod([1 + r for r in rets]) - 1), 6),
                "monthly_returns": [round(float(r), 6) for r in rets],
                "factor_description": (
                    "最高因子值(放量最异常)" if li == 0 else
                    "较高因子值" if li == 1 else
                    "中等因子值" if li == 2 else
                    "较低因子值" if li == 3 else
                    "最低因子值(量能正常)"
                ),
            }
    
    # 单调性检验
    avg_rets = [layer_summary[f"Layer{i+1}"]["avg_monthly_return"] for i in range(n_layers)
                if f"Layer{i+1}" in layer_summary]
    monotonically_decreasing = all(avg_rets[i] >= avg_rets[i + 1] for i in range(len(avg_rets) - 1))
    monotonically_increasing = all(avg_rets[i] <= avg_rets[i + 1] for i in range(len(avg_rets) - 1))
    monotonic = "下降(反转)" if monotonically_decreasing else ("上升(动量)" if monotonically_increasing else "无单调")
    
    top_bottom_diff = avg_rets[0] - avg_rets[-1] if len(avg_rets) >= 2 else 0
    
    result = {
        "factor": "GaussianVolume",
        "method": "月度分层回测",
        "n_layers": n_layers,
        "stocks_per_layer": stocks_per_layer,
        "n_months": len(rebalance_dates) - 1,
        "rebalance_dates": rebalance_dates,
        "monotonicity": monotonic,
        "top_bottom_diff": round(float(top_bottom_diff), 6),
        "layers": layer_summary,
    }
    
    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), 'output_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'factor_layer_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"结果已保存: {out_path}")
    print(f"{'='*60}")
    print(f"分层单调性: {monotonic}")
    print(f"Layer1(高因子) vs Layer{n_layers}(低因子) 差值: {top_bottom_diff:+.2%}")
    for li in range(n_layers):
        key = f"Layer{li+1}"
        if key in layer_summary:
            ls = layer_summary[key]
            print(f"  {key}: 月均{ls['avg_monthly_return']:+.2%} 标准差{ls['std_monthly_return']:.2%} 累计{ls['cumulative_return']:+.2%}")
    
    # 自检
    print(f"\n--- 自检 ---")
    checks = []
    if len(avg_rets) >= 2:
        checks.append(("✓" if abs(top_bottom_diff) > 0.001 else "✗",
                       f"Top-Bottom差异显著: {top_bottom_diff:+.2%}"))
    checks.append(("✓" if monotonic != "无单调" else "⚠",
                   f"单调性: {monotonic}"))
    for status, msg in checks:
        print(f"  {status} {msg}")
    
    return result


if __name__ == "__main__":
    result = run_layer_backtest(n_months=12, n_layers=5, stocks_per_layer=100)
    if result is None:
        sys.exit(1)
