"""
收盘大盘快照 — 自动抓取大盘状态并输出 Markdown，可追加到工作日志

数据源:
  - 指数行情: AKShare stock_zh_index_spot_em (上证/深证/创业板/科创50)
  - 涨跌比/涨跌停: AKShare stock_zh_a_spot_em (全市场个股实时行情)
  - 板块Top5/Bottom5: AKShare stock_board_industry_name_em
  - 北向资金: Tushare Pro moneyflow_hsgt / AKShare stock_hsgt_north_net_flow_in_em

用法:
  python market_snapshot.py               # 打印到终端
  python market_snapshot.py >> work.md    # 追加到工作日志

输出格式:
  ## 大盘快照 YYYY-MM-DD 收盘
  | 指数 | 收盘价 | 涨跌幅 |
  | ... | ... | ... |
  涨跌比: xxx/xxx | 涨停xx 跌停xx
  板块Top5: ...
  板块Bottom5: ...
  北向资金: +xx亿
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, date
from typing import Optional

import tushare as ts

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 线程超时辅助（复用项目已有模式） ──
_POOL = ThreadPoolExecutor(max_workers=1)

def _ak_call(timeout: int, func) -> Optional:
    """在线程中执行 func，超时返回 None"""
    fut = _POOL.submit(func)
    try:
        return fut.result(timeout=timeout)
    except (TimeoutError, Exception):
        fut.cancel()
        return None


# ── Tushare Pro 初始化（复用项目统一 token 路径） ──
_TPRO = None

def _get_tushare_pro():
    global _TPRO
    if _TPRO is None:
        cfg_path = os.path.join(SCRIPT_DIR, "config.json")
        token = ""
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            token = cfg.get("_tushare", {}).get("token", "")
        except Exception:
            pass
        if not token:
            return None
        ts.set_token(token)
        _TPRO = ts.pro_api()
    return _TPRO


# ══════════════════════════════════════════
# 数据获取函数（各自独立 try/except）
# ══════════════════════════════════════════

def get_index_data() -> list:
    """获取主要指数行情，返回 [{name, price, chg_pct}, ...]"""
    target_indices = {
        "上证指数": "sh000001",
        "深证成指": "sz399001",
        "创业板指": "sz399006",
        "科创50":   "sh000688",
    }
    try:
        import akshare as ak
        df = _ak_call(10, lambda: ak.stock_zh_index_spot_em())
        if df is None or df.empty:
            return []
        rows = []
        for _, row in df.iterrows():
            name = str(row.get("名称", ""))
            if name in target_indices:
                price = row.get("最新价", row.get("现价", None))
                chg_pct = row.get("涨跌幅", None)
                if price is not None:
                    rows.append({
                        "name": name,
                        "price": float(price),
                        "chg_pct": float(chg_pct) if chg_pct is not None else None,
                    })
        return rows
    except Exception as e:
        print(f"  [WARN] 指数行情获取失败: {e}", file=sys.stderr)
        return []


def get_breadth() -> dict:
    """从全市场快照计算涨跌比、涨跌停（复用 scanner._calc_breadth 逻辑）"""
    try:
        import akshare as ak
        df = _ak_call(15, lambda: ak.stock_zh_a_spot_em())
        if df is None or df.empty:
            return {}
        up = down = limit_up = limit_down = 0
        total_count = 0
        for _, row in df.iterrows():
            chg = row.get("涨跌幅", None)
            if chg is None:
                continue
            chg = float(chg)
            total_count += 1
            if chg > 0:
                up += 1
            elif chg < 0:
                down += 1
            if chg >= 9.8:
                limit_up += 1
            elif chg <= -9.8:
                limit_down += 1
        return {
            "up": up,
            "down": down,
            "total": total_count,
            "limit_up": limit_up,
            "limit_down": limit_down,
        }
    except Exception as e:
        print(f"  [WARN] 涨跌比获取失败: {e}", file=sys.stderr)
        return {}


def get_sector_performance(top_n: int = 5) -> tuple:
    """获取行业板块涨跌幅 Top5 和 Bottom5，返回 (top_list, bottom_list)"""
    try:
        import akshare as ak
        df = _ak_call(15, lambda: ak.stock_board_industry_name_em())
        if df is None or df.empty:
            return [], []
        # 提取名称+涨跌幅
        records = []
        for _, row in df.iterrows():
            name = str(row.get("板块名称", ""))
            chg = row.get("涨跌幅", None)
            if name and chg is not None:
                records.append((name, float(chg)))
        records.sort(key=lambda x: x[1], reverse=True)
        top = records[:top_n]
        bottom = records[-top_n:][::-1]  # 跌幅最大的排前面
        return top, bottom
    except Exception as e:
        print(f"  [WARN] 板块数据获取失败: {e}", file=sys.stderr)
        return [], []


def get_northbound_flow() -> Optional[float]:
    """获取北向资金当日净流入（亿元），优先 Tushare，降级 AKShare"""
    # 方案A: Tushare moneyflow_hsgt
    today_str = date.today().strftime("%Y%m%d")
    pro = _get_tushare_pro()
    if pro is not None:
        try:
            df = pro.moneyflow_hsgt(start_date=today_str, end_date=today_str)
            if df is not None and not df.empty and "north_money" in df.columns:
                val = float(df.iloc[0]["north_money"])
                if val != 0:
                    # north_money 单位是万元，转亿元
                    return round(val / 1e4, 2)
        except Exception as e:
            print(f"  [WARN] Tushare 北向资金获取失败: {e}", file=sys.stderr)

    # 方案B: AKShare 沪股通+深股通合计
    try:
        import akshare as ak
        sh = sz = 0
        df_sh = _ak_call(10, lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="沪股通"))
        if df_sh is not None and not df_sh.empty:
            sh = float(df_sh.iloc[-1].get("value", 0))
        df_sz = _ak_call(10, lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="深股通"))
        if df_sz is not None and not df_sz.empty:
            sz = float(df_sz.iloc[-1].get("value", 0))
        total = sh + sz
        if total != 0:
            return round(total, 2)
    except Exception as e:
        print(f"  [WARN] AKShare 北向资金获取失败: {e}", file=sys.stderr)

    return None


def get_trade_cal() -> Optional[str]:
    """取最近交易日字符串 YYYYMMDD，判断今天是否交易日"""
    today = date.today()
    today_str = today.strftime("%Y%m%d")
    pro = _get_tushare_pro()
    if pro is None:
        return today_str  # 无法判断，默认当天

    try:
        df = pro.trade_cal(start_date=today_str, end_date=today_str)
        if df is not None and not df.empty:
            is_open = df.iloc[0].get("is_open", 0)
            if is_open == 1:
                return today_str
        # 不是交易日 -> 找最近交易日
        # trade_cal 返回降序（最新在前）
        df = pro.trade_cal(start_date="20260101", end_date=today_str)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if row.get("is_open", 0) == 1:
                    return str(row["cal_date"])
    except Exception as e:
        print(f"  [WARN] 交易日历获取失败: {e}", file=sys.stderr)

    return today_str


# ══════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════

def build_snapshot() -> str:
    """获取所有数据并格式化为 Markdown 字符串"""
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    weekday = today.weekday()  # 0=Mon ... 6=Sun

    # 判断是否为交易日/盘中
    trade_date = get_trade_cal()
    is_trade_day = (trade_date == today.strftime("%Y%m%d"))
    is_weekend = weekday >= 5

    now = datetime.now()
    # 简单判断：15:00 之前算盘中
    is_closing = now.hour >= 15

    # 收集所有数据（并行）
    index_rows = get_index_data()
    breadth = get_breadth()
    top_sectors, bottom_sectors = get_sector_performance(5)
    north_flow = get_northbound_flow()

    # ── 组装输出 ──
    lines = []

    # 标题
    status = "收盘" if (is_closing and is_trade_day) else ("非交易日" if is_weekend or not is_trade_day else "盘中数据")
    lines.append(f"## 大盘快照 {date_str} {status}")
    lines.append("")

    # 1) 指数表格
    lines.append("| 指数 | 收盘价 | 涨跌幅 |")
    lines.append("|------|--------|--------|")
    if index_rows:
        for r in index_rows:
            price_str = f"{r['price']:.2f}"
            if r["chg_pct"] is not None:
                chg = r["chg_pct"]
                sign = "+" if chg >= 0 else ""
                chg_str = f"{sign}{chg:.2f}%"
            else:
                chg_str = "--"
            lines.append(f"| {r['name']} | {price_str} | {chg_str} |")
    else:
        lines.append("| -- | -- | -- |")

    lines.append("")

    # 2) 涨跌比 & 涨跌停
    if breadth and breadth.get("total", 0) > 0:
        up = breadth["up"]
        down = breadth["down"]
        total = breadth["total"]
        lu = breadth["limit_up"]
        ld = breadth["limit_down"]
        lines.append(f"涨跌比: {up}/{down} (共{total}) | 涨停{lu} 跌停{ld}")
    else:
        lines.append("涨跌比: --")

    lines.append("")

    # 3) 板块 Top5 / Bottom5
    if top_sectors:
        top_str = " ".join([f"{name}({chg:+.2f}%)" for name, chg in top_sectors])
        lines.append(f"板块Top5: {top_str}")
    else:
        lines.append("板块Top5: --")

    if bottom_sectors:
        bottom_str = " ".join([f"{name}({chg:+.2f}%)" for name, chg in bottom_sectors])
        lines.append(f"板块Bottom5: {bottom_str}")

    lines.append("")

    # 4) 北向资金
    if north_flow is not None:
        sign = "+" if north_flow >= 0 else ""
        lines.append(f"北向资金: {sign}{north_flow}亿")
    else:
        lines.append("北向资金: --")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    result = build_snapshot()
    print(result)


if __name__ == "__main__":
    main()
