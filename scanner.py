"""
A股全市场选股扫描器 v2.1
功能: 板块分类 + 威科夫形态检测（Spring/SOS/LPS/Upthrust）

流程: 新浪行情中心 -> 流动性过滤 -> 行业映射 -> 威科夫+趋势分析 -> 板块分组报告

数据源: 腾讯财经HTTP API（k线）、新浪（实时行情）、baostock（行业分类）
"""

import requests
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 威科夫形态分析器
# ============================================================

class WyckoffAnalyzer:
    """检测多种威科夫交易形态"""

    @staticmethod
    def _avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 1

    @classmethod
    def analyze_all(cls, closes, highs, lows, volumes, trend="空头"):
        """
        返回该股票所有威科夫信号，按得分降序排列。
        return: [(signal_type, score, detail), ...]
        """
        if len(closes) < 30:
            return [("数据不足", 0, "")]

        signals = []

        # ---- 1. Spring 弹簧 ----
        sig = cls.detect_spring(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 2. SOS 强势信号 ----
        sig = cls.detect_sos(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 3. LPS 最后支撑点 ----
        sig = cls.detect_lps(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 4. Upthrust 上冲回落 ----
        sig = cls.detect_upthrust(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # 按得分降序
        signals.sort(key=lambda x: -x[1])
        return signals

    # -------------------------------------------------------
    # Spring 弹簧
    # -------------------------------------------------------
    @classmethod
    def detect_spring(cls, closes, highs, lows, volumes, trend):
        """
        弹簧：价格跌破近期支撑，快速收回，伴随放量
        只在多头/盘整趋势中有效
        """
        if trend == "空头" or trend == "错误":
            return None

        support_data = lows[-12:-3]
        if len(support_data) < 3:
            return None
        support = float(min(support_data))

        cur = float(closes[-1])
        if cur > support * 1.15 or cur < support * 0.95:
            return None

        bg_vol = cls._avg(volumes[-15:-3])

        best = None
        for i in range(len(lows[-3:])):
            low = float(lows[-3:][i])
            close = float(closes[-3:][i])
            vol = float(volumes[-3:][i])

            if low >= support * 0.997 or close <= support:
                continue

            depth = (support - low) / support * 100
            vratio = vol / bg_vol
            bounce = (close - low) / low * 100

            score = 0
            parts = []
            if depth >= 4: score += 25; parts.append(f"深探{depth:.1f}%")
            elif depth >= 2: score += 20; parts.append(f"中探{depth:.1f}%")
            else: score += 12; parts.append(f"浅探{depth:.1f}%")

            if vratio >= 2.5: score += 25; parts.append("巨量")
            elif vratio >= 1.8: score += 20; parts.append("放量")
            elif vratio >= 1.2: score += 12; parts.append("微放量")
            else: score += 5; parts.append("量平")

            if bounce >= 5: score += 25; parts.append(f"强弹{bounce:.1f}%")
            elif bounce >= 3: score += 18; parts.append(f"中弹{bounce:.1f}%")
            elif bounce >= 1.5: score += 10; parts.append(f"弱弹{bounce:.1f}%")

            if i + 1 < len(closes[-3:]):
                nc = float(closes[-3:][i + 1])
                if nc > close: score += 15; parts.append("确认")

            if score > (best[1] if best else 0):
                sig = "Spring" if score >= 55 else "弱Spring"
                best = (sig, score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # SOS Sign of Strength 强势信号
    # -------------------------------------------------------
    @classmethod
    def detect_sos(cls, closes, highs, lows, volumes, trend):
        """
        强势信号：大阳线 + 放量 + 高位收盘
        表明需求强劲，供应被吸收
        """
        if trend != "多头":
            return None
        if len(closes) < 50:
            return None

        # 只用最近一根K线检测
        today_o = float(closes[-2])  # 昨日收盘 = 今日开盘（简化）
        today_c = float(closes[-1])
        today_h = float(highs[-1])
        today_l = float(lows[-1])
        today_v = float(volumes[-1])

        # 必须是阳线
        if today_c <= today_o:
            return None

        bg_vol = cls._avg(volumes[-20:-1])
        bg_range = cls._avg([highs[i] - lows[i] for i in range(-20, -1)])

        # 阳线实体
        body = today_c - today_o
        total_range = today_h - today_l
        if total_range < 0.01:
            return None
        pos_in_range = (today_c - today_l) / total_range  # 收盘在K线中的位置(0~1)

        vratio = today_v / bg_vol
        range_ratio = total_range / bg_range if bg_range > 0 else 0

        score = 0
        parts = []

        # 涨幅
        pct = (today_c - today_o) / today_o * 100
        if pct >= 5: score += 25; parts.append(f"大涨{pct:.1f}%")
        elif pct >= 3: score += 20; parts.append(f"中涨{pct:.1f}%")
        elif pct >= 1.5: score += 12; parts.append(f"小涨{pct:.1f}%")
        else: return None  # 涨幅太小不算SOS

        # 量
        if vratio >= 2.0: score += 25; parts.append("放量")
        elif vratio >= 1.5: score += 18; parts.append("增量")
        elif vratio >= 1.2: score += 10; parts.append("微增")
        else: score += 5; parts.append("量平")

        # 高位收盘
        if pos_in_range >= 0.8: score += 25; parts.append("收高位")
        elif pos_in_range >= 0.6: score += 15; parts.append("收中上")
        else: score += 5; parts.append("收中位")

        # 振幅
        if range_ratio >= 1.5: score += 15; parts.append("宽幅")
        elif range_ratio >= 1.0: score += 8; parts.append("正常幅")

        # 价格在MA50之上
        if len(closes) >= 50 and today_c > np.mean(closes[-50:]):
            score += 10; parts.append("趋势上")

        if score >= 50:
            return ("SOS", score, " | ".join(parts))
        return None

    # -------------------------------------------------------
    # LPS Last Point of Support 最后支撑点
    # -------------------------------------------------------
    @classmethod
    def detect_lps(cls, closes, highs, lows, volumes, trend):
        """
        最后支撑点：放量上攻后，缩量回调至支撑附近
        威科夫最佳入场点
        """
        if trend != "多头":
            return None
        if len(closes) < 30:
            return None

        # 近期支撑（类似Spring的支撑计算）
        support = float(min(lows[-12:-3]))

        cur = float(closes[-1])
        # 价格必须在支撑附近（支撑上方0~5%）
        if not (support <= cur <= support * 1.05):
            return None

        # 最近3天是否有过放量上涨（LPS的前提是之前有SOS或上涨）
        recent_volumes = volumes[-10:-3]
        bg_vol = cls._avg(volumes[-25:-10])
        had_upsurge = any(
            float(volumes[-10:-3][i]) > bg_vol * 1.3 and
            float(closes[-10:-3][i]) > float(closes[-11:-4][i])
            for i in range(min(5, len(volumes[-10:-3])))
        )

        if not had_upsurge:
            return None

        # 当前成交量应该萎缩
        today_v = float(volumes[-1])
        vratio = today_v / bg_vol if bg_vol > 0 else 1

        score = 30  # 基础分
        parts = []

        # 缩量
        if vratio <= 0.5: score += 30; parts.append("地量")
        elif vratio <= 0.7: score += 22; parts.append("缩量")
        elif vratio <= 0.9: score += 12; parts.append("微缩")
        else: parts.append("量平")

        # K线形态：小实体更好
        body = abs(float(closes[-1]) - float(closes[-2]))
        avg_body = cls._avg([abs(closes[i] - closes[i-1]) for i in range(-10, -1)])
        if body < avg_body * 0.6: score += 15; parts.append("小实体")
        elif body < avg_body * 0.9: score += 8; parts.append("中实体")

        # 支撑效力：之前该支撑是否被Spring确认过
        support_tested = any(float(lows[i]) < support and float(closes[i]) > support
                             for i in range(-15, 0))
        if support_tested: score += 15; parts.append("支撑验证")

        # 价格波动收窄
        recent_range = float(max(highs[-5:])) - float(min(lows[-5:]))
        older_range = float(max(highs[-10:-5])) - float(min(lows[-10:-5]))
        if older_range > 0 and recent_range < older_range * 0.7:
            score += 10; parts.append("波幅收窄")

        detail = " | ".join(parts) if parts else ""
        sig = "LPS" if score >= 60 else "弱LPS"
        return (sig, score, detail) if score >= 40 else None

    # -------------------------------------------------------
    # Upthrust 上冲回落（UT/UTAD）
    # -------------------------------------------------------
    @classmethod
    def detect_upthrust(cls, closes, highs, lows, volumes, trend):
        """
        上冲回落：价格突破阻力后迅速收回，放量
        威科夫做空/派发信号
        """
        # Upthrust 在多头趋势末期也有效，不限制趋势
        if len(closes) < 20:
            return None

        # 阻力：过去12天的最高点（排除最近3天）
        resistance = float(max(highs[-12:-3]))
        if resistance <= 0:
            return None

        cur = float(closes[-1])
        # 价格必须在阻力附近（阻力下方3%到上方3%）
        if not (resistance * 0.97 <= cur <= resistance * 1.03):
            return None

        bg_vol = cls._avg(volumes[-15:-3])

        best = None
        for i in range(len(highs[-3:])):
            high = float(highs[-3:][i])
            close = float(closes[-3:][i])
            low = float(lows[-3:][i])
            vol = float(volumes[-3:][i])

            # 必须突破过阻力
            if high < resistance:
                continue

            # 收盘必须回到阻力之下或附近（上冲失败）
            if close > resistance * 1.01:
                continue

            vratio = vol / bg_vol
            total_range = high - low
            pos_in_range = (close - low) / total_range if total_range > 0 else 0.5

            score = 30  # 基础分
            parts = []

            # 突破幅度
            thrust = (high - resistance) / resistance * 100
            if thrust >= 3: score += 20; parts.append(f"深探{thrust:.1f}%")
            elif thrust >= 1.5: score += 15; parts.append(f"中探{thrust:.1f}%")
            else: score += 8; parts.append(f"浅探{thrust:.1f}%")

            # 量
            if vratio >= 2.0: score += 25; parts.append("巨量")
            elif vratio >= 1.5: score += 18; parts.append("放量")
            elif vratio >= 1.2: score += 10; parts.append("微放量")
            else: score += 5; parts.append("量平")

            # 低位收盘（上冲回落的特征）
            if pos_in_range <= 0.3: score += 20; parts.append("收低位")
            elif pos_in_range <= 0.5: score += 10; parts.append("收中低")

            # 上影线
            upper_wick = high - max(close, float(closes[-3:][i-1]) if i > 0 else close)
            if upper_wick > total_range * 0.4: score += 15; parts.append("长上影")

            # 次日确认
            if i + 1 < len(closes[-3:]):
                nc = float(closes[-3:][i + 1])
                if nc < close: score += 10; parts.append("次日跌")

            if score > (best[1] if best else 0):
                sig = "Upthrust" if score >= 55 else "弱Upthrust"
                best = (sig, score, " | ".join(parts))

        return best


# ============================================================
# Scanner
# ============================================================

class Scanner:

    SINA_HQ_URL = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"

    INDUSTRY_CACHE = os.path.join(os.path.dirname(__file__), "industry_cache.json")

    def __init__(self):
        self.all_stocks = []
        self.snapshot = {}
        self.industry_map = {}
        self.candidates = []
        self.results = []

    # ==================== 行业分类 ====================

    def build_industry_map(self, force=False):
        if not force and os.path.exists(self.INDUSTRY_CACHE):
            try:
                with open(self.INDUSTRY_CACHE, encoding="utf-8") as f:
                    self.industry_map = json.load(f)
                return
            except Exception: pass

        import baostock as bs
        bs.login()
        try:
            rs = bs.query_stock_industry()
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 4 and row[3]:
                    self.industry_map[row[1]] = row[3]
        finally:
            bs.logout()

        try:
            with open(self.INDUSTRY_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.industry_map, f, ensure_ascii=False)
        except Exception: pass

    def get_industry(self, code):
        return self.industry_map.get(code, "其他")

    # ==================== 获取全市场数据 ====================

    @staticmethod
    def _fetch_page(page, num=100):
        params = {"page": page, "num": num, "sort": "symbol",
                  "asc": "1", "node": "hs_a", "_s_r_a": "init"}
        r = requests.get(Scanner.SINA_HQ_URL, params=params, timeout=20)
        return r.json()

    def fetch_all_stocks(self, max_pages=80):
        all_data = []
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {pool.submit(self._fetch_page, p): p for p in range(1, max_pages + 1)}
            for f in as_completed(futures):
                try:
                    data = f.result()
                    if data:
                        all_data.extend(data)
                except Exception: pass

        stocks, snapshot = [], {}
        for item in all_data:
            code = item.get("code", "")
            name = item.get("name", "")
            symbol = item.get("symbol", "")

            if symbol.startswith("bj") or code.startswith("9"):
                continue

            bs_code = f"{symbol[:2]}.{symbol[2:]}" if len(symbol) >= 4 else f"sz.{code}"
            try:
                price = float(item.get("trade", 0))
                amount = float(item.get("amount", 0))
                volume = float(item.get("volume", 0))
                turnover = float(item.get("turnoverratio", 0))
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                change_pct = float(item.get("changepercent", 0))
            except (ValueError, TypeError):
                continue

            if price <= 0 or amount <= 0:
                continue

            stocks.append({"code": bs_code, "name": name})
            snapshot[bs_code] = {
                "code": bs_code, "name": name, "price": price,
                "amount": amount, "volume": volume, "turnover": turnover,
                "high": high, "low": low, "change_pct": change_pct,
            }

        self.all_stocks = stocks
        self.snapshot = snapshot
        return stocks, snapshot

    # ==================== 过滤 ====================

    def filter_candidates(self, min_amount=5e8):
        candidates = [d for d in self.snapshot.values() if d["amount"] >= min_amount]
        candidates.sort(key=lambda x: x["amount"], reverse=True)
        self.candidates = candidates
        return candidates

    # ==================== 个股分析 ====================

    def _analyze_stock(self, code):
        """完整分析：趋势 + 威科夫形态"""
        result = {"trend": "错误", "strength": 0,
                  "wyckoff_sig": "-", "wyckoff_score": 0, "wyckoff_detail": "",
                  "industry": self.get_industry(code)}

        try:
            key = code.replace(".", "")
            url = self.TENCENT_KLINE_URL.format(code=key)
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("code") != 0:
                return result

            klines = data.get("data", {}).get(key, {}).get("qfqday") or \
                     data.get("data", {}).get(key, {}).get("day") or []

            closes, highs, lows, volumes = [], [], [], []
            for k in klines:
                try:
                    closes.append(float(k[2]))
                    highs.append(float(k[3]))
                    lows.append(float(k[4]))
                    volumes.append(float(k[5]) * 100)
                except (ValueError, IndexError):
                    pass

            # 趋势判断
            if len(closes) >= 200:
                ma50 = np.mean(closes[-50:])
                ma200 = np.mean(closes)
                s = (ma50 / ma200 - 1) * 100
                result["trend"] = "多头" if ma50 > ma200 else "空头"
                result["strength"] = round(s, 1)
            elif len(closes) >= 50:
                result["trend"] = "数据不足"
            else:
                result["trend"] = "新股"

            # 威科夫分析
            if len(closes) >= 30:
                signals = WyckoffAnalyzer.analyze_all(
                    closes, highs, lows, volumes, trend=result["trend"]
                )
                if signals and signals[0][1] > 0:
                    result["wyckoff_sig"] = signals[0][0]
                    result["wyckoff_score"] = signals[0][1]
                    result["wyckoff_detail"] = signals[0][2]

        except Exception:
            pass

        return result

    def run_analysis(self, max_stocks=80):
        results = []
        total = min(len(self.candidates), max_stocks)

        for i, stock in enumerate(self.candidates[:max_stocks]):
            analysis = self._analyze_stock(stock["code"])
            results.append({**stock, **analysis})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{total}")
            time.sleep(0.02)

        # 排序：威科夫信号得分 > 多头 > 成交额
        def sort_key(x):
            ws = -x["wyckoff_score"]  # 高分在前
            tr = 0 if x["trend"] == "多头" else 1
            return (ws, tr, -x["amount"])

        results.sort(key=sort_key)
        self.results = results
        return results

    # ==================== 报告 ====================

    def print_report(self, top_n=15):
        bullish = sum(1 for r in self.results if r["trend"] == "多头")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 信号统计
        sig_counts = {}
        for r in self.results:
            t = r["wyckoff_sig"]
            if t not in ("-", "无信号", "数据不足"):
                sig_counts[t] = sig_counts.get(t, 0) + 1

        print("\n" + "=" * 80)
        print(f"  A股扫描报告  {now}")
        print("=" * 80)
        print(f"  覆盖: {len(self.all_stocks)} 只  流动性通过: {len(self.candidates)} 只")
        print(f"  多头: {bullish}只  ", end="")
        for k, v in sig_counts.items():
            print(f" {k}:{v}只 ", end="")
        print("\n")

        if not self.results:
            print("  无符合条件标的")
            return

        # === 板块分布 ===
        sectors = {}
        for r in self.results:
            ind = r.get("industry", "其他")
            if ind not in sectors:
                sectors[ind] = {"count": 0, "sig_count": 0, "names": []}
            sectors[ind]["count"] += 1
            if r["wyckoff_sig"] not in ("-", "无信号", "数据不足"):
                sectors[ind]["sig_count"] += 1
            if len(sectors[ind]["names"]) < 3:
                sectors[ind]["names"].append(r["name"])

        sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["count"])
        print(f"  【板块分布】")
        for ind, info in sorted_sec[:12]:
            tag = f" 🌱{info['sig_count']}信号" if info['sig_count'] > 0 else ""
            top = ", ".join(info["names"])
            print(f"  {ind:<18} {info['count']:>2}只{tag:<12} | {top}")
        print()

        # === Top N ===
        print(f"  [Top {top_n}]")
        print(f"  {'#':<3} {'代码':<8} {'名称':<7} {'板块':<12} {'趋势':<10} {'威科夫':<18} {'价格':<8} {'成交额':<8}")
        print(f"  " + "-" * 96)

        for i, r in enumerate(self.results[:top_n]):
            sym = r["code"].split(".")[1]
            name = r["name"][:6]
            ind = r.get("industry", "其他")[:10]
            trend = f"{r['trend']}({r['strength']:+.1f}%)" if r["trend"] in ("多头", "空头") else r["trend"]

            sig = r["wyckoff_sig"]
            sc = r["wyckoff_score"]
            sig_str = f"{sig}({sc})" if sig not in ("-", "无信号") else sig

            price = f"{r['price']:.2f}"
            amt = f"{r['amount']/1e8:.1f}亿"
            print(f"  {i+1:<3} {sym:<8} {name:<7} {ind:<12} {trend:<10} {sig_str:<18} {price:<8} {amt:<8}")

        print()

        # === 信号精选 ===
        valid = [r for r in self.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
        if valid:
            print(f"  【威科夫信号精选 Top {min(10, len(valid))}】")
            print(f"  {'信号':<14} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'细节':<32}")
            print(f"  " + "-" * 88)
            for r in valid[:10]:
                sym = r["code"].split(".")[1]
                name = r["name"][:6]
                ind = r.get("industry", "其他")[:10]
                sig = r["wyckoff_sig"]
                sc = r["wyckoff_score"]
                detail = r.get("wyckoff_detail", "")[:30]
                print(f"  {sig:<14} {sym:<8} {name:<7} {ind:<12} {sc:<5} {detail:<32}")
            print()

    # ==================== 主流程 ====================

    def run(self, min_amount=5e8, max_analysis=80, top_n=15, quick=False):
        print("A股扫描器 v2.1（威科夫形态检测）")
        print("=" * 62)
        print(f"成交额门槛: {min_amount/1e8:.0f}亿\n")

        print("[1/3] 获取全市场数据 (新浪)...", end=" ")
        t0 = time.time()
        self.fetch_all_stocks()
        print(f"{len(self.snapshot)} 只有效 ({time.time()-t0:.1f}s)")

        print("[2/3] 流动性过滤...", end=" ")
        t0 = time.time()
        self.filter_candidates(min_amount)
        print(f"{len(self.candidates)} 只 ({time.time()-t0:.1f}s)")

        print("[2.5/3] 行业分类...", end=" ")
        t0 = time.time()
        self.build_industry_map()
        print(f"{len(self.industry_map)} 只映射 ({time.time()-t0:.1f}s)")

        if quick:
            self.results = []
            for s in self.candidates[:max_analysis]:
                self.results.append({
                    **s, "trend": "-", "strength": 0,
                    "wyckoff_sig": "-", "wyckoff_score": 0, "wyckoff_detail": "",
                    "industry": self.get_industry(s["code"]),
                })
        else:
            print("[3/3] 威科夫+趋势分析...")
            t0 = time.time()
            self.run_analysis(max_analysis)
            print(f"  完成 ({time.time()-t0:.1f}s)")

        self.print_report(top_n)


def run_scanner(min_amount=5e8, quick=False):
    Scanner().run(min_amount=min_amount, quick=quick)


if __name__ == "__main__":
    min_amt = 5e8
    quick = False
    for arg in sys.argv[1:]:
        if arg.replace(".", "").isdigit():
            min_amt = float(arg) * 1e8
        elif arg == "--quick":
            quick = True
    run_scanner(min_amt, quick)
