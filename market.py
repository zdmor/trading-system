"""
大盘分析模块
功能: 指数行情、涨跌统计、涨停质量、量价对比、大盘健康度判断、交易时间检查
数据源: 腾讯财经 HTTP API（免费）
"""
import requests
import numpy as np
from datetime import datetime, time


class MarketAnalyzer:

    INDEX_CODES = {
        "上证指数": "sh000001",
        "深证成指": "sz399001",
        "创业板指": "sz399006",
        "科创50":  "sh000688",
    }

    QT_URL = "http://qt.gtimg.cn/q={codes}"

    # ───────────────────────────── 指数实时行情 ─────────────────────────────

    @classmethod
    def fetch_indices(cls):
        """获取主要指数实时行情"""
        codes = ",".join(cls.INDEX_CODES.values())
        result = {}
        try:
            r = requests.get(cls.QT_URL.format(codes=codes), timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
            for line in r.text.strip().split(";"):
                line = line.strip()
                if "=" not in line:
                    continue
                try:
                    key = line.split("=")[0].replace("v_", "")
                    parts = line.split("=", 1)[1].strip('"').split("~")
                    name = parts[1]
                    price  = float(parts[3])  if len(parts) > 3  and parts[3] else 0
                    pre_c  = float(parts[4])  if len(parts) > 4  and parts[4] else 0
                    open_p = float(parts[5])  if len(parts) > 5  and parts[5] else 0
                    change = float(parts[31]) if len(parts) > 31 and parts[31] else 0
                    chg_pct = float(parts[32]) if len(parts) > 32 and parts[32] else 0
                    high   = float(parts[33]) if len(parts) > 33 and parts[33] else 0
                    low    = float(parts[34]) if len(parts) > 34 and parts[34] else 0
                    volume = float(parts[38]) if len(parts) > 38 and parts[38] else 0
                    amount = float(parts[39]) if len(parts) > 39 and parts[39] else 0
                    if name:
                        result[key] = dict(name=name, price=price, change=change,
                                           change_pct=chg_pct, volume=volume,
                                           high=high, low=low,
                                           open=open_p, pre_close=pre_c)
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass
        return result

    # ───────────────────────────── 指数K线 ─────────────────────────────

    @classmethod
    def fetch_index_kline(cls, code, days=5):
        """获取指数日K线 [date, open, close, high, low, volume, amount]"""
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            if data.get("code") != 0:
                return []
            return (data.get("data", {}).get(code, {}).get("qfqday") or
                    data.get("data", {}).get(code, {}).get("day") or [])
        except Exception:
            return []

    @classmethod
    def get_index_trend(cls, code, name=""):
        """获取指数短期趋势 + 量对比（成交量）"""
        klines = cls.fetch_index_kline(code, days=5)
        if len(klines) < 2:
            return None

        closes  = [float(k[2]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        info = dict(
            name=name or code,
            today_close=closes[-1],
            yesterday_close=closes[-2],
            change_pct=(closes[-1] / closes[-2] - 1) * 100,
            today_vol=volumes[-1],
            yesterday_vol=volumes[-2],
            vol_change_pct=(volumes[-1] / volumes[-2] - 1) * 100,
        )
        if len(closes) >= 3:
            info["high_3"] = max(closes[-3:])
            info["low_3"]  = min(closes[-3:])
            info["range_3_pct"] = (info["high_3"] / info["low_3"] - 1) * 100
        return info

    # ───────────────────────────── 涨跌统计 ─────────────────────────────

    @classmethod
    def analyze_breadth(cls, snapshot):
        """从全市场快照计算涨跌比、涨跌停"""
        total = len(snapshot)
        if total == 0:
            return None

        advance = decline = flat = 0
        limit_up = limit_down = 0
        rise_5 = fall_5 = 0
        total_amount = 0.0

        for s in snapshot.values():
            chg = s.get("change_pct", 0)
            amt = s.get("amount", 0)
            total_amount += amt

            if chg >= 9.8:
                limit_up += 1; advance += 1
            elif chg <= -9.8:
                limit_down += 1; decline += 1
            elif chg > 0:
                advance += 1
                if chg >= 5: rise_5 += 1
            elif chg < 0:
                decline += 1
                if chg <= -5: fall_5 += 1
            else:
                flat += 1

        ad_ratio = round(advance / max(decline, 1), 2)
        ul_ratio = round(limit_up / max(limit_down, 1), 1)
        limit_up_total = limit_up + limit_down

        return dict(
            total=total, advance=advance, decline=decline, flat=flat,
            limit_up=limit_up, limit_down=limit_down,
            rise_5=rise_5, fall_5=fall_5,
            ad_ratio=ad_ratio, ul_ratio=ul_ratio,
            advance_pct=round(advance / total * 100, 1),
            total_amount=round(total_amount / 1e8, 0),
        )

    # ───────────────────────────── 交易所成交额 ─────────────────────────────

    @staticmethod
    def get_exchange_amounts(snapshot):
        """按交易所拆分成交额（sh=沪市, sz=深市）"""
        sh_amt = sz_amt = 0.0
        for code, s in snapshot.items():
            amt = s.get("amount", 0)
            if code.startswith("sh."):
                sh_amt += amt
            else:
                sz_amt += amt
        return dict(
            sh=round(sh_amt / 1e8, 0),
            sz=round(sz_amt / 1e8, 0),
            total=round((sh_amt + sz_amt) / 1e8, 0),
        )

    # ───────────────────────────── 涨停质量分析 ─────────────────────────────

    @classmethod
    def analyze_limitup_quality(cls, snapshot):
        """
        分析今日涨停个股的开盘表现，判断追涨热度。
        分类:
          - 一字板: open >= pre_close * 1.095（开盘即涨停）
          - 高开涨停: open >= pre_close * 1.05（跳空高开后涨停）
          - 低开拉起: open < pre_close * 1.02（平开/低开后拉涨停）
        """
        one_word = 0   # 一字板
        gap_up   = 0   # 高开涨停
        pulled   = 0   # 低开拉起
        limit_pool = []

        for code, s in snapshot.items():
            chg = s.get("change_pct", 0)
            if chg < 9.8:
                continue
            o  = s.get("open", 0)
            pc = s.get("pre_close", 0)
            if pc <= 0:
                continue
            open_pct = (o / pc - 1) * 100

            if open_pct >= 9.5:
                one_word += 1
                limit_pool.append("一字")
            elif open_pct >= 5:
                gap_up += 1
                limit_pool.append("高开")
            else:
                pulled += 1
                limit_pool.append("拉起")

        total = one_word + gap_up + pulled
        if total == 0:
            return None

        return dict(
            total=total,
            one_word=one_word, one_word_pct=round(one_word / total * 100, 1),
            gap_up=gap_up, gap_up_pct=round(gap_up / total * 100, 1),
            pulled=pulled, pulled_pct=round(pulled / total * 100, 1),
        )

    # ───────────────────────────── 大盘健康度 ─────────────────────────────

    # ───────────────────────────── 交易日/交易时间判断 ─────────────────────────────

    @staticmethod
    def is_trading_time(now=None):
        """
        判断当前是否在A股交易时段。
        返回: (can_trade, status_str)
        """
        now = now or datetime.now()
        weekday = now.weekday()
        t = now.time()

        # 周末
        if weekday >= 5:
            return False, "非交易日（周末）"

        # 早盘 09:30-11:30
        if time(9, 30) <= t <= time(11, 30):
            return True, "盘中（早盘）"
        # 午盘 13:00-15:00
        if time(13, 0) <= t <= time(15, 0):
            return True, "盘中（午盘）"
        # 集合竞价 09:15-09:25
        if time(9, 15) <= t < time(9, 30):
            return False, "集合竞价（可报单不可撤单）"
        # 午休 11:30-13:00
        if time(11, 30) < t < time(13, 0):
            return False, "午休"
        # 收盘后
        if t > time(15, 0):
            return False, "已收盘"

        return False, "非交易时段"

    @classmethod
    def judge(cls, indices, breadth, limitup_quality=None):
        """综合判断大盘健康度"""
        if not indices and not breadth:
            return "数据不足", [], []

        positives = []
        warnings_list = []

        # ── 指数 ──
        if indices:
            avg_chg = np.mean([i["change_pct"] for i in indices.values()])
            if avg_chg >= 1:
                positives.append(f"主要指数普涨（均值{avg_chg:+.2f}%）")
            elif avg_chg >= 0.3:
                positives.append(f"指数小幅上涨（均值{avg_chg:+.2f}%）")
            elif avg_chg >= 0:
                positives.append(f"指数平盘（均值{avg_chg:+.2f}%）")
            elif avg_chg >= -0.5:
                warnings_list.append(f"指数小幅回调（均值{avg_chg:+.2f}%）")
            else:
                warnings_list.append(f"指数明显下跌（均值{avg_chg:+.2f}%）")

        # ── 涨跌比 ──
        if breadth:
            if breadth["ad_ratio"] >= 2:
                positives.append(f"涨跌比{breadth['ad_ratio']}（普涨格局）")
            elif breadth["ad_ratio"] >= 1.2:
                positives.append(f"涨跌比{breadth['ad_ratio']}（涨多跌少）")
            elif breadth["ad_ratio"] >= 0.8:
                warnings_list.append(f"涨跌比{breadth['ad_ratio']}（偏弱）")
            else:
                warnings_list.append(f"涨跌比{breadth['ad_ratio']}（极弱）")

        # ── 涨跌停 ──
        if breadth and breadth["limit_up"] + breadth["limit_down"] > 0:
            if breadth["ul_ratio"] >= 5:
                positives.append(f"涨停{breadth['limit_up']}跌停{breadth['limit_down']}（做多情绪强）")
            elif breadth["ul_ratio"] >= 1:
                positives.append(f"涨停{breadth['limit_up']}跌停{breadth['limit_down']}（正常）")
            else:
                warnings_list.append(f"涨停{breadth['limit_up']}跌停{breadth['limit_down']}（做空占优）")
            if breadth["limit_up"] > 50:
                positives.append("涨停数超50只（短线活跃）")

        # ── 涨停质量（情绪延续性） ──
        if limitup_quality:
            if limitup_quality["one_word_pct"] >= 40:
                positives.append(f"一字板占比{limitup_quality['one_word_pct']:.0f}%（追涨意愿强）")
            elif limitup_quality["pulled_pct"] >= 40:
                warnings_list.append(f"低开拉起占比{limitup_quality['pulled_pct']:.0f}%（涨停底气偏弱）")

        # ── 综合评分 ──
        score = len(positives) - len(warnings_list) * 1.5
        if score >= 3:
            health = "健康 — 可积极参与"
        elif score >= 1:
            health = "较好 — 精选个股操作"
        elif score >= -1:
            health = "一般 — 注意仓位管理"
        elif score >= -3:
            health = "偏弱 — 建议降低仓位"
        else:
            health = "较差 — 防守为主，减少交易"

        return health, positives, warnings_list
