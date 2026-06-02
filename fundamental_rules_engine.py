#!/usr/bin/env python3
"""
fundamental_rules_engine.py — 多书融合可编码交易规则引擎

从4本投资经典中提取的60条规则整合为统一评分引擎：
  - Graham 《聪明的投资者》         18条  (价值筛选 / 安全边际)
  - Lynch  《彼得林奇投资经典全集》   15条  (PEG估值 / 6类公司 / 仓位管理)
  - Huang  《财务报表分析》           12条  (财务舞弊识别 / 哈佛框架)
  - Trend  《股市趋势技术分析》       15条  (形态识别 / 突破确认 / 趋势线)

核心接口:
    engine = FundamentalRulesEngine()
    result = engine.analyze_stock("000001.SZ", fundamental_data)

输出:
    {"score": 72, "verdict": "PASS", "flags": [...], "filter_out": false}

独立运行: python fundamental_rules_engine.py <code> <data_json>
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════
# 数据类型定义
# ═══════════════════════════════════════════════════════════════════

class Verdict(Enum):
    PASS = "PASS"        # score >= 60
    WARN = "WARN"        # 40 <= score < 60
    FAIL = "FAIL"        # score < 40
    VETO = "VETO"        # 一票否决


@dataclass
class RuleFlag:
    rule: str                      # 规则ID, e.g. "GRAHAM_001"
    severity: str                  # "pass" | "warn" | "fail" | "veto" | "info"
    detail: str                    # 人类可读详情
    score_change: int = 0          # 本次得分变化


@dataclass
class AnalysisResult:
    score: int = 100
    verdict: str = Verdict.PASS.value
    flags: List[Dict] = field(default_factory=list)
    filter_out: bool = False
    summary: str = ""


# 规则优先级 -> 扣分值映射
PRIORITY_WEIGHTS = {
    "must": 3,
    "should": 2,
    "reference": 1,
}

# 及格线
PASS_THRESHOLD = 60
WARN_THRESHOLD = 40


# ═══════════════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════════════

class FundamentalRulesEngine:
    """多书融合规则引擎"""

    def __init__(self, rules_dir: str = None):
        """
        初始化引擎, 加载4个JSON规则集

        Args:
            rules_dir: JSON规则文件目录, 默认 ../rules/ 相对本脚本
        """
        if rules_dir is None:
            rules_dir = Path(__file__).parent / "rules"
        self.rules_dir = Path(rules_dir)
        self.rules: Dict[str, List[Dict]] = {}  # {"graham": [...], "lynch": [...], ...}
        self._all_rules: List[Dict] = []         # 所有规则平铺
        self._load_rules()

    # ── 加载规则 ────────────────────────────────────────────

    def _load_rules(self):
        """从4个JSON文件加载规则"""
        mapping = {
            "graham": "rules_graham.json",
            "lynch":  "rules_lynch.json",
            "huang":  "rules_huang.json",
            "trend":  "rules_trend.json",
        }
        for key, filename in mapping.items():
            filepath = self.rules_dir / filename
            if not filepath.exists():
                print(f"[WARN] 规则文件不存在: {filepath}", file=sys.stderr)
                self.rules[key] = []
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                rules = json.load(f)
            self.rules[key] = rules
            self._all_rules.extend(rules)
        print(f"[INFO] 加载完成: {len(self._all_rules)}条规则 (Graham:{len(self.rules.get('graham',[]))} "
              f"Lynch:{len(self.rules.get('lynch',[]))} Huang:{len(self.rules.get('huang',[]))} "
              f"Trend:{len(self.rules.get('trend',[]))})")

    # ── 主分析接口 ─────────────────────────────────────────

    def analyze_stock(self, code: str, fundamental_data: Dict) -> Dict:
        """
        综合4书规则对个股进行评分

        Args:
            code:            股票代码, e.g. "000001.SZ"
            fundamental_data: 财务与技术数据字典 (见 DATA_SCHEMA.md)

        Returns:
            {"score": int, "verdict": str, "flags": [...], "filter_out": bool, "summary": str}

        评分模型:
            - 满分 100
            - 每条 must   规则违反 → -3
            - 每条 should 规则违反 → -2
            - 每条 reference 规则违反 → -1
            - HUANG_010 (审计非标) → 直接 score=0, filter_out=true
            - 规则通过 → 不扣分 (部分加分规则可加1分)
        """
        result = AnalysisResult()
        data = fundamental_data

        # ── Graham 规则 (18条) ──
        self._check_graham_pe(result, data)                    # GRAHAM_001
        self._check_pe_x_pb(result, data)                      # GRAHAM_002
        self._check_ncav(result, data)                         # GRAHAM_003
        self._check_dividend_years(result, data)               # GRAHAM_004
        self._check_current_ratio(result, data)                 # GRAHAM_005
        self._check_eps_history(result, data)                  # GRAHAM_006
        self._check_scale(result, data)                        # GRAHAM_007
        self._check_diversification_graham(result, data)       # GRAHAM_008
        self._check_allocation(result, data)                   # GRAHAM_009
        self._check_bond_coverage(result, data)                # GRAHAM_010
        self._check_season_buy(result, data)                   # GRAHAM_011
        self._check_crash_buy(result, data)                    # GRAHAM_012
        self._check_no_stop_loss(result, data)                 # GRAHAM_013
        self._check_sell_at_fair_value(result, data)           # GRAHAM_014
        self._check_netnet_position(result, data)              # GRAHAM_015
        self._check_tangible_book_ratio(result, data)          # GRAHAM_016
        self._check_margin_of_safety(result, data)             # GRAHAM_017
        self._check_low_pe_djia(result, data)                  # GRAHAM_018

        # ── Lynch 规则 (15条) ──
        self._check_peg(result, data)                          # LYNCH_001
        self._check_portfolio_count(result, data)              # LYNCH_002
        self._check_company_type(result, data)                  # LYNCH_003 (info)
        self._check_everyday_observation(result, data)          # LYNCH_004 (info)
        self._check_net_cash(result, data)                     # LYNCH_005
        self._check_institutional_ownership(result, data)       # LYNCH_006
        self._check_insider_buying(result, data)               # LYNCH_007
        self._check_no_panic_sell(result, data)                # LYNCH_008 (info)
        self._check_cyclical_inverse_pe(result, data)          # LYNCH_009
        self._check_hot_industry(result, data)                 # LYNCH_010
        self._check_debt_safety(result, data)                  # LYNCH_011
        self._check_review_schedule(result, data)              # LYNCH_012 (info)
        self._check_fast_grower(result, data)                  # LYNCH_013
        self._check_stalwart(result, data)                     # LYNCH_014
        self._check_asset_play_sell(result, data)              # LYNCH_015 (info)

        # ── Huang 规则 (12条) ──
        # HUANG_010 (审计非标) 必须在其他黄氏规则之前执行: 一票否决
        if self._check_audit_veto(result, data):               # HUANG_010 → VETO
            return result.__dict__

        self._check_receivable_growth(result, data)            # HUANG_001
        self._check_ocf_vs_income(result, data)                # HUANG_002
        self._check_goodwill_ratio(result, data)               # HUANG_003
        self._check_related_party(result, data)                # HUANG_004
        self._check_impairment_bath(result, data)              # HUANG_005
        self._check_inventory_growth(result, data)             # HUANG_006
        self._check_debt_ratio_huang(result, data)             # HUANG_007
        self._check_q4_revenue(result, data)                   # HUANG_008
        self._check_gross_margin_change(result, data)           # HUANG_009
        self._check_prior_period_adjustment(result, data)      # HUANG_011
        self._check_harvard_framework(result, data)            # HUANG_012 (info)

        # ── Trend 规则 (15条) ──
        self._check_dow_theory(result, data)                   # TREND_001 (info)
        self._check_hns_bottom(result, data)                   # TREND_002
        self._check_hns_top(result, data)                      # TREND_003
        self._check_double_bottom(result, data)                # TREND_004
        self._check_double_top(result, data)                   # TREND_005
        self._check_volume_breakout(result, data)              # TREND_006
        self._check_support_resistance_swap(result, data)      # TREND_007 (info)
        self._check_trendline_break(result, data)              # TREND_008
        self._check_consolidation_breakout(result, data)       # TREND_009
        self._check_gap_analysis(result, data)                 # TREND_010 (info)
        self._check_technical_stop_loss(result, data)          # TREND_011
        self._check_ma_cross(result, data)                     # TREND_012
        self._check_trend_diversification(result, data)         # TREND_013
        self._check_pullback_buy(result, data)                 # TREND_014
        self._check_false_breakout(result, data)               # TREND_015

        # ── 生成综合评分 ──
        result.verdict = self._compute_verdict(result.score, result.filter_out)
        result.summary = self._build_summary(result)
        return result.__dict__

    # ── 辅助方法 ───────────────────────────────────────────

    def _deduct(self, result: AnalysisResult, rule_id: str, detail: str,
                 priority: str = "must") -> None:
        """扣分并记录flag"""
        weight = PRIORITY_WEIGHTS.get(priority, 2)
        result.score = max(0, result.score - weight)
        sev = "fail" if priority == "must" else "warn"
        result.flags.append({
            "rule": rule_id,
            "severity": sev,
            "detail": detail,
            "score_change": -weight,
        })

    def _pass(self, result: AnalysisResult, rule_id: str, detail: str,
              bonus: int = 0) -> None:
        """通过检查 (可选加分)"""
        sev = "pass"
        if bonus > 0:
            result.score = min(100, result.score + bonus)
            result.flags.append({
                "rule": rule_id, "severity": "pass",
                "detail": f"{detail} (+{bonus}分)",
                "score_change": bonus,
            })
        else:
            result.flags.append({
                "rule": rule_id, "severity": "pass",
                "detail": f"{detail} [OK]",
                "score_change": 0,
            })

    def _info(self, result: AnalysisResult, rule_id: str, detail: str) -> None:
        """仅记录信息"""
        result.flags.append({
            "rule": rule_id, "severity": "info",
            "detail": detail,
            "score_change": 0,
        })

    def _has_tech(self, data: Dict) -> bool:
        """判断是否提供了技术面数据"""
        return data.get("last_close") is not None and data.get("volume") is not None

    @staticmethod
    def _compute_verdict(score: int, filter_out: bool) -> str:
        if filter_out:
            return Verdict.VETO.value
        if score >= PASS_THRESHOLD:
            return Verdict.PASS.value
        if score >= WARN_THRESHOLD:
            return Verdict.WARN.value
        return Verdict.FAIL.value

    @staticmethod
    def _build_summary(result: AnalysisResult) -> str:
        fail_count = sum(1 for f in result.flags if f["severity"] == "fail")
        warn_count = sum(1 for f in result.flags if f["severity"] == "warn")
        pass_count = sum(1 for f in result.flags if f["severity"] == "pass")
        veto_count = sum(1 for f in result.flags if f["severity"] == "veto")
        parts = [f"得分:{result.score}/100 | 判定:{result.verdict}"]
        if veto_count:
            parts.append(f"一票否决: {veto_count}条")
        parts.append(f"违反(must:{fail_count} should:{warn_count}) 通过:{pass_count}")
        return " | ".join(parts)

    # ═══════════════════════════════════════════════════════════
    # Graham《聪明的投资者》— 18条规则实现
    # ═══════════════════════════════════════════════════════════

    def _check_graham_pe(self, r: AnalysisResult, d: Dict):
        """GRAHAM_001: PE ≤ 25×(7y) ∧ PE ≤ 20×(TTM)"""
        pe_7y = d.get("pe_7yr_avg")
        pe_ttm = d.get("pe_ttm")
        violations = []
        if pe_7y is not None and pe_7y > 25:
            violations.append(f"7年PE={pe_7y:.1f} > 25")
        if pe_ttm is not None and pe_ttm > 20:
            violations.append(f"TTM PE={pe_ttm:.1f} > 20")
        if violations:
            self._deduct(r, "GRAHAM_001", "; ".join(violations), "must")
        elif pe_ttm is not None and pe_7y is not None:
            self._pass(r, "GRAHAM_001", f"7年PE={pe_7y:.1f}, TTM PE={pe_ttm:.1f}")

    def _check_pe_x_pb(self, r: AnalysisResult, d: Dict):
        """GRAHAM_002: PE × PB ≤ 22.5"""
        pe = d.get("pe_ttm") or d.get("pe_7yr_avg")
        pb = d.get("pb")
        if pe is not None and pb is not None:
            pxpb = pe * pb
            if pxpb > 22.5:
                self._deduct(r, "GRAHAM_002", f"PE×PB={pxpb:.1f} > 22.5", "must")
            else:
                self._pass(r, "GRAHAM_002", f"PE×PB={pxpb:.1f} ≤ 22.5", bonus=1)

    def _check_ncav(self, r: AnalysisResult, d: Dict):
        """GRAHAM_003: 股价 ≤ NCAV × 2/3

        NCAV = (流动资产 - 全部负债) / 总股本
        必须与每股股价比较 (不是公司总价值)。
        """
        price = d.get("last_close")
        ncav_ps = d.get("ncav_per_share")       # 优先: 直接给好的每股NCAV
        if ncav_ps is None:
            ca = d.get("current_assets")
            tl = d.get("total_liabilities")
            shares = d.get("total_shares")
            if ca is None or tl is None or shares is None or shares <= 0:
                return
            ncav_ps = (ca - tl) / shares
        if ncav_ps is None or price is None or ncav_ps <= 0:
            return
        if price <= ncav_ps * 2 / 3:
            self._pass(r, "GRAHAM_003",
                       f"股价={price:.2f} ≤ NCAV每股×2/3={ncav_ps*2/3:.2f} (net-net信号!)", bonus=3)

    def _check_dividend_years(self, r: AnalysisResult, d: Dict):
        """GRAHAM_004: 连续 ≥10 年派息"""
        years = d.get("dividend_years")
        if years is not None and years < 10:
            self._deduct(r, "GRAHAM_004", f"连续派息仅{years}年 (<10年)", "must")
        elif years is not None:
            self._pass(r, "GRAHAM_004", f"连续派息{years}年 ≥10")

    def _check_current_ratio(self, r: AnalysisResult, d: Dict):
        """GRAHAM_005: CR ≥ 2.0, 长期债务 ≤ 净资产50%"""
        cr = d.get("current_ratio")
        lt_debt = d.get("long_term_debt") or d.get("non_current_liabilities")
        equity = d.get("total_hldr_eqy")
        violations = []
        if cr is not None and cr < 2.0:
            violations.append(f"流动比率={cr:.2f} < 2.0")
        if lt_debt is not None and equity is not None and equity > 0:
            if lt_debt > equity * 0.5:
                violations.append(f"长期债务/净资产={lt_debt/equity*100:.0f}% > 50%")
        if violations:
            self._deduct(r, "GRAHAM_005", "; ".join(violations), "should")

    def _check_eps_history(self, r: AnalysisResult, d: Dict):
        """GRAHAM_006: 过去10年无亏损, 10年EPS复合增长 ≥ 33%"""
        eps_hist = d.get("eps_history_10yr")
        if eps_hist is None:
            return
        any_neg = any(v < 0 for v in eps_hist)
        if any_neg:
            self._deduct(r, "GRAHAM_006", "过去10年存在亏损年份", "should")
            return
        if len(eps_hist) >= 10:
            growth = (eps_hist[-1] / eps_hist[0]) - 1 if eps_hist[0] > 0 else 0
            if growth < 0.33:
                self._deduct(r, "GRAHAM_006", f"10年EPS增长={growth*100:.1f}% < 33%", "should")
            else:
                self._pass(r, "GRAHAM_006", f"10年EPS增长={growth*100:.1f}% ≥ 33%")

    def _check_scale(self, r: AnalysisResult, d: Dict):
        """GRAHAM_007: 企业规模——总资产 ≥ 50亿 (现代化阈值, 原标准为5000万美元)"""
        ta = d.get("total_assets")
        threshold = 5e9  # 50亿人民币 (原5000万美元现代化调整)
        if ta is not None and ta < threshold:
            self._deduct(r, "GRAHAM_007",
                        f"总资产={ta/1e8:.1f}亿 < {threshold/1e8:.0f}亿", "should")

    def _check_diversification_graham(self, r: AnalysisResult, d: Dict):
        """GRAHAM_008: 多样化——单股 ≤ 4%组合, 同行业 ≤ 20%"""
        cnt = d.get("portfolio_count")
        sector_pct = d.get("sector_concentration")
        if cnt is not None and cnt > 25:
            self._deduct(r, "GRAHAM_008", f"持仓{cnt}只 > 25 (过度集中)", "must")

    def _check_allocation(self, r: AnalysisResult, d: Dict):
        """GRAHAM_009: 债券/股票 25%~75% 动态平衡"""
        stock_pct = d.get("stock_allocation_pct")
        if stock_pct is not None:
            if stock_pct < 25 or stock_pct > 75:
                self._deduct(r, "GRAHAM_009", f"股票配置={stock_pct}% (建议25~75%)", "should")
            else:
                self._pass(r, "GRAHAM_009", f"股票配置={stock_pct}% 在合理区间")

    def _check_bond_coverage(self, r: AnalysisResult, d: Dict):
        """GRAHAM_010: 债券利息保障倍数 ≥7×(工业)/5×(铁路)/4×(公用)"""
        coverage = d.get("interest_coverage")
        sector = d.get("sector_type", "industrial")
        thresholds = {"industrial": 7, "railway": 5, "utility": 4}
        min_threshold = thresholds.get(sector, 7)
        if coverage is not None and coverage < min_threshold:
            self._deduct(r, "GRAHAM_010",
                        f"利息保障={coverage:.1f}× < {min_threshold}× ({sector})", "should")

    def _check_season_buy(self, r: AnalysisResult, d: Dict):
        """GRAHAM_011: 10-12月税损卖出期 → 买入良机"""
        month = d.get("season_month")
        if month in [10, 11, 12]:
            self._pass(r, "GRAHAM_011", f"当前为{month}月 (年度税损卖出窗口, 买入时机)", bonus=1)

    def _check_crash_buy(self, r: AnalysisResult, d: Dict):
        """GRAHAM_012: 市场崩溃 → 买入良机"""
        crash = d.get("market_crash_signal")
        if crash:
            self._pass(r, "GRAHAM_012", "市场恐慌信号! 格雷厄姆式买入良机", bonus=3)

    def _check_no_stop_loss(self, r: AnalysisResult, d: Dict):
        """GRAHAM_013: 禁止自动止损"""
        stop_active = d.get("stop_loss_active")
        if stop_active:
            self._deduct(r, "GRAHAM_013",
                        "已激活自动止损! 格雷厄姆强烈反对", "should")

    def _check_sell_at_fair_value(self, r: AnalysisResult, d: Dict):
        """GRAHAM_014: 仅在超过内在价值时卖出"""
        intrinsic = d.get("estimated_intrinsic_value")
        price = d.get("last_close")
        if intrinsic is not None and price is not None:
            if price > intrinsic:
                self._deduct(r, "GRAHAM_014",
                            f"股价={price:.2f} > 内在价值={intrinsic:.2f} (考虑卖出)", "should")

    def _check_netnet_position(self, r: AnalysisResult, d: Dict):
        """GRAHAM_015: net-net仓位 ≤5%单股"""
        is_netnet = d.get("is_net_net_stock")
        weight = d.get("portfolio_weight_pct")
        if is_netnet and weight is not None and weight > 5:
            self._deduct(r, "GRAHAM_015", f"net-net股持仓{weight}% > 5%上限", "should")

    def _check_tangible_book_ratio(self, r: AnalysisResult, d: Dict):
        """GRAHAM_016: 股价 ≤ 1.5× 有形账面价值"""
        tb = d.get("tangible_book_per_share")
        price = d.get("last_close")
        if tb is not None and price is not None and tb > 0:
            ratio = price / tb
            if ratio > 1.5:
                self._deduct(r, "GRAHAM_016",
                            f"股价/有形账面={ratio:.1f} > 1.5", "should")

    def _check_margin_of_safety(self, r: AnalysisResult, d: Dict):
        """GRAHAM_017: 安全边际 ≥ 33% (估价/市价 ≥ 1.33)"""
        intrinsic = d.get("estimated_intrinsic_value")
        price = d.get("last_close")
        if intrinsic is not None and price is not None and price > 0:
            mos = (intrinsic / price) - 1
            if mos < 0.33:
                self._deduct(r, "GRAHAM_017",
                            f"安全边际={mos*100:.1f}% < 33% (估价/市价={intrinsic/price:.2f})", "must")
            else:
                self._pass(r, "GRAHAM_017", f"安全边际={mos*100:.1f}% ≥ 33%", bonus=2)

    def _check_low_pe_djia(self, r: AnalysisResult, d: Dict):
        """GRAHAM_018: 低PE策略 (道琼斯最低10只)"""
        rank = d.get("pe_rank_in_index")
        if rank is not None and rank <= 10:
            self._pass(r, "GRAHAM_018", f"PE在指数中排名第{rank} (低PE策略候选)", bonus=1)

    # ═══════════════════════════════════════════════════════════
    # Lynch《战胜华尔街》— 15条规则实现
    # ═══════════════════════════════════════════════════════════

    def _check_peg(self, r: AnalysisResult, d: Dict):
        """LYNCH_001: PEG ≤ 1.0"""
        peg = d.get("peg")
        if peg is None:
            pe = d.get("pe_ttm")
            growth = d.get("eps_growth_rate")
            if pe is not None and growth is not None and growth > 0:
                peg = pe / (growth * 100)  # growth 可能是小数形式
        if peg is not None:
            if peg > 1.0:
                self._deduct(r, "LYNCH_001", f"PEG={peg:.2f} > 1.0", "must")
            else:
                self._pass(r, "LYNCH_001", f"PEG={peg:.2f} ≤ 1.0", bonus=2)

    def _check_portfolio_count(self, r: AnalysisResult, d: Dict):
        """LYNCH_002: 持仓 ≤5(业余) ~ ≤12(深度研究)"""
        cnt = d.get("portfolio_count")
        if cnt is not None:
            if cnt > 12:
                self._deduct(r, "LYNCH_002", f"持仓{cnt}只 (林奇建议≤12只)", "must")
            elif cnt > 5:
                self._info(r, "LYNCH_002", f"持仓{cnt}只 (深度研究中)")
            else:
                self._pass(r, "LYNCH_002", f"持仓{cnt}只 (优秀集中度)")

    def _check_company_type(self, r: AnalysisResult, d: Dict):
        """LYNCH_003: 6类公司分类"""
        ctype = d.get("company_type")
        if ctype:
            types_map = {
                "slow_grower": "缓慢增长型",
                "stalwart": "稳健蓝筹型",
                "fast_grower": "快速增长型",
                "cyclical": "周期性",
                "turnaround": "困境反转型",
                "asset_play": "资产富余型",
            }
            label = types_map.get(ctype, ctype)
            self._info(r, "LYNCH_003", f"公司类型: {label}")

    def _check_everyday_observation(self, r: AnalysisResult, d: Dict):
        """LYNCH_004: 日常生活观察选股 (不可编码)"""
        self._info(r, "LYNCH_004", "日常观察选股 (不可量化, 由投资者自行判断)")

    def _check_net_cash(self, r: AnalysisResult, d: Dict):
        """LYNCH_005: 净现金公司 (现金>总债务)"""
        cash = d.get("cash_equivalents")
        debt = d.get("total_debt")
        if cash is not None and debt is not None:
            if cash > debt:
                ncps = cash - debt
                self._pass(r, "LYNCH_005",
                           f"净现金={ncps/1e8:.2f}亿 (林奇Buff: 极度安全!)", bonus=3)
            # 不违反规则, 不是净现金不是错

    def _check_institutional_ownership(self, r: AnalysisResult, d: Dict):
        """LYNCH_006: 机构持股比例低 → 潜在买家多"""
        inst = d.get("inst_ownership_pct")
        size = d.get("total_mv")
        if inst is not None:
            if inst < 5:
                self._pass(r, "LYNCH_006",
                           f"机构持股仅{inst:.1f}% (未来增量买盘潜力)", bonus=1)

    def _check_insider_buying(self, r: AnalysisResult, d: Dict):
        """LYNCH_007: 内部人买入"""
        buying = d.get("insider_buying")
        if buying:
            self._pass(r, "LYNCH_007", "高管正在买入! 强烈看涨信号", bonus=2)

    def _check_no_panic_sell(self, r: AnalysisResult, d: Dict):
        """LYNCH_008: 不因股价下跌卖出"""
        is_panic = d.get("panic_sell_signal")
        if is_panic:
            self._deduct(r, "LYNCH_008",
                        "检测到恐慌性卖出信号! 林奇警告: 好公司跌了更应该买", "must")

    def _check_cyclical_inverse_pe(self, r: AnalysisResult, d: Dict):
        """LYNCH_009: 周期股PE高低反转逻辑"""
        ctype = d.get("company_type")
        pe = d.get("pe_ttm")
        if ctype == "cyclical" and pe is not None:
            if pe < 10:
                self._deduct(r, "LYNCH_009",
                            f"周期股PE={pe:.1f}<10 (盈利高峰中! → 卖出信号, 不是买入)", "should")
            elif pe > 30:
                self._pass(r, "LYNCH_009",
                           f"周期股PE={pe:.1f}>30 (盈利低谷中 → 反向买入信号!)", bonus=2)

    def _check_hot_industry(self, r: AnalysisResult, d: Dict):
        """LYNCH_010: 避开热门行业"""
        hot = d.get("is_hot_industry")
        if hot:
            self._deduct(r, "LYNCH_010", "热门行业! 林奇警告避开", "must")

    def _check_debt_safety(self, r: AnalysisResult, d: Dict):
        """LYNCH_011: 资产负债表稳健"""
        debt_ratio = d.get("debt_ratio")
        if debt_ratio is not None and debt_ratio > 0.7:
            self._deduct(r, "LYNCH_011",
                        f"资产负债率={debt_ratio*100:.1f}% > 70%", "must")

    def _check_review_schedule(self, r: AnalysisResult, d: Dict):
        """LYNCH_012: 6个月定期复盘"""
        days = d.get("days_since_last_review")
        if days is not None and days > 180:
            self._info(r, "LYNCH_012", f"距上次复盘{days}天 (>180天, 建议检查)")

    def _check_fast_grower(self, r: AnalysisResult, d: Dict):
        """LYNCH_013: 小盘快速增长型——市值<100亿, 增长率>20%, PEG≤1.0"""
        mv = d.get("total_mv")
        growth = d.get("eps_growth_rate")
        peg = d.get("peg")
        if mv is not None and growth is not None:
            if mv < 10e8 and growth > 0.20 and (peg is None or peg <= 1.0):
                self._pass(r, "LYNCH_013",
                           f"市值={mv/1e8:.1f}亿, 增长率={growth*100:.0f}% (快速成长候选!)", bonus=2)

    def _check_stalwart(self, r: AnalysisResult, d: Dict):
        """LYNCH_014: 稳健蓝筹——市值大, 增长率10~20%, PEG≤1.5"""
        mv = d.get("total_mv")
        growth = d.get("eps_growth_rate")
        peg = d.get("peg")
        if mv is not None and growth is not None:
            if mv > 50e8 and 0.10 <= growth <= 0.20 and (peg is None or peg <= 1.5):
                self._pass(r, "LYNCH_014",
                           f"蓝筹候选: 增长率={growth*100:.0f}%, PEG={peg:.2f}", bonus=1)

    def _check_asset_play_sell(self, r: AnalysisResult, d: Dict):
        """LYNCH_015: 资产富余型卖出——隐蔽资产已被认识"""
        realized = d.get("hidden_asset_realized")
        if realized:
            self._deduct(r, "LYNCH_015",
                         "隐蔽资产已被市场发现 (林奇卖出信号)", "should")

    # ═══════════════════════════════════════════════════════════
    # Huang《财务报表分析》— 12条规则实现
    # ═══════════════════════════════════════════════════════════

    def _check_receivable_growth(self, r: AnalysisResult, d: Dict):
        """HUANG_001: 应收账款增幅 > 收入增幅 × 1.5"""
        ar_g = d.get("receivables_growth")
        rev_g = d.get("revenue_growth")
        if ar_g is not None and rev_g is not None:
            ratio = ar_g / rev_g if rev_g > 0 else float('inf')
            if ratio > 1.5:
                self._deduct(r, "HUANG_001",
                            f"应收增幅={ar_g*100:.1f}% 为营收增幅={rev_g*100:.1f}%的{ratio:.1f}倍", "must")
            else:
                self._pass(r, "HUANG_001",
                          f"应收/营收增速比={ratio:.1f} ≤ 1.5")

    def _check_ocf_vs_income(self, r: AnalysisResult, d: Dict):
        """HUANG_002: CFO < 0 ∧ NI > 0 (连续)"""
        ocf = d.get("ocf")
        ni = d.get("n_income")
        ocf_neg_periods = d.get("ocf_negative_periods", 0)
        if ocf is not None and ni is not None:
            if ocf < 0 and ni > 0 and ocf_neg_periods >= 2:
                self._deduct(r, "HUANG_002",
                            f"经营CFO={ocf/1e8:.1f}亿<0, 净利润={ni/1e8:.1f}亿>0 (连续{ocf_neg_periods}期)", "must")

    def _check_goodwill_ratio(self, r: AnalysisResult, d: Dict):
        """HUANG_003: 商誉/总资产 > 20%"""
        gw = d.get("goodwill")
        ta = d.get("total_assets")
        if gw is not None and ta is not None and ta > 0:
            ratio = gw / ta
            if ratio > 0.20:
                self._deduct(r, "HUANG_003",
                            f"商誉/总资产={ratio*100:.1f}% > 20% (AT&T NCR式并购溢价风险)", "should")

    def _check_related_party(self, r: AnalysisResult, d: Dict):
        """HUANG_004: 关联交易占比 > 30%"""
        rp = d.get("related_party_revenue_ratio")
        if rp is not None and rp > 0.30:
            self._deduct(r, "HUANG_004",
                        f"关联交易/营收={rp*100:.1f}% > 30% (重庆实业式风险)", "must")

    def _check_impairment_bath(self, r: AnalysisResult, d: Dict):
        """HUANG_005: 减值损失 > 净利润 × 2 (洗大澡)"""
        imp = d.get("impairment_loss")
        ni = d.get("n_income")
        if imp is not None and ni is not None and abs(ni) > 0:
            if imp > abs(ni) * 2:
                self._deduct(r, "HUANG_005",
                            f"减值损失={imp/1e8:.1f}亿 > 2×净利润 (大洗澡信号!)", "should")

    def _check_inventory_growth(self, r: AnalysisResult, d: Dict):
        """HUANG_006: 存货增长 > 销售增长 × 1.5"""
        inv_g = d.get("inventory_growth")
        rev_g = d.get("revenue_growth")
        if inv_g is not None and rev_g is not None:
            ratio = inv_g / rev_g if rev_g > 0 else float('inf')
            if ratio > 1.5:
                self._deduct(r, "HUANG_006",
                            f"存货增幅={inv_g*100:.1f}% 远超收入增幅={rev_g*100:.1f}% ({ratio:.1f}倍)", "should")

    def _check_debt_ratio_huang(self, r: AnalysisResult, d: Dict):
        """HUANG_007: 资产负债率 > 70%, 利息保障 < 2×"""
        dr = d.get("debt_ratio")
        ic = d.get("interest_coverage")
        violations = []
        if dr is not None and dr > 0.70:
            violations.append(f"资产负债率={dr*100:.1f}% > 70%")
        if ic is not None and ic < 2.0:
            violations.append(f"利息保障={ic:.1f}× < 2×")
        if violations:
            self._deduct(r, "HUANG_007", "; ".join(violations), "should")

    def _check_q4_revenue(self, r: AnalysisResult, d: Dict):
        """HUANG_008: Q4收入占全年 > 40%"""
        q4r = d.get("q4_revenue_ratio")
        if q4r is not None and q4r > 0.40:
            self._deduct(r, "HUANG_008",
                        f"Q4收入占比={q4r*100:.1f}% > 40% (年末突击确认收入风险)", "should")

    def _check_gross_margin_change(self, r: AnalysisResult, d: Dict):
        """HUANG_009: 毛利率变动 > 5pp"""
        gm = d.get("gross_margin")
        gm_prev = d.get("gross_margin_prev")
        if gm is not None and gm_prev is not None:
            change = abs(gm - gm_prev)
            if change > 5:
                self._deduct(r, "HUANG_009",
                            f"毛利率变动={change:.1f}pp > 5pp (从{gm_prev:.1f}% → {gm:.1f}%)", "should")

    def _check_audit_veto(self, r: AnalysisResult, d: Dict) -> bool:
        """HUANG_010: 审计非标 → 一票否决!"""
        opinion = d.get("audit_opinion", "").strip()
        # 先排除标准无保留意见 (含"保留"二字但不构成非标)
        if "标准无保留" in opinion or "无保留意见" in opinion:
            self._pass(r, "HUANG_010", f"审计意见: {opinion} [OK]")
            return False
        # 非标准意见关键词
        non_standard = ["保留意见", "否定", "无法表示", "拒绝表示",
                        "qualified", "adverse", "disclaimer"]
        is_non_standard = any(kw in opinion for kw in non_standard)
        if is_non_standard:
            r.score = 0
            r.filter_out = True
            r.flags.append({
                "rule": "HUANG_010",
                "severity": "veto",
                "detail": f"审计意见: '{opinion}' → 一票否决! score=0, filter_out=true",
                "score_change": -100,
            })
            return True
        if opinion:
            self._pass(r, "HUANG_010", f"审计意见: {opinion} [OK]")
        return False

    def _check_prior_period_adjustment(self, r: AnalysisResult, d: Dict):
        """HUANG_011: 前期差错更正 > 净利润50%"""
        ppa = d.get("prior_period_adjustment")
        ni = d.get("n_income")
        if ppa is not None and ni is not None and abs(ni) > 0:
            if abs(ppa) > abs(ni) * 0.5:
                self._deduct(r, "HUANG_011",
                            f"前期差错更正={abs(ppa)/1e8:.1f}亿 > 50%净利润", "should")

    def _check_harvard_framework(self, r: AnalysisResult, d: Dict):
        """HUANG_012: 哈佛四步框架检查"""
        steps_completed = d.get("harvard_steps_completed", 0)
        if steps_completed < 4:
            self._info(r, "HUANG_012",
                      f"哈佛框架完成度 {steps_completed}/4 "
                      "(战略→会计→财务→前景, 建议全部完成后再做财务分析)")

    # ═══════════════════════════════════════════════════════════
    # Trend《股市趋势技术分析》— 15条规则实现
    # ═══════════════════════════════════════════════════════════

    def _check_dow_theory(self, r: AnalysisResult, d: Dict):
        """TREND_001: 道氏理论 — 工业+运输双指数确认"""
        dow_confirm = d.get("dow_theory_confirmed")
        if dow_confirm is not None:
            if dow_confirm:
                self._pass(r, "TREND_001", "道氏理论双指数趋势确认", bonus=1)
            else:
                self._info(r, "TREND_001", "道氏理论未双确认 (谨慎)")

    def _check_hns_bottom(self, r: AnalysisResult, d: Dict):
        """TREND_002: 头肩底形态 — 颈线突破+放量确认"""
        pattern = d.get("pattern", "")
        vol_ratio = d.get("breakout_volume_ratio", 0)
        if "head_shoulders_bottom" in pattern and vol_ratio >= 1.5:
            self._pass(r, "TREND_002",
                      f"头肩底突破! 量比={vol_ratio:.1f}≥1.5 → 底部确认", bonus=3)

    def _check_hns_top(self, r: AnalysisResult, d: Dict):
        """TREND_003: 头肩顶形态 — 跌破颈线+放量确认"""
        pattern = d.get("pattern", "")
        if "head_shoulders_top" in pattern:
            self._deduct(r, "TREND_003", "头肩顶形态! 顶部反转信号", "should")

    def _check_double_bottom(self, r: AnalysisResult, d: Dict):
        """TREND_004: 双底(W底) — 突破中间高点"""
        pattern = d.get("pattern", "")
        if "double_bottom" in pattern:
            self._pass(r, "TREND_004", "双底(W底)形态 → 底部反转确认", bonus=2)

    def _check_double_top(self, r: AnalysisResult, d: Dict):
        """TREND_005: 双顶(M顶) — 跌破中间低点"""
        pattern = d.get("pattern", "")
        if "double_top" in pattern:
            self._deduct(r, "TREND_005", "双顶(M顶)形态! 顶部反转确认", "should")

    def _check_volume_breakout(self, r: AnalysisResult, d: Dict):
        """TREND_006: 突破阻力位 → 成交量 ≥ 150% 均量"""
        vol_ratio = d.get("breakout_volume_ratio")
        price = d.get("last_close")
        resistance = d.get("resistance_level")
        if vol_ratio is not None and price is not None and resistance is not None:
            if price > resistance:
                if vol_ratio >= 1.5:
                    self._pass(r, "TREND_006",
                              f"放量突破! 量比={vol_ratio:.1f}≥1.5 (真实突破确认)", bonus=2)
                else:
                    self._deduct(r, "TREND_006",
                                f"缩量突破 量比={vol_ratio:.1f}<1.5 (可能是假突破!)", "must")

    def _check_support_resistance_swap(self, r: AnalysisResult, d: Dict):
        """TREND_007: 支撑/阻力角色转换"""
        self._info(r, "TREND_007", "支撑/阻力角色互换原则 (参考)")

    def _check_trendline_break(self, r: AnalysisResult, d: Dict):
        """TREND_008: 趋势线跌破"""
        tl_break = d.get("trendline_broken")
        if tl_break:
            self._deduct(r, "TREND_008", "上升趋势线已跌破! 趋势反转预警", "should")

    def _check_consolidation_breakout(self, r: AnalysisResult, d: Dict):
        """TREND_009: 整固形态突破"""
        pattern = d.get("pattern", "")
        if "consolidation_breakout" in pattern:
            self._pass(r, "TREND_009", "整固形态向上突破 → 原有趋势恢复", bonus=1)

    def _check_gap_analysis(self, r: AnalysisResult, d: Dict):
        """TREND_010: 缺口分析"""
        gap_type = d.get("gap_type")
        if gap_type:
            gap_labels = {
                "breakaway": "突破缺口 (强信号)",
                "measuring": "测量缺口 (中继)",
                "exhaustion": "消耗缺口 (趋势末端预警!)",
                "common": "普通缺口 (可忽略)",
            }
            label = gap_labels.get(gap_type, gap_type)
            if gap_type == "exhaustion":
                self._deduct(r, "TREND_010", f"缺口: {label}", "should")
            else:
                self._info(r, "TREND_010", f"缺口: {label}")

    def _check_technical_stop_loss(self, r: AnalysisResult, d: Dict):
        """TREND_011: 技术止损"""
        stop_price = d.get("stop_loss_price")
        last = d.get("last_close")
        if stop_price is not None and last is not None and last <= stop_price:
            self._deduct(r, "TREND_011",
                        f"已触发止损! last={last:.2f} ≤ stop={stop_price:.2f}", "should")

    def _check_ma_cross(self, r: AnalysisResult, d: Dict):
        """TREND_012: MA金叉/死叉"""
        cross = d.get("ma_cross", "")
        if "golden" in cross:
            self._pass(r, "TREND_012", "MA金叉 → 多头信号", bonus=1)
        elif "dead" in cross:
            self._deduct(r, "TREND_012", "MA死叉 → 空头信号", "should")

    def _check_trend_diversification(self, r: AnalysisResult, d: Dict):
        """TREND_013: 分散持仓 5~10只"""
        cnt = d.get("portfolio_count")
        if cnt is not None and cnt > 10:
            self._deduct(r, "TREND_013", f"持仓{cnt}只 > 10 (趋势分散风险)", "must")

    def _check_pullback_buy(self, r: AnalysisResult, d: Dict):
        """TREND_014: 回调买入 — 价格接近趋势线支撑"""
        price = d.get("last_close")
        support = d.get("trendline_support")
        if price is not None and support is not None and support > 0:
            if 0.95 <= price / support <= 1.05:
                self._pass(r, "TREND_014",
                           f"股价={price:.2f} 接近趋势线支撑={support:.2f} (回调买入信号)", bonus=1)

    def _check_false_breakout(self, r: AnalysisResult, d: Dict):
        """TREND_015: 假突破 — 突破后3日内回补"""
        days = d.get("days_above_resistance")
        if days is not None and 0 < days < 3:
            self._deduct(r, "TREND_015",
                        f"突破仅维持{days}天 → 假突破风险! (回补确认卖出)", "must")
        elif days is not None and days >= 3:
            self._pass(r, "TREND_015", f"突破维持{days}天 ≥ 3天 (真实突破确认)")


# ═══════════════════════════════════════════════════════════════════
# Tushare Pro 数据获取 (可选)
# ═══════════════════════════════════════════════════════════════════

class TushareDataFetcher:
    """
    从 Tushare Pro 获取引擎所需 fundamental_data

    需要 config.json:
      {"tushare_token": "YOUR_TOKEN_HERE"}
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        self.config_path = Path(config_path)
        self.token = None
        self._load_config()
        self.api = None

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.token = (cfg.get("_tushare", {}) or {}).get("token", "") or cfg.get("tushare_token", "")
        else:
            print(f"[WARN] 配置文件不存在: {self.config_path}", file=sys.stderr)

    def _init_api(self):
        if self.api is not None:
            return
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("需要安装 tushare: pip install tushare")
        ts.set_token(self.token)
        self.api = ts.pro_api()

    def fetch_stock_basic(self, code: str) -> Dict:
        """获取 stock_basic + daily_basic 估值数据"""
        self._init_api()
        ts_code = code
        # daily_basic: pe_ttm, pb, total_mv
        try:
            df = self.api.daily_basic(ts_code=ts_code,
                                       fields='ts_code,trade_date,pe_ttm,pb,total_mv')
            if df is None or df.empty:
                return {}
            latest = df.iloc[0].to_dict()
            return {
                "pe_ttm": latest.get("pe_ttm"),
                "pb": latest.get("pb"),
                "total_mv": latest.get("total_mv") * 10000 if latest.get("total_mv") else None,
            }
        except Exception as e:
            print(f"[ERR] daily_basic: {e}", file=sys.stderr)
            return {}

    def fetch_balance_sheet(self, code: str) -> Dict:
        """获取资产负债表关键字段"""
        self._init_api()
        try:
            df = self.api.balancesheet(
                ts_code=code,
                fields='ts_code,ann_date,total_assets,current_assets,'
                       'total_liab,total_hldr_eqy,goodwill,inventories,'
                       'accounts_receiv,current_ratio,quick_ratio',
                limit=1
            )
            if df is None or df.empty:
                return {}
            r = df.iloc[0].to_dict()
            return {
                "total_assets": float(r.get("total_assets") or 0),
                "current_assets": float(r.get("current_assets") or 0),
                "total_liabilities": float(r.get("total_liab") or 0),
                "total_hldr_eqy": float(r.get("total_hldr_eqy") or 0),
                "goodwill": float(r.get("goodwill") or 0),
                "inventory": float(r.get("inventories") or 0),
                "accounts_receivable": float(r.get("accounts_receiv") or 0),
                "current_ratio": float(r.get("current_ratio") or 0),
                "quick_ratio": float(r.get("quick_ratio") or 0),
            }
        except Exception as e:
            print(f"[ERR] balancesheet: {e}", file=sys.stderr)
            return {}

    def fetch_income(self, code: str) -> Dict:
        """获取利润表"""
        self._init_api()
        try:
            df = self.api.income(
                ts_code=code,
                fields='ts_code,ann_date,revenue,total_cogs,n_income,'
                       'eps,ebit,fin_exp,',
                limit=1
            )
            if df is None or df.empty:
                return {}
            r = df.iloc[0].to_dict()
            rev = float(r.get("revenue") or 0)
            cogs = float(r.get("total_cogs") or 0)
            gross_margin = (rev - cogs) / rev * 100 if rev > 0 else 0
            return {
                "revenue": rev,
                "n_income": float(r.get("n_income") or 0),
                "eps": float(r.get("eps") or 0),
                "ebit": float(r.get("ebit") or 0),
                "fin_exp": float(r.get("fin_exp") or 0),
                "gross_margin": gross_margin,
                "interest_coverage": (
                    float(r.get("ebit") or 0) / float(r.get("fin_exp") or 1)
                    if float(r.get("fin_exp") or 0) > 0 else None
                ),
            }
        except Exception as e:
            print(f"[ERR] income: {e}", file=sys.stderr)
            return {}

    def fetch_cashflow(self, code: str) -> Dict:
        """获取现金流量表"""
        self._init_api()
        try:
            df = self.api.cashflow(
                ts_code=code,
                fields='ts_code,ann_date,n_cashflow_act',
                limit=1
            )
            if df is None or df.empty:
                return {}
            return {"ocf": float(df.iloc[0].get("n_cashflow_act") or 0)}
        except Exception as e:
            print(f"[ERR] cashflow: {e}", file=sys.stderr)
            return {}

    def fetch_full_data(self, code: str) -> Dict:
        """
        获取完整的 fundamental_data 字典 (不含技术面和portfolio字段)

        返回字典可直接传入 analyze_stock()
        """
        data = {}
        data.update(self.fetch_stock_basic(code))
        data.update(self.fetch_balance_sheet(code))
        data.update(self.fetch_income(code))
        data.update(self.fetch_cashflow(code))

        # 计算衍生字段
        pe = data.get("pe_ttm")
        pb = data.get("pb")
        if pe and pb:
            data["pe_x_pb"] = pe * pb

        ta = data.get("total_assets", 0)
        tl = data.get("total_liabilities", 0)
        if ta > 0:
            data["debt_ratio"] = tl / ta

        # 默认值
        data.setdefault("portfolio_count", 1)
        data.setdefault("audit_opinion", "标准无保留意见")
        data.setdefault("season_month", None)
        data.setdefault("market_crash_signal", False)
        data.setdefault("harvard_steps_completed", 2)

        return data


# ═══════════════════════════════════════════════════════════════════
# CLI & Demo
# ═══════════════════════════════════════════════════════════════════

def demo():
    """演示用法 — 用模拟数据测试所有规则"""
    engine = FundamentalRulesEngine()

    # ── Case 1: 优质蓝筹 (通过) ──
    quality_stock = {
        "pe_ttm": 15.3, "pe_7yr_avg": 18.0, "pb": 2.1,
        "pe_x_pb": 32.1,  # 略超 (PE×PB=32.1 > 22.5)
        "current_assets": 500e8, "total_liabilities": 200e8,
        "total_assets": 800e8, "total_hldr_eqy": 500e8,
        "current_ratio": 2.5, "goodwill": 10e8,
        "dividend_years": 12,
        "eps_history_10yr": [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5],
        "eps_growth_rate": 0.15, "peg": 1.02,
        "revenue": 300e8, "n_income": 50e8,
        "ocf": 60e8, "ocf_negative_periods": 0,
        "debt_ratio": 0.25, "interest_coverage": 12.0,
        "receivables_growth": 0.10, "revenue_growth": 0.12,
        "inventory_growth": 0.08,
        "related_party_revenue_ratio": 0.05,
        "q4_revenue_ratio": 0.28,
        "gross_margin": 45.0, "gross_margin_prev": 43.0,
        "audit_opinion": "标准无保留意见",
        "portfolio_count": 8,
        "season_month": 11,
        "market_crash_signal": False,
        "last_close": 28.5,
        "volume": 50000000, "avg_volume_20d": 40000000,
        "pattern": "",
        "ma_cross": "golden_cross",
        "estimated_intrinsic_value": 42.0,
        "cash_equivalents": 80e8, "total_debt": 120e8,
        "inst_ownership_pct": 12.5,
        "insider_buying": False,
        "is_hot_industry": False,
        "company_type": "stalwart",
        "harvard_steps_completed": 3,
        "days_above_resistance": 5,
        "breakout_volume_ratio": 1.3,
        "sector_concentration": 0.18,
    }

    print("\n" + "=" * 60)
    print("Case 1: 优质蓝筹 — 贵州茅台类型 (PE合理, CFO强劲)")
    print("=" * 60)
    r1 = engine.analyze_stock("600519.SH", quality_stock)
    print(json.dumps(r1, ensure_ascii=False, indent=2))

    # ── Case 2: 财务舞弊怀疑 (HUANG扣分) ──
    fraud_stock = {
        "pe_ttm": 35.0, "pe_7yr_avg": 42.0, "pb": 4.5,
        "pe_x_pb": 157.5,
        "current_assets": 30e8, "total_liabilities": 80e8,
        "total_assets": 120e8, "total_hldr_eqy": 40e8,
        "current_ratio": 0.8, "goodwill": 35e8,
        "dividend_years": 2,
        "eps_history_10yr": [0.1,0.2,0.1,-0.05,0.3,0.4,0.2,0.3,0.4,0.5],
        "eps_growth_rate": 0.05, "peg": 7.0,
        "revenue": 50e8, "n_income": 5e8,
        "ocf": -8e8, "ocf_negative_periods": 3,
        "debt_ratio": 0.67, "interest_coverage": 1.5,
        "receivables_growth": 0.45, "revenue_growth": 0.12,
        "inventory_growth": 0.35,
        "related_party_revenue_ratio": 0.42,
        "impairment_loss": 12e8,
        "q4_revenue_ratio": 0.55,
        "gross_margin": 28.0, "gross_margin_prev": 35.0,
        "audit_opinion": "标准无保留意见",
        "portfolio_count": 15,
        "last_close": 12.0,
        "pattern": "head_shoulders_top",
        "ma_cross": "dead_cross",
        "breakout_volume_ratio": 0.6,
        "estimated_intrinsic_value": 10.0,
        "cash_equivalents": 3e8, "total_debt": 20e8,
        "company_type": "cyclical",
        "harvard_steps_completed": 1,
        "season_month": 6,
        "days_above_resistance": 0,
    }

    print("\n" + "=" * 60)
    print("Case 2: 舞弊风险股 — 康得新类型 (应收暴增, CFO为负, 关联交易高)")
    print("=" * 60)
    r2 = engine.analyze_stock("002450.SZ", fraud_stock)
    print(json.dumps(r2, ensure_ascii=False, indent=2))

    # ── Case 3: 审计非标 → 一票否决 ──
    veto_stock = {
        "pe_ttm": 8.0, "pb": 0.8,
        "current_assets": 100e8, "total_liabilities": 30e8,
        "total_assets": 200e8, "total_hldr_eqy": 150e8,
        "current_ratio": 3.3,
        "dividend_years": 15,
        "eps_history_10yr": [1,1.2,1.3,1.5,1.6,1.8,2.0,2.1,2.3,2.5],
        "revenue": 80e8, "n_income": 20e8,
        "ocf": 18e8, "ocf_negative_periods": 0,
        "debt_ratio": 0.15,
        "receivables_growth": 0.08, "revenue_growth": 0.10,
        "inventory_growth": 0.05,
        "related_party_revenue_ratio": 0.03,
        "q4_revenue_ratio": 0.26,
        "gross_margin": 60.0, "gross_margin_prev": 58.0,
        # ↓ 这就是一票否决!
        "audit_opinion": "无法表示意见",
        "portfolio_count": 5,
        "last_close": 22.0,
        "estimated_intrinsic_value": 35.0,
        "company_type": "fast_grower",
        "eps_growth_rate": 0.25, "peg": 0.32,
    }

    print("\n" + "=" * 60)
    print("Case 3: 审计一票否决 — 财务数据看起来不错, 但审计'无法表示意见'")
    print("=" * 60)
    r3 = engine.analyze_stock("000999.SZ", veto_stock)
    print(json.dumps(r3, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI: python fundamental_rules_engine.py cli <code> <data_json>
        if len(sys.argv) < 4:
            print("用法: python fundamental_rules_engine.py cli <code> <data_json_path>")
            sys.exit(1)
        code = sys.argv[2]
        data_path = sys.argv[3]
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        engine = FundamentalRulesEngine()
        result = engine.analyze_stock(code, data)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        demo()