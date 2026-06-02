"""
Vault knowledge integration — 将 INDEX.md 的 #tag 路由表变成可调用的代码。

功能:
  1. 解析 INDEX.md, 建立 tag → file 映射
  2. 读取知识文件, 提取结构化规则 (行为阈值、评分权重、信号质量)
  3. 暴露 get(tag) API, 供 run.py/scoring.py 等模块使用

用法:
    from vault_knowledge import VaultKnowledge
    vk = VaultKnowledge()
    rules = vk.get("behavior")   # 行为约束阈值
    weights = vk.get("scoring")  # 评分配置
"""
import os, re
from pathlib import Path

VAULT_DIR = Path(r"D:\DiskMigration\MySecondBrain")
INDEX_FILE = VAULT_DIR / "INDEX.md"


class VaultKnowledge:
    """知识库加载器 — 将 INDEX.md 路由表变成结构化数据"""

    def __init__(self, vault_dir=None):
        self.vault_dir = Path(vault_dir) if vault_dir else VAULT_DIR
        self._tag_map = {}   # tag → filepath
        self._cache = {}     # tag → structured dict
        self._loaded = False

    # ── 公共 API ──

    def get(self, tag: str) -> dict:
        """获取某个 tag 的结构化知识, 如 get("behavior") → {...}"""
        if not self._loaded:
            self._load_index()
        tag = tag.lstrip("#")
        if tag in self._cache:
            return self._cache[tag]
        if tag not in self._tag_map:
            return {}
        filepath = self._tag_map[tag]
        content = self._read_file(filepath)
        parsed = self._parse(tag, content)
        self._cache[tag] = parsed
        return parsed

    def get_behavior(self) -> dict:
        return self.get("behavior")

    def get_scoring(self) -> dict:
        return self.get("scoring")

    def get_signal(self) -> dict:
        return self.get("signal")

    def reload(self):
        self._tag_map.clear()
        self._cache.clear()
        self._loaded = False

    # ── INDEX.md 解析 ──

    def _load_index(self):
        if not INDEX_FILE.exists():
            return
        text = INDEX_FILE.read_text(encoding="utf-8")
        for m in re.finditer(
            r"\|\s*(#\w+)\s*\|\s*\[\[([^\]]+)\]\]", text):
            tag = m.group(1).lstrip("#")
            rel_path = m.group(2)
            if not rel_path.endswith(".md"):
                rel_path += ".md"
            full = self.vault_dir / rel_path
            if full.exists():
                self._tag_map[tag] = full
        self._loaded = True

    def _read_file(self, filepath: Path) -> str:
        try:
            return filepath.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _parse(self, tag: str, content: str) -> dict:
        parser = getattr(self, f"_parse_{tag}", None)
        if parser:
            return parser(content)
        return {"raw": content[:500]}

    # ── 行为约束解析 ──

    def _parse_behavior(self, text: str) -> dict:
        rules = {
            "entry_forbidden": {"bearish": False, "low_score": False},
            "score_thresholds": {},
            "stop_distance_pct": None,
            "trailing_stop": {},
            "exit_rules": {},
        }

        m = re.search(r"低分禁买.*?评分<(\d+)", text)
        if m:
            rules["entry_forbidden"]["low_score"] = int(m.group(1))
            rules["score_thresholds"]["min_entry"] = int(m.group(1))

        # 空头禁买
        if "空头禁买" in text and "趋势=空头" in text:
            rules["entry_forbidden"]["bearish"] = True

        # 评分 <45 减仓预警
        m = re.search(r"评分\s*<(\d+).*?建议减仓", text)
        if m:
            rules["score_thresholds"]["reduce_alert"] = int(m.group(1))

        # 距止损 <5%
        m = re.search(r"距止损.*?<(\d+)%", text)
        if m:
            rules["stop_distance_pct"] = int(m.group(1))

        # 移动止盈: 浮盈>10% + 回撤>5%
        m = re.search(r"浮盈.*?>(\d+)%.*?回撤.*?>(\d+)%", text)
        if m:
            rules["trailing_stop"] = {
                "min_profit_pct": int(m.group(1)),
                "pullback_pct": int(m.group(2)),
            }

        # 退出规则
        has_logic = "逻辑破" in text
        has_exhaust = "高点不抬高" in text
        if has_logic or has_exhaust:
            lookback = 3
            m = re.search(r"(\d+)根K线", text)
            if m:
                lookback = int(m.group(1))
            rules["exit_rules"] = {
                "logic_broken": has_logic,
                "trend_exhaustion": has_exhaust,
                "lookback": lookback,
            }

        return rules

    # ── 评分系统解析 ──

    def _parse_scoring(self, text: str) -> dict:
        """提取最新的因子权重表和评分阈值"""
        result = {
            "factors": [],
            "thresholds": {},
            "n_factors": 0,
        }

        lines = text.split("\n")
        # 找最近的权重表：找 "| 威科夫信号" 所在表格（第一张）
        in_table = False
        table_done = False
        for line in lines:
            if table_done:
                break
            parts = [p.strip() for p in line.split("|")]

            # 检测表格开始: 表头含 "因子" 和 "权重"
            if (len(parts) >= 4 and "因子" in parts[1] and "权重" in parts[2]
                    and "说明" not in parts[2]):
                in_table = True
                continue

            if not in_table:
                continue

            # 空行或非表格行 → 结束表格
            if not line.strip() or line.strip().startswith("---"):
                continue
            if not line.startswith("|"):
                break

            if len(parts) >= 4 and "%" in parts[2]:
                name = parts[1].strip()
                try:
                    weight = int(parts[2].strip().rstrip("%"))
                    result["factors"].append({"name": name, "weight": weight})
                except ValueError:
                    pass

        if result["factors"]:
            result["n_factors"] = len(result["factors"])

        # 综合分阈值
        for m in re.finditer(r">=(\d+)\s*(强加仓|加仓|持有|减仓|离场)", text):
            result["thresholds"][m.group(2)] = int(m.group(1))

        return result

    # ── 信号质量解析 ──

    def _parse_signal(self, text: str) -> dict:
        result = {"signals": {}}
        in_table = False
        for line in text.split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5 and "信号" in parts[1] and "次数" in parts[2]:
                in_table = True
                continue
            if not in_table:
                continue
            if "--" in line:
                continue
            if not line.startswith("|"):
                break
            if len(parts) >= 5 and parts[1].strip():
                name = parts[1].strip()
                try:
                    result["signals"][name] = {
                        "count": int(parts[2].strip()),
                        "avg_return": float(parts[3].strip().rstrip("%")),
                        "win_rate": float(parts[4].strip().rstrip("%")),
                    }
                except (ValueError, IndexError):
                    continue
        return result


# ── 全局单例 ──
_default = None


def _vk():
    global _default
    if _default is None:
        _default = VaultKnowledge()
        _default.get("behavior")
    return _default


def get_knowledge(tag: str) -> dict:
    return _vk().get(tag)


def get_behavior_rules() -> dict:
    return _vk().get_behavior()


def get_scoring_config() -> dict:
    return _vk().get_scoring()


def get_signal_stats() -> dict:
    return _vk().get_signal()


# ── 知识摘要 (启动时加载) ──
def print_knowledge_summary():
    vk = _vk()
    lines = ["[vault] 知识库加载:"]
    lines.append(f"        tag映射: {len(vk._tag_map)}个")

    behavior = vk.get_behavior()
    st = behavior.get("score_thresholds", {})
    if st:
        parts = []
        if "min_entry" in st:
            parts.append(f"入场分>={st['min_entry']}")
        if "reduce_alert" in st:
            parts.append(f"减仓<{st['reduce_alert']}")
        lines.append(f"        行为约束: {' | '.join(parts)}")
    if behavior.get("stop_distance_pct"):
        lines.append(f"        止损距离: >{behavior['stop_distance_pct']}%")
    er = behavior.get("exit_rules", {})
    if er:
        lines.append(f"        退出规则: 逻辑破={'on' if er.get('logic_broken') else 'off'} | "
                      f"K线衰竭={'on' if er.get('trend_exhaustion') else 'off'}")

    scoring = vk.get_scoring()
    if scoring.get("factors"):
        lines.append(f"        评分: {scoring['n_factors']}因子 | "
                      f"阈值: {scoring.get('thresholds', {})}")

    signal = vk.get_signal()
    if signal.get("signals"):
        lines.append(f"        信号质量: {len(signal['signals'])}种信号有历史记录")

    return "\n".join(lines)


if __name__ == "__main__":
    print(print_knowledge_summary())
