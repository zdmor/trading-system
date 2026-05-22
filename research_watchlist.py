"""
深度研究个股追踪器
记录经过深度研究(市值风云等平台)确认的个股，
在扫描和评分中给予差异化处理

数据存储在 JSON 文件中，轻量无数据库依赖
"""

import json
import os
from datetime import datetime

RESEARCH_FILE = os.path.join(os.path.dirname(__file__), "research_watchlist.json")


class ResearchTracker:
    """深度研究个股追踪"""

    _cache = None

    @classmethod
    def _load(cls) -> dict:
        if cls._cache is None:
            if os.path.exists(RESEARCH_FILE):
                try:
                    with open(RESEARCH_FILE, encoding="utf-8") as f:
                        cls._cache = json.load(f)
                except Exception:
                    cls._cache = {"stocks": []}
            else:
                cls._cache = {"stocks": []}
        return cls._cache

    @classmethod
    def _save(cls):
        if cls._cache is not None:
            with open(RESEARCH_FILE, "w", encoding="utf-8") as f:
                json.dump(cls._cache, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_stock(cls, code: str) -> dict | None:
        """获取个股的研究记录"""
        data = cls._load()
        for s in data.get("stocks", []):
            if s["code"] == code and s.get("active", True):
                return s
        return None

    @classmethod
    def is_confirmed(cls, code: str) -> bool:
        """是否经过深度研究确认"""
        return cls.get_stock(code) is not None

    @classmethod
    def get_quality_bypass(cls, code: str) -> dict:
        """
        返回可跳过哪些质量检查
        Returns: {'roe': bool, 'revenue': bool}
        """
        stock = cls.get_stock(code)
        if stock:
            return {
                "roe": stock.get("bypass_roe", False),
                "revenue": stock.get("bypass_revenue", False),
            }
        return {"roe": False, "revenue": False}

    @classmethod
    def get_score_bonus(cls, code: str) -> int:
        """深度研究确认股票的评分加分"""
        stock = cls.get_stock(code)
        if not stock:
            return 0
        conviction_bonus = {"high": 15, "medium": 8, "low": 3}
        return conviction_bonus.get(stock.get("conviction", "medium"), 5)

    @classmethod
    def add(cls, code: str, name: str, reason: str, conviction: str = "high",
            bypass_roe: bool = False, bypass_revenue: bool = False):
        """添加深度研究个股"""
        data = cls._load()
        # 去重
        data["stocks"] = [s for s in data["stocks"] if s["code"] != code]
        data["stocks"].append({
            "code": code,
            "name": name,
            "date_added": datetime.now().strftime("%Y-%m-%d"),
            "reason": reason,
            "conviction": conviction,
            "bypass_roe": bypass_roe,
            "bypass_revenue": bypass_revenue,
            "active": True,
        })
        cls._save()

    @classmethod
    def remove(cls, code: str):
        """标记为不活跃（软删除）"""
        data = cls._load()
        for s in data.get("stocks", []):
            if s["code"] == code:
                s["active"] = False
        cls._save()

    @classmethod
    def list_active(cls) -> list:
        """列出所有活跃的深度研究个股"""
        data = cls._load()
        return [s for s in data.get("stocks", []) if s.get("active", True)]
