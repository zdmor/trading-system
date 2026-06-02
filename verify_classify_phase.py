"""
classify_phase() 验证脚本 — 19 个算例
======================================
对每个算例: 拉取数据 → 截至指定日期 → 调用 classify_phase → 对比预期
"""
import sys, os, json, time, pickle
import numpy as np
import urllib.request

SCRIPT_DIR = r"D:\ClaudeWorkspace\trading_system"
sys.path.insert(0, SCRIPT_DIR)

from scanner import WyckoffAnalyzer

# ─── 数据获取 ───

CACHE_DIR = os.path.join(SCRIPT_DIR, "data_cache", "spring_bt")

def fetch_stock_data(code, max_days=350):
    """获取日线数据（优先缓存→腾讯API）"""
    cache_file = os.path.join(CACHE_DIR, f"{code}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    market = "sh" if code.startswith("6") else "sz"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,{max_days},qfq"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            klines = raw["data"][f"{market}{code}"].get("qfqday", [])
            if not klines:
                klines = raw["data"][f"{market}{code}"].get("day", [])
            dates, closes, highs, lows, opens, volumes = [], [], [], [], [], []
            for row in klines:
                if len(row) < 6: continue
                try:
                    dates.append(row[0])
                    opens.append(float(row[1]))
                    closes.append(float(row[2]))
                    highs.append(float(row[3]))
                    lows.append(float(row[4]))
                    volumes.append(float(row[5]) * 100)
                except: continue
            return {"dates": dates, "opens": opens, "closes": closes, "highs": highs, "lows": lows, "volumes": volumes}
        except Exception as e:
            if attempt == 2:
                print(f"  ⚠️ {code} 拉取失败: {e}")
                return None
            time.sleep(1.5)

def slice_to_date(data, end_date):
    """截至 end_date (含) 的数据切片"""
    mask = [d <= end_date for d in data["dates"]]
    n = sum(mask)
    if n == 0:
        return {k: [] for k in data}
    return {k: np.array(v[:n], dtype=(str if k == "dates" else float)) for k, v in data.items()}

# ─── 算例定义 ───

CASES = {
    # Accum_B (5个)
    "1":  {"code": "600352", "name": "浙江医药", "date": "2026-05-26", "expect": "Accum_B"},
    "2":  {"code": "600392", "name": "盛和资源", "date": "2026-05-26", "expect": "Accum_B"},
    "3":  {"code": "600660", "name": "福耀玻璃", "date": "2026-05-26", "expect": "Accum_B"},
    "4":  {"code": "600801", "name": "华新水泥", "date": "2026-05-26", "expect": "Accum_B"},
    "5":  {"code": "600685", "name": "中船防务", "date": "2026-05-26", "expect": "Accum_B"},
    # Markup (4个)
    "6":  {"code": "300308", "name": "中际旭创", "date": "2026-05-26", "expect": "Markup"},
    "7":  {"code": "002050", "name": "三花智控", "date": "2026-05-26", "expect": "Markup"},
    "8":  {"code": "600176", "name": "中国巨石", "date": "2026-05-26", "expect": "Markup"},
    "9":  {"code": "600563", "name": "法拉电子", "date": "2026-05-26", "expect": "Markup"},
    # Distribute_B (4个)
    "10": {"code": "300274", "name": "阳光电源", "date": "2026-05-26", "expect": "Distribute_B"},
    "11": {"code": "600909", "name": "华安证券", "date": "2026-05-26", "expect": "Distribute_B"},
    "12": {"code": "600580", "name": "卧龙电驱", "date": "2026-05-26", "expect": "Distribute_B"},
    "13": {"code": "600699", "name": "均胜电子", "date": "2026-05-26", "expect": "Distribute_B"},
    # Markdown (4个)
    "14": {"code": "600038", "name": "中直股份", "date": "2026-06-01", "expect": "Markdown"},
    "15": {"code": "600588", "name": "用友网络", "date": "2026-05-26", "expect": "Markdown"},
    "16": {"code": "600208", "name": "新湖中宝", "date": "2026-05-26", "expect": "Markdown"},
    "17": {"code": "600346", "name": "恒力石化", "date": "2026-05-26", "expect": "Markdown"},
    # Accum_A (2个)
    "18": {"code": "600316", "name": "洪都航空", "date": "2026-05-26", "expect": "Accum_A_OR_C"},
    "19": {"code": "002074", "name": "国轩高科", "date": "2026-05-26", "expect": "Accum_A"},
}

def run_verification():
    results = []
    passed = 0
    failed = 0

    for case_id, c in CASES.items():
        code, name, date, expect = c["code"], c["name"], c["date"], c["expect"]
        print(f"\n[{case_id}/19] {code} {name} (截至 {date}, 预期 {expect})")

        data = fetch_stock_data(code)
        if data is None:
            print(f"  ❌ 数据拉取失败")
            results.append((case_id, code, name, expect, "数据失败", -1, {}))
            failed += 1
            continue

        sliced = slice_to_date(data, date)
        n = len(sliced["closes"])
        if n < 60:
            print(f"  ❌ 数据不足 ({n}<60)")
            results.append((case_id, code, name, expect, "数据不足", -1, {}))
            failed += 1
            continue

        # 调用 classify_phase
        label, conf, desc = WyckoffAnalyzer.classify_phase(
            sliced["closes"], sliced["highs"], sliced["lows"],
            sliced["opens"], sliced["volumes"]
        )

        # 判定
        if expect == "Accum_A_OR_C":
            ok = label in ("Accum_A", "Accum_C")
        else:
            ok = label == expect

        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  {status} 分类器: {label}(conf={conf})  预期: {expect}")
        print(f"         {desc}")

        results.append((case_id, code, name, expect, label, conf, {"desc": desc, "ok": ok}))

    print(f"\n{'='*60}")
    print(f"  总计: {passed}✅ / {failed}❌ / {len(CASES)}")
    print(f"{'='*60}")

    # 详细表格
    print(f"\n{'#':<3} {'代码':<8} {'名称':<8} {'预期':<16} {'实际':<16} {'置信':<5} {'结果'}")
    for r in results:
        cid, code, name, exp, actual, conf, meta = r
        mark = "✅" if meta.get("ok") else "❌"
        print(f"{cid:<3} {code:<8} {name:<8} {exp:<16} {actual:<16} {conf:<5} {mark}")

    return results

if __name__ == "__main__":
    print("=" * 60)
    print("  classify_phase() 验证 — 19个算例")
    print("=" * 60)
    run_verification()
