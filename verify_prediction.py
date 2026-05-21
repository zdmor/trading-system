"""
预测验证工具
到期检查 predictions.json 中的预测是否准确，写入结果并汇报
验证通过的判断会自动写入 vault 知识库，形成累积经验
"""
import json, os, sys
from datetime import datetime, timedelta
import pandas as pd
from main import DataFetcher

PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), "predictions.json")
VAULT_SIGNAL_FILE = r"D:\DiskMigration\MySecondBrain\knowledge\信号质量.md"

VAULT_RESULTS = {
    "002050": "三花智控", "600038": "中直股份", "600416": "湘电股份",
    "000021": "深科技", "002185": "华天科技", "002156": "通富微电",
    "603501": "豪威集团", "000100": "TCL科技", "600183": "生益科技",
    "688981": "中芯国际", "688347": "华虹公司", "688017": "绿的谐波",
    "601689": "拓普集团", "603005": "晶方科技",
    "300033": "同花顺", "002463": "沪电股份", "688498": "源杰科技",
    "300308": "中际旭创", "688008": "澜起科技", "603986": "兆易创新",
}


def load():
    try:
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"predictions": []}


def save(data):
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_vault_result(pred, result):
    """验证通过的判断写入 vault 信号质量.md"""
    if not result or result.get("judgment_correct") is not True:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    avg = result["avg_change"]
    pred_date = pred["date"]
    pred_type = pred.get("prediction_type", "信号验证")
    detail = pred["judgment"]

    lines = [
        f"\n### {pred_date} 预测 → {today} 验证 — 判断正确",
        f"",
        f"**类型:** {pred_type}",
        f"**判断:** {detail}",
        f"**验证结果:** 平均 {avg:+.2f}%（判断正确）",
        f"**结论:** 该信号逻辑可靠，纳入经验库",
    ]
    if result.get("changes"):
        lines.append(f"")
        lines.append(f"个股表现:")
        for c in result["changes"][:8]:
            arr = "↑" if c["change_pct"] >= 0 else "↓"
            lines.append(f"- {c['name']:<8} {c['entry']:>8.2f} → {c['current']:>8.2f}  {arr}{abs(c['change_pct']):>6.2f}%")

    block = "\n".join(lines)

    try:
        with open(VAULT_SIGNAL_FILE, encoding="utf-8") as f:
            content = f.read()
        # 追加到"实战预测验证"区域
        marker = "## 实战预测验证"
        if marker in content:
            idx = content.index(marker)
            insert_point = content.find("\n---", idx)
            if insert_point == -1:
                insert_point = len(content)
            content = content[:insert_point] + block + "\n\n---\n" + content[insert_point:]
        else:
            content += f"\n\n---\n\n{marker}\n\n每日预测和验证记录。\n{block}\n"
        with open(VAULT_SIGNAL_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ 验证结论已写入 vault")
    except Exception as e:
        print(f"  ✗ 写入 vault 失败: {e}")


def check_prediction(pred):
    """验证一条预测，返回验证结果 dict"""
    fetcher = DataFetcher()
    changes = []
    total_change = 0.0
    valid_count = 0

    print(f"\n  验证: {pred['date']} 预测 → {pred['judgment'][:50]}...")

    pred_date = pd.Timestamp(pred["date"])
    for sym in pred["stocks"]:
        try:
            daily = fetcher.get_daily(sym, days=60)
            if daily is None or len(daily) < 2:
                continue
            date_diffs = (daily["date"] - pred_date).abs()
            entry_price = daily.loc[date_diffs.idxmin(), "close"]
            current_price = daily["close"].iloc[-1]
            change = (current_price - entry_price) / entry_price * 100
            name = VAULT_RESULTS.get(sym, sym)
            changes.append({
                "symbol": sym, "name": name,
                "entry": round(entry_price, 2),
                "current": round(current_price, 2),
                "change_pct": round(change, 2),
            })
            total_change += change
            valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        return {"status": "数据不足", "changes": [], "avg_change": None,
                "judgment_correct": None}

    avg_change = round(total_change / valid_count, 2)
    judgment_correct = avg_change < 0  # 判断为"下跌"，平均下跌即为判断正确

    return {
        "status": "已验证",
        "changes": changes,
        "avg_change": avg_change,
        "judgment_correct": judgment_correct,
    }


def verify_all():
    """扫描所有到期未验证的预测"""
    data = load()
    preds = data.get("predictions", [])
    if not preds:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    verified_any = False

    for pred in preds:
        if pred.get("verified") is True:
            continue
        if pred["verify_date"] > today:
            continue

        print(f"\n  {'='*50}")
        print(f"  预测验证: {pred['date']} → {pred['verify_date']}")
        result = check_prediction(pred)
        pred["verified"] = True
        pred["result"] = result
        verified_any = True

        if result["status"] == "数据不足":
            print(f"  ✗ 数据不足，无法验证")
        else:
            verdict = "✓ 判断正确" if result["judgment_correct"] else " 判断错误"
            print(f"  平均涨跌: {result['avg_change']:+.2f}% → {verdict}")
            for c in result["changes"][:5]:
                arr = "↑" if c["change_pct"] >= 0 else "↓"
                print(f"    {c['name']:<8} {c['entry']:>8.2f} → {c['current']:>8.2f}  {arr}{abs(c['change_pct']):>6.2f}%")

            # 如果判断正确，写入 vault 知识库
            if result["judgment_correct"]:
                _write_vault_result(pred, result)

    if verified_any:
        save(data)


def add_prediction(prediction_type, judgment, stocks, verify_days=10):
    """手动添加一条预测"""
    data = load()
    today = datetime.now().strftime("%Y-%m-%d")
    verify_date = (datetime.now() + timedelta(days=verify_days)).strftime("%Y-%m-%d")

    pred = {
        "date": today,
        "prediction_type": prediction_type,
        "observer": "系统",
        "judgment": judgment,
        "verify_date": verify_date,
        "verify_days": verify_days,
        "stocks": stocks,
        "verified": False,
        "result": None,
    }
    data["predictions"].append(pred)
    save(data)
    print(f"\n  ✓ 预测已记录，验证日期: {verify_date}")
    return pred


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add":
        args = sys.argv[2:]
        add_prediction(args[0] if len(args) > 0 else "手动验证",
                       args[1] if len(args) > 1 else "",
                       args[2].split(",") if len(args) > 2 else [],
                       int(args[3]) if len(args) > 3 else 10)
    else:
        verify_all()
