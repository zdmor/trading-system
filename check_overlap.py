"""Quick overlap check: our candidates vs scanner top-50."""
import sys
sys.path.insert(0, r'D:\ClaudeWorkspace\trading_system')
from scanner import Scanner

# Get our candidates (same logic as new get_candidates)
s = Scanner()
s.fetch_all_stocks()
candidates = s.filter_candidates()

bj_prefixes = ("8", "4", "9")
seen = set()
our_codes = []
for c in candidates:
    code = c["code"]
    name = c.get("name", "")
    sym = code.split(".")[-1] if "." in code else code
    if sym.startswith(bj_prefixes) or code.startswith("bj"):
        continue
    if "ST" in name or "*ST" in name:
        continue
    if code in seen:
        continue
    seen.add(code)
    our_codes.append(code)

our_set = set(our_codes[:2000])
print(f"Our candidates: {len(our_codes)} total")

# Run scanner quick mode to get top 50
s2 = Scanner()
s2.run(min_amount=500000000, max_analysis=50, top_n=50, quick=True)
top50 = [r["code"] for r in s2.results[:50]]
top50_set = set(top50)

overlap = our_set & top50_set
print(f"Scanner top 50: {len(top50)}")
print(f"Overlap: {len(overlap)}/50")
print(f"Overlapping: {sorted(overlap)[:20]}")
print(f"\nIn top50 but NOT in ours: {top50_set - our_set}")