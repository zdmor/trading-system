import pandas as pd, os
cache_dir = r'D:\ClaudeWorkspace\trading_system\data_cache\daily'
WINDOW_START = '2023-06-01'
WINDOW_END = '2026-05-27'

latest_min = None
earliest_max = None
short_count = 0
total = 0

for f in os.listdir(cache_dir):
    if not f.endswith('.csv'):
        continue
    total += 1
    df = pd.read_csv(os.path.join(cache_dir, f), parse_dates=['date'])
    df = df[(df['date'] >= pd.Timestamp(WINDOW_START)) & (df['date'] <= pd.Timestamp(WINDOW_END))]
    if len(df) < 120:
        short_count += 1
        continue
    dmin = df['date'].min().date()
    dmax = df['date'].max().date()
    if latest_min is None or dmin > latest_min:
        latest_min = dmin
        latest_min_file = f
    if earliest_max is None or dmax < earliest_max:
        earliest_max = dmax
        earliest_max_file = f

print(f'Total files: {total}')
print(f'Short (<120): {short_count}')
print(f'Latest min_date: {latest_min} (file: {latest_min_file})')
print(f'Earliest max_date: {earliest_max} (file: {earliest_max_file})')
if latest_min and earliest_max:
    days = (earliest_max - latest_min).days
    print(f'Intersection span: {days} calendar days')
    print(f'Est. trading days: {days * 5/7:.0f}')