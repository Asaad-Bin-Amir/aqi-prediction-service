"""Diagnose data collection issues"""
from src.feature_store import AQIFeatureStore
import pandas as pd
from datetime import timedelta

print("\n" + "="*70)
print("🔍 DATA GAP ANALYSIS")
print("="*70)

with AQIFeatureStore() as fs:
    data = list(fs.raw_features.find({}).sort('timestamp', 1))

if not data:
    print("❌ No data found!")
    exit()

df = pd.DataFrame(data)

print(f"\n📊 Basic Stats:")
print(f"   Total records: {len(df)}")
print(f"   First: {df['timestamp'].min()}")
print(f"   Last: {df['timestamp'].max()}")
print(f"   Time span: {df['timestamp'].max() - df['timestamp'].min()}")

# Calculate gaps
df = df.sort_values('timestamp').reset_index(drop=True)
df['gap'] = df['timestamp'].diff()

print(f"\n⏱️ Gap Analysis:")
print(f"   Smallest gap: {df['gap'].min()}")
print(f"   Largest gap: {df['gap'].max()}")
print(f"   Average gap: {df['gap'].mean()}")
print(f"   Median gap: {df['gap'].median()}")

# Find large gaps
large_gaps = df[df['gap'] > timedelta(hours=2)]
print(f"\n🚨 Gaps larger than 2 hours: {len(large_gaps)}")

if len(large_gaps) > 0:
    print("\n   Showing first 10 large gaps:")
    for idx, row in large_gaps.head(10).iterrows():
        print(f"   {row['timestamp']}: {row['gap']}")

# Check consecutive sequences
print(f"\n🔗 Consecutive Data Check:")
consecutive = []
current_seq = 1

for i in range(1, len(df)):
    gap = df.loc[i, 'gap']
    if gap <= timedelta(hours=1.5):
        current_seq += 1
    else:
        if current_seq > 1:
            consecutive.append(current_seq)
        current_seq = 1

if current_seq > 1:
    consecutive.append(current_seq)

if consecutive:
    print(f"   Longest consecutive sequence: {max(consecutive)} records")
    print(f"   Number of sequences: {len(consecutive)}")
    print(f"   Sequences: {consecutive}")
else:
    print(f"   ❌ No consecutive sequences found!")

# Per-day analysis
df['date'] = pd.to_datetime(df['timestamp']).dt.date
daily = df.groupby('date').size().sort_index()

print(f"\n📅 Records per day:")
for date, count in daily.items():
    status = '✅' if count >= 20 else '⚠️' if count >= 10 else '❌'
    print(f"   {status} {date}: {count} records")

print("\n" + "="*70)