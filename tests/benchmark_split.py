import json, config
from pathlib import Path
with open(Path(config.BENCHMARK_DIR) / 'benchmark.json', encoding='utf-8') as f:
    data = json.load(f)

labels = {}
subtypes = {}
corruption_types = {}

for r in data:
    l = r['label']
    s = r.get('subtype') or 'REAL'
    c = r.get('corruption_type') or 'none'
    labels[l] = labels.get(l, 0) + 1
    subtypes[s] = subtypes.get(s, 0) + 1
    corruption_types[c] = corruption_types.get(c, 0) + 1

print('--- Labels ---')
for k,v in sorted(labels.items()): print(f'  {k}: {v}')
print('--- Subtypes ---')
for k,v in sorted(subtypes.items()): print(f'  {k}: {v}')
print('--- Corruption types ---')
for k,v in sorted(corruption_types.items()): print(f'  {k}: {v}')
