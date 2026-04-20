import json, config
from pathlib import Path
with open(Path(config.BENCHMARK_DIR) / 'benchmark.json', encoding='utf-8') as f:
    data = json.load(f)
real = [r for r in data if r['label'] == 'REAL']
print('REAL count:', len(real))
print('Sample:', real[0])
