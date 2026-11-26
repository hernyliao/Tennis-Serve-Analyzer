#ADD DROP TO CONTACT FRAME COUNT AS FEATURE
#add multidimentional correlation

import os
import csv
import re
from typing import List, Dict, Tuple
import numpy as np

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
	mask = np.isfinite(x) & np.isfinite(y)
	if mask.sum() < 2:
		return float('nan')
	x2 = x[mask]
	y2 = y[mask]

	return float(np.corrcoef(x2, y2)[0, 1])

def load_rows(path: str) -> List[Dict[str, str]]:
	with open(path, newline='') as f:
		reader = csv.DictReader(f)
		return list(reader)

def extract_player_from_id(serve_id: str) -> str:
	m = re.match(r"^([A-Za-z]+)", serve_id)
	if m:
		return m.group(1)
	return serve_id.split('_')[0]

def analyze_per_player(path: str = 'serve_features.csv', top_n: int = 5, min_serves: int = 3):
	rows = load_rows(path)

	# collect numeric feature names
	all_fields = list(rows[0].keys())

	feature_fields = [f for f in all_fields if f not in ('serve_id', 'serve_speed_label')]
	# group rows by player
	groups: Dict[str, List[Dict[str, str]]] = {}
	for r in rows:
		pid = extract_player_from_id(r.get('serve_id', ''))
		groups.setdefault(pid, []).append(r)

	for player, prow in sorted(groups.items()):
		if len(prow) < min_serves:
			continue

		speed = np.array([float(r.get('serve_speed_label', float('nan'))) for r in prow], dtype=float)
		results: List[Tuple[str, float]] = []
		for f in feature_fields:
			arr = np.array([float(r.get(f, float('nan'))) for r in prow], dtype=float)
			rcoef = pearsonr_safe(arr, speed)
			results.append((f, rcoef))

		results.sort(key=lambda x: abs(x[1]), reverse=True)
		print(f"\nPlayer: {player}  (n_serves={len(prow)})")
		for i, (feat, rcoef) in enumerate(results[:top_n], start=1):
			direction = 'increase' if rcoef > 0 else 'decrease'
			print(f"  {i}. {feat}: r={rcoef:.3f} -> recommendation: try to {direction} {feat} to increase speed")

analyze_per_player(path='serve_features.csv', top_n=5, min_serves=3)
