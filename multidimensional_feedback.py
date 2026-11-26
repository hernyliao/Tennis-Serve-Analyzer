import csv
import re
import itertools
from typing import List, Dict, Tuple
import numpy as np

def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

def extract_player_from_id(serve_id: str) -> str:
    m = re.match(r"^([A-Za-z]+)", serve_id)
    if m:
        return m.group(1)
    return serve_id.split('_')[0]

def linear_regression_r2(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(y)
    for col in range(X.shape[1]):
        mask &= np.isfinite(X[:, col])

    if mask.sum() <= X.shape[1]:
        return float('nan'), None

    X2 = X[mask]
    y2 = y[mask]

    # add intercept column
    X_design = np.column_stack([np.ones(len(X2)), X2])
    beta, _, _, _ = np.linalg.lstsq(X_design, y2, rcond=None)

    y_pred = X_design @ beta

    ss_res = np.sum((y2 - y_pred) ** 2)
    ss_tot = np.sum((y2 - np.mean(y2)) ** 2)

    if ss_tot == 0:
        return 0.0, beta

    r2 = 1 - ss_res / ss_tot
    return float(r2), beta


def analyze_per_player(path: str, top_n: int = 5, min_serves: int = 8, max_features: int = 3):
    rows = load_rows(path)

    all_fields = list(rows[0].keys())
    feature_fields = [
        f for f in all_fields
        if f not in ('serve_id', 'serve_speed_label')
    ]

    # group rows by player
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        pid = extract_player_from_id(r.get('serve_id', ''))
        groups.setdefault(pid, []).append(r)

    for player, prow in sorted(groups.items()):

        n_serves = len(prow)

        if n_serves < min_serves:
            print(f"\nSkipping player {player}: only {n_serves} serves (min {min_serves})")
            continue

        print(f"\n{'='*60}")
        print(f"Player: {player}  (n_serves={n_serves})")
        print(f"{'='*60}")

        feature_data = {}
        for f in feature_fields:
            feature_data[f] = np.array(
                [float(r.get(f, float('nan'))) for r in prow],
                dtype=float
            )

        y = np.array(
            [float(r.get('serve_speed_label', float('nan'))) for r in prow],
            dtype=float
        )

        results: List[Tuple[Tuple[str, ...], float, np.ndarray]] = []

        for k in range(1, max_features + 1):
            for combo in itertools.combinations(feature_fields, k):
                X = np.column_stack([feature_data[f] for f in combo])
                r2, beta = linear_regression_r2(X, y)

                if not np.isnan(r2):
                    results.append((combo, r2, beta))

        results.sort(key=lambda x: x[1], reverse=True)

        for i, (combo, r2, beta) in enumerate(results[:top_n], start=1):

            print(f"\n{i}. Features: {combo}")
            print(f"   R^2 = {r2:.4f}")

            for j, feat in enumerate(combo):
                coef = beta[j + 1]                                                                                                                                                                                                                                                                                                  
                direction = "increase" if coef > 0 else "decrease"
                print(f"   {feat}: coef={coef:.4f} → try to {direction}")

analyze_per_player(path='serve_features.csv', top_n=5, min_serves=8, max_features=2)