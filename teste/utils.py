import csv
import json

import numpy as np
from sklearn.preprocessing import StandardScaler


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(rows, columns):
    if not rows:
        return ""
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def combine_blocks(feature_map, names, indices=None, fit_indices=None):
    arrays = []
    scalers = []
    for name in names:
        X = feature_map[name]
        scaler = StandardScaler()
        fit_X = X if fit_indices is None else X[fit_indices]
        scaler.fit(fit_X)
        Xt = scaler.transform(X if indices is None else X[indices])
        arrays.append(Xt)
        scalers.append(scaler)
    return np.hstack(arrays).astype(np.float32), scalers

