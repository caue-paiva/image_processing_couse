import numpy as np

import config
import features
import io_utils
import utils
import visualization


def run_retrieval(records, fast=False, include_bovw=False):
    feature_map = features.load_all_features()
    y, files, _ = io_utils.labels_arrays(records)
    combos = {"gch": ["gch"], "gabor": ["gabor"], "gch+lbp": ["gch", "lbp"]} if fast else config.COMBINATIONS
    rows = []
    rankings_by_method = {}
    for combo_name, desc_names in combos.items():
        X, _ = utils.combine_blocks(feature_map, desc_names)
        metrics, rankings = evaluate_retrieval(X, y)
        row = {
            "method": combo_name,
            "descriptors": "+".join(desc_names),
            "dim": X.shape[1],
            "mAP": f"{metrics['mAP']:.4f}",
            "P@1": f"{metrics['P@1']:.4f}",
            "P@5": f"{metrics['P@5']:.4f}",
            "P@10": f"{metrics['P@10']:.4f}",
        }
        rows.append(row)
        rankings_by_method[combo_name] = rankings
        print(f"busca {combo_name}: mAP={metrics['mAP']:.4f} P@1={metrics['P@1']:.4f}")
    if include_bovw:
        X, _, _ = features.load_feature("bovw")
        metrics, rankings = evaluate_retrieval(X, y)
        rows.append({
            "method": "bovw",
            "descriptors": "bovw",
            "dim": X.shape[1],
            "mAP": f"{metrics['mAP']:.4f}",
            "P@1": f"{metrics['P@1']:.4f}",
            "P@5": f"{metrics['P@5']:.4f}",
            "P@10": f"{metrics['P@10']:.4f}",
        })
        rankings_by_method["bovw"] = rankings
    fieldnames = ["method", "descriptors", "dim", "mAP", "P@1", "P@5", "P@10"]
    path = config.METRICS_DIR / ("retrieval_results_with_bovw.csv" if include_bovw else "retrieval_results.csv")
    md_path = path.with_suffix(".md")
    utils.write_csv(path, rows, fieldnames)
    md_path.write_text(utils.markdown_table(rows, fieldnames), encoding="utf-8")
    if not include_bovw:
        best = max(rows, key=lambda r: float(r["mAP"]))["method"]
        write_retrieval_examples(records, y, files, rankings_by_method[best], best)
    return rows


def evaluate_retrieval(X, y):
    X = X.astype(np.float32)
    norms = np.sum(X * X, axis=1)
    dists = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    dists = np.maximum(dists, 0.0)
    np.fill_diagonal(dists, np.inf)
    rankings = np.argsort(dists, axis=1)
    p1, p5, p10, aps = [], [], [], []
    for i, ranking in enumerate(rankings):
        rel = (y[ranking] == y[i]).astype(np.float32)
        p1.append(float(rel[:1].mean()))
        p5.append(float(rel[:5].mean()))
        p10.append(float(rel[:10].mean()))
        total_relevant = int(np.sum(y == y[i]) - 1)
        if total_relevant <= 0:
            aps.append(0.0)
        else:
            hits = np.cumsum(rel)
            precisions = hits / (np.arange(len(rel)) + 1)
            aps.append(float(np.sum(precisions * rel) / total_relevant))
    return {
        "P@1": float(np.mean(p1)),
        "P@5": float(np.mean(p5)),
        "P@10": float(np.mean(p10)),
        "mAP": float(np.mean(aps)),
    }, rankings


def write_retrieval_examples(records, y, files, rankings, method):
    candidates = []
    for i, ranking in enumerate(rankings):
        top = ranking[:5]
        correct = int(np.sum(y[top] == y[i]))
        candidates.append((correct, i))
    success = [i for correct, i in sorted(candidates, reverse=True) if correct >= 3][:2]
    failure = [i for correct, i in sorted(candidates) if correct == 0][:1]
    selected = [("best_success_1", success[0] if success else candidates[-1][1])]
    selected.append(("best_success_2", success[1] if len(success) > 1 else candidates[-2][1]))
    selected.append(("best_failure_1", failure[0] if failure else candidates[0][1]))
    for name, idx in selected:
        result_idx = rankings[idx][:8]
        out = config.RETRIEVAL_EXAMPLES_DIR / f"{name}.png"
        visualization.plot_retrieval_grid(records, idx, result_idx, y, method, out)

