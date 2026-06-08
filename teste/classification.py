from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import config
import features
import io_utils
import utils
import visualization


def make_split(records):
    by_class = defaultdict(list)
    for pos, record in enumerate(records):
        by_class[record.class_name].append(pos)
    removed = []
    valid = []
    for cls, idxs in sorted(by_class.items()):
        if len(idxs) < 3:
            removed.append(cls)
        else:
            valid.extend(idxs)
    valid = np.array(sorted(valid), dtype=np.int32)
    labels = np.array([records[i].class_name for i in valid], dtype=object)
    train, temp = train_test_split(
        valid,
        train_size=290,
        test_size=72,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    val, test = train_test_split(
        temp,
        train_size=36,
        test_size=36,
        random_state=config.RANDOM_SEED,
        stratify=None,
    )
    split = {
        "train": sorted([int(i) for i in train]),
        "val": sorted([int(i) for i in val]),
        "test": sorted([int(i) for i in test]),
        "removed_classes": removed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "strategy": "Global 80/10/10: train stratificado com 290 imagens; temporario de 72 imagens dividido em 36 validacao e 36 teste.",
    }
    utils.write_json(config.SPLITS_DIR / "classification_split.json", split)
    return split


def run_classification(records, fast=False):
    split = make_split(records)
    feature_map = features.load_all_features()
    y_all, _, _ = io_utils.labels_arrays(records)
    train_idx = np.array(split["train"], dtype=np.int32)
    val_idx = np.array(split["val"], dtype=np.int32)
    test_idx = np.array(split["test"], dtype=np.int32)
    rows = []
    matrices = {}
    combos = {"gch": ["gch"], "gabor": ["gabor"], "gch+lbp": ["gch", "lbp"]} if fast else config.COMBINATIONS
    for combo_name, desc_names in combos.items():
        X_train, _ = utils.combine_blocks(feature_map, desc_names, indices=train_idx, fit_indices=train_idx)
        X_val, _ = utils.combine_blocks(feature_map, desc_names, indices=val_idx, fit_indices=train_idx)
        X_test, _ = utils.combine_blocks(feature_map, desc_names, indices=test_idx, fit_indices=train_idx)
        y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
        best = None
        for params in config.CLASSIFICATION_GRID:
            model = SVC(**params)
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            val_acc = accuracy_score(y_val, pred_val)
            if best is None or val_acc > best["val_acc"]:
                best = {"params": params, "model": model, "val_acc": val_acc}
        pred_test = best["model"].predict(X_test)
        test_acc = accuracy_score(y_test, pred_test)
        f1_macro = f1_score(y_test, pred_test, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, pred_test, average="weighted", zero_division=0)
        dim = X_train.shape[1]
        row = {
            "method": combo_name,
            "descriptors": "+".join(desc_names),
            "dim": dim,
            "best_model": _format_params(best["params"]),
            "val_acc": f"{best['val_acc']:.4f}",
            "test_acc": f"{test_acc:.4f}",
            "f1_macro": f"{f1_macro:.4f}",
            "f1_weighted": f"{f1_weighted:.4f}",
        }
        rows.append(row)
        labels = sorted(np.unique(y_all[np.concatenate([train_idx, val_idx, test_idx])]).tolist())
        matrices[combo_name] = {
            "matrix": confusion_matrix(y_test, pred_test, labels=labels),
            "labels": labels,
            "test_acc": test_acc,
            "f1_macro": f1_macro,
            "is_individual": len(desc_names) == 1,
        }
        print(f"classificacao {combo_name}: val={best['val_acc']:.3f} test={test_acc:.3f}")
    fieldnames = ["method", "descriptors", "dim", "best_model", "val_acc", "test_acc", "f1_macro", "f1_weighted"]
    utils.write_csv(config.METRICS_DIR / "classification_results.csv", rows, fieldnames)
    (config.METRICS_DIR / "classification_results.md").write_text(utils.markdown_table(rows, fieldnames), encoding="utf-8")
    _write_confusion_figures(matrices)
    _write_confusion_summary(matrices)
    return rows


def _write_confusion_figures(matrices):
    for method, data in matrices.items():
        visualization.plot_confusion(
            data["matrix"],
            data["labels"],
            config.FIGURES_DIR / f"confusion_{_safe_name(method)}.png",
            f"Matriz de confusao - {method}",
        )
    if "gch" in matrices:
        visualization.plot_confusion(
            matrices["gch"]["matrix"],
            matrices["gch"]["labels"],
            config.FIGURES_DIR / "confusion_gch_baseline.png",
            "Matriz de confusao - GCH",
        )
    best_ind = max((v for v in matrices.values() if v["is_individual"]), key=lambda m: m["test_acc"])
    best_combo = max((v for v in matrices.values() if not v["is_individual"]), key=lambda m: m["test_acc"], default=best_ind)
    visualization.plot_confusion(
        best_ind["matrix"],
        best_ind["labels"],
        config.FIGURES_DIR / "confusion_best_individual.png",
        "Matriz de confusao - melhor descritor individual",
    )
    visualization.plot_confusion(
        best_combo["matrix"],
        best_combo["labels"],
        config.FIGURES_DIR / "confusion_best_combination.png",
        "Matriz de confusao - melhor combinacao",
    )


def _write_confusion_summary(matrices):
    rows = []
    for method, data in matrices.items():
        if not data["is_individual"]:
            continue
        matrix = data["matrix"]
        total = int(matrix.sum())
        correct = int(np.trace(matrix))
        rows.append({
            "method": method,
            "test_acc": f"{data['test_acc']:.4f}",
            "f1_macro": f"{data['f1_macro']:.4f}",
            "correct": correct,
            "total": total,
            "top_confusions": _top_confusions(matrix, data["labels"]),
        })
    fieldnames = ["method", "test_acc", "f1_macro", "correct", "total", "top_confusions"]
    utils.write_csv(config.METRICS_DIR / "confusion_summary.csv", rows, fieldnames)
    (config.METRICS_DIR / "confusion_summary.md").write_text(utils.markdown_table(rows, fieldnames), encoding="utf-8")


def _top_confusions(matrix, labels, limit=3):
    items = []
    for real_idx, pred_idx in np.argwhere(matrix > 0):
        if real_idx == pred_idx:
            continue
        items.append((int(matrix[real_idx, pred_idx]), labels[real_idx], labels[pred_idx]))
    if not items:
        return "sem confusoes"
    items.sort(key=lambda item: (-item[0], item[1], item[2]))
    return "; ".join(f"{real}->{pred} ({count})" for count, real, pred in items[:limit])


def _safe_name(name):
    return name.replace("+", "_").replace(" ", "_")


def _format_params(params):
    return ";".join(f"{k}={v}" for k, v in sorted(params.items()))
