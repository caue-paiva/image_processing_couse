from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
    rng = np.random.default_rng(config.RANDOM_SEED)
    train, val, test = [], [], []
    removed = []
    for cls, idxs in sorted(by_class.items()):
        idxs = np.array(idxs, dtype=np.int32)
        if len(idxs) < 3:
            removed.append(cls)
            continue
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(0.10 * n)))
        n_val = max(1, int(round(0.10 * n)))
        if n - n_test - n_val < 1:
            n_test, n_val = 1, 1
        test.extend(idxs[:n_test].tolist())
        val.extend(idxs[n_test : n_test + n_val].tolist())
        train.extend(idxs[n_test + n_val :].tolist())
    split = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "removed_classes": removed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
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
            "is_individual": len(desc_names) == 1,
        }
        print(f"classificacao {combo_name}: val={best['val_acc']:.3f} test={test_acc:.3f}")
    fieldnames = ["method", "descriptors", "dim", "best_model", "val_acc", "test_acc", "f1_macro", "f1_weighted"]
    utils.write_csv(config.METRICS_DIR / "classification_results.csv", rows, fieldnames)
    (config.METRICS_DIR / "classification_results.md").write_text(utils.markdown_table(rows, fieldnames), encoding="utf-8")
    _write_confusion_figures(matrices)
    return rows


def _write_confusion_figures(matrices):
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


def _format_params(params):
    return ";".join(f"{k}={v}" for k, v in sorted(params.items()))

