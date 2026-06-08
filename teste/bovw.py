import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import config
import descriptors
import io_utils
import retrieval
import utils


def patch_descriptors(image):
    gray = descriptors.rgb_to_gray(image)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)
    bins = np.minimum((ang / 22.5).astype(np.int32), 7)
    rows = []
    ps = config.BOVW_PATCH_SIZE
    step = config.BOVW_PATCH_STEP
    for y0 in range(0, gray.shape[0] - ps + 1, step):
        for x0 in range(0, gray.shape[1] - ps + 1, step):
            ys = slice(y0, y0 + ps)
            xs = slice(x0, x0 + ps)
            hist = np.bincount(bins[ys, xs].ravel(), weights=mag[ys, xs].ravel(), minlength=8).astype(np.float32)
            extra = np.array([gray[ys, xs].mean(), gray[ys, xs].std()], dtype=np.float32)
            vec = np.concatenate([hist, extra])
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)
    return np.vstack(rows).astype(np.float32)


def build_bovw(records, fast=False, force=False):
    config.ensure_output_dirs()
    out = config.FEATURES_DIR / "bovw.npz"
    if out.exists() and not force:
        return out
    y, files, class_ids = io_utils.labels_arrays(records)
    per_image = []
    all_desc = []
    for i, record in enumerate(records, start=1):
        image = io_utils.read_image(record.path)
        desc = patch_descriptors(image)
        per_image.append(desc)
        all_desc.append(desc)
        if i % 50 == 0:
            print(f"bovw patches: {i}/{len(records)} imagens")
    all_desc = np.vstack(all_desc)
    rng = np.random.default_rng(config.RANDOM_SEED)
    max_patches = 5_000 if fast else config.BOVW_MAX_PATCHES
    if len(all_desc) > max_patches:
        sample_idx = rng.choice(len(all_desc), size=max_patches, replace=False)
        train_desc = all_desc[sample_idx]
    else:
        train_desc = all_desc
    k = config.BOVW_FAST_K if fast else config.BOVW_K
    kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=5, max_iter=100)
    kmeans.fit(train_desc)
    hists = []
    for desc in per_image:
        words = kmeans.predict(desc)
        hist = np.bincount(words, minlength=k).astype(np.float32)
        hists.append(hist)
    X = normalize(np.vstack(hists), norm="l1").astype(np.float32)
    np.savez_compressed(out, X=X, y=y, files=files, class_ids=class_ids, descriptor_name="bovw", k=np.array([k]))
    rows = []
    metrics, _ = retrieval.evaluate_retrieval(X, y)
    rows.append({
        "method": "bovw",
        "k": k,
        "dim": X.shape[1],
        "mAP": f"{metrics['mAP']:.4f}",
        "P@1": f"{metrics['P@1']:.4f}",
        "P@5": f"{metrics['P@5']:.4f}",
        "P@10": f"{metrics['P@10']:.4f}",
    })
    fieldnames = ["method", "k", "dim", "mAP", "P@1", "P@5", "P@10"]
    utils.write_csv(config.METRICS_DIR / "bovw_results.csv", rows, fieldnames)
    (config.METRICS_DIR / "bovw_results.md").write_text(utils.markdown_table(rows, fieldnames), encoding="utf-8")
    print(f"bovw: salvo em {out} com shape {X.shape}")
    return out

