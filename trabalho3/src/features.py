import time

import numpy as np

import config
import descriptors
import io_utils


def extract_features(records, descriptor_names=None, force=False):
    config.ensure_output_dirs()
    descriptor_names = descriptor_names or config.DESCRIPTORS
    y, files, class_ids = io_utils.labels_arrays(records)
    outputs = {}
    for name in descriptor_names:
        path = config.FEATURES_DIR / f"{name}.npz"
        if path.exists() and not force:
            outputs[name] = np.load(path, allow_pickle=True)["X"]
            continue
        func = descriptors.DESCRIPTOR_FUNCS[name]
        rows = []
        start = time.time()
        for i, record in enumerate(records, start=1):
            image = io_utils.read_image(record.path)
            rows.append(func(image))
            if i % 50 == 0:
                print(f"{name}: {i}/{len(records)} imagens")
        X = np.vstack(rows).astype(np.float32)
        np.savez_compressed(
            path,
            X=X,
            y=y,
            files=files,
            class_ids=class_ids,
            descriptor_name=name,
            seconds=np.array([time.time() - start], dtype=np.float32),
        )
        outputs[name] = X
        print(f"{name}: salvo em {path} com shape {X.shape}")
    return outputs


def load_feature(name):
    path = config.FEATURES_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Features ausentes: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["files"]


def load_all_features(names=None):
    names = names or config.DESCRIPTORS
    return {name: load_feature(name)[0] for name in names}

