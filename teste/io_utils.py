import csv
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import config


@dataclass(frozen=True)
class PetRecord:
    index: int
    class_id: int
    class_name: str
    filename: str
    path: Path


def prepare_data():
    config.ensure_output_dirs()
    if not config.PETS256_DIR.exists():
        _extract_zip(config.PETS256_ZIP, config.DATA_DIR)
    if not config.PETS_ORIGINAL_DIR.exists():
        _extract_zip(config.PETS_ORIGINAL_ZIP, config.DATA_DIR)
    records = load_records()
    validate_records(records)
    write_dataset_summary(records)
    return records


def _extract_zip(zip_path, target_dir):
    if not zip_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def load_records(limit=None):
    if not config.CSV_PATH.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {config.CSV_PATH}")
    records = []
    with config.CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for idx, row in enumerate(reader):
            clean = {k.strip(): v.strip() for k, v in row.items()}
            filename = clean["filename"]
            record = PetRecord(
                index=idx,
                class_id=int(clean["class_id"]),
                class_name=clean["class_name"],
                filename=filename,
                path=config.PETS256_DIR / filename,
            )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def validate_records(records):
    if not records:
        raise ValueError("Nenhum registro encontrado no CSV.")
    missing = [r.filename for r in records if not r.path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imagens ausentes. Exemplo: {missing[:3]}")
    if len(records) == 367:
        class_count = len({r.class_name for r in records})
        if class_count != 43:
            raise ValueError(f"Esperado 43 classes; encontrado {class_count}.")


def read_image(path):
    image = imageio.imread(path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Imagem com formato invalido: {path} {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Imagem sem 3 canais RGB: {path} {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def labels_arrays(records):
    y = np.array([r.class_name for r in records], dtype=object)
    files = np.array([r.filename for r in records], dtype=object)
    class_ids = np.array([r.class_id for r in records], dtype=np.int32)
    return y, files, class_ids


def write_dataset_summary(records):
    counts = Counter(r.class_name for r in records)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_images": len(records),
        "total_classes": len(counts),
        "min_images_per_class": min(counts.values()),
        "max_images_per_class": max(counts.values()),
        "classes_lt_3": sorted([k for k, v in counts.items() if v < 3]),
        "images_lt_3": sum(v for v in counts.values() if v < 3),
        "classification_classes": sum(v >= 3 for v in counts.values()),
        "classification_images": sum(v for v in counts.values() if v >= 3),
    }
    (config.METRICS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (config.METRICS_DIR / "class_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "count"])
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([name, count])

