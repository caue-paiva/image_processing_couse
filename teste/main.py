import argparse
import os

import bovw
import classification
import config
import features
import io_utils
import report_assets
import retrieval
import visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline do Trabalho 3 de Processamento de Imagens")
    parser.add_argument(
        "command",
        choices=["prepare", "extract", "classify", "retrieve", "bovw", "visualize", "report-assets", "validate", "all"],
    )
    parser.add_argument("--force", action="store_true", help="Recalcula caches existentes.")
    parser.add_argument("--fast", action="store_true", help="Executa subconjunto menor para teste rapido.")
    parser.add_argument("--limit", type=int, default=None, help="Limita numero de imagens para teste.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(config.MPLCONFIGDIR))
    config.ensure_output_dirs()
    records = io_utils.prepare_data()
    if args.limit is not None and args.command != "prepare":
        raise ValueError("--limit deve ser usado apenas com prepare para nao criar caches parciais.")
    if args.limit is not None:
        records = records[: args.limit]
    if args.command == "prepare":
        visualization.plot_class_distribution(records)
        print("prepare concluido")
        return
    if args.command in ("extract", "all"):
        visualization.plot_class_distribution(records)
        features.extract_features(records, force=args.force)
    if args.command in ("classify", "all"):
        _ensure_features(records, args.force)
        classification.run_classification(records, fast=args.fast)
    if args.command in ("retrieve", "all"):
        _ensure_features(records, args.force)
        retrieval.run_retrieval(records, fast=args.fast)
    if args.command in ("bovw", "all"):
        bovw.build_bovw(records, fast=args.fast, force=args.force)
    if args.command in ("visualize", "all"):
        if not (config.FEATURES_DIR / "bovw.npz").exists():
            bovw.build_bovw(records, fast=args.fast, force=args.force)
        visualization.plot_bovw_embeddings(records)
    if args.command in ("report-assets", "all"):
        report_assets.build_report_summary()
    if args.command in ("validate", "all"):
        report_assets.scan_for_forbidden_imports()
        report_assets.assert_expected_outputs()
        print("validacao concluida")


def _ensure_features(records, force):
    missing = [name for name in config.DESCRIPTORS if not (config.FEATURES_DIR / f"{name}.npz").exists()]
    if missing:
        print(f"features ausentes: {missing}. Extraindo antes da etapa.")
        features.extract_features(records, descriptor_names=missing, force=force)


if __name__ == "__main__":
    main()
