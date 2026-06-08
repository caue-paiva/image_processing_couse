from pathlib import Path

import config


def build_report_summary():
    sections = ["# Artefatos para o relatorio\n"]
    for title, path in [
        ("Classificacao", config.METRICS_DIR / "classification_results.md"),
        ("Resumo das matrizes de confusao", config.METRICS_DIR / "confusion_summary.md"),
        ("Busca", config.METRICS_DIR / "retrieval_results.md"),
        ("BoVW", config.METRICS_DIR / "bovw_results.md"),
    ]:
        sections.append(f"## {title}\n")
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
        else:
            sections.append(f"`{path}` ainda nao foi gerado.\n")
        sections.append("")
    sections.append("## Figuras principais\n")
    figure_names = [
        "class_distribution.png",
        "confusion_best_individual.png",
        "confusion_best_combination.png",
        "confusion_gch_baseline.png",
        "confusion_gch.png",
        "confusion_lbp.png",
        "confusion_glcm.png",
        "confusion_hog.png",
        "confusion_correlogram.png",
        "confusion_gabor.png",
        "confusion_gch_lbp.png",
        "confusion_gch_glcm.png",
        "confusion_gch_lbp_glcm.png",
        "confusion_gch_lbp_glcm_gabor.png",
        "confusion_all.png",
        "bovw_tsne_all.png",
        "bovw_umap_all.png",
        "bovw_tsne_top_classes.png",
    ]
    for name in figure_names:
        path = config.FIGURES_DIR / name
        status = "ok" if path.exists() else "pendente"
        sections.append(f"- `{path.relative_to(config.ROOT)}`: {status}")
    sections.append("\n## Exemplos de busca\n")
    for path in sorted(config.RETRIEVAL_EXAMPLES_DIR.glob("*.png")):
        sections.append(f"- `{path.relative_to(config.ROOT)}`")
    out = config.METRICS_DIR / "report_summary.md"
    out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return out


def assert_expected_outputs():
    expected = [
        config.METRICS_DIR / "dataset_summary.json",
        config.METRICS_DIR / "class_distribution.csv",
        config.FIGURES_DIR / "class_distribution.png",
        config.METRICS_DIR / "classification_results.csv",
        config.METRICS_DIR / "retrieval_results.csv",
        config.METRICS_DIR / "bovw_results.csv",
        config.FIGURES_DIR / "bovw_tsne_all.png",
        config.METRICS_DIR / "report_summary.md",
    ]
    missing = [str(p.relative_to(config.ROOT)) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError("Outputs esperados ausentes: " + ", ".join(missing))


def scan_for_forbidden_imports():
    names = ["cv2", "PIL", "pillow"]
    forbidden = [f"{prefix} {name}" for name in names for prefix in ("import", "from")]
    hits = []
    for path in (config.ROOT / "src").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for item in forbidden:
            if item in text:
                hits.append(f"{path.name}: {item}")
    if hits:
        raise RuntimeError("Imports proibidos encontrados: " + ", ".join(hits))
