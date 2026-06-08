from collections import Counter

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import config
import io_utils


def plot_class_distribution(records):
    counts = Counter(r.class_name for r in records)
    names, vals = zip(*sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(vals)), vals, color="#3d6f8e")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Imagens")
    ax.set_title("Distribuicao de imagens por classe")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_confusion(matrix, labels, path, title):
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_retrieval_grid(records, query_idx, result_idx, y, method, out_path):
    images = [io_utils.read_image(records[query_idx].path)]
    titles = ["query"]
    borders = [(255, 220, 0)]
    for idx in result_idx:
        images.append(io_utils.read_image(records[idx].path))
        titles.append(records[idx].class_name)
        borders.append((0, 180, 80) if y[idx] == y[query_idx] else (210, 45, 45))
    tiles = [_tile_with_border((img * 255).astype(np.uint8), border) for img, border in zip(images, borders)]
    h, w, _ = tiles[0].shape
    canvas = np.full((h, w * len(tiles), 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        canvas[:, i * w : (i + 1) * w] = tile
    imageio.imwrite(out_path, canvas)
    fig, ax = plt.subplots(figsize=(14, 2.4))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"Busca {method} - query: {records[query_idx].class_name}")
    for i, title in enumerate(titles):
        ax.text((i + 0.5) / len(titles), -0.04, title, ha="center", va="top", transform=ax.transAxes, fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _tile_with_border(img, color, border=6):
    tile = np.full((img.shape[0] + 2 * border, img.shape[1] + 2 * border, 3), color, dtype=np.uint8)
    tile[border:-border, border:-border] = img
    return tile


def plot_bovw_embeddings(records):
    data = np.load(config.FEATURES_DIR / "bovw.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, init="pca", learning_rate="auto", perplexity=20)
    emb = tsne.fit_transform(X)
    _scatter_embedding(emb, y, config.FIGURES_DIR / "bovw_tsne_all.png", "t-SNE dos histogramas BoVW")
    _scatter_embedding_top(emb, y, config.FIGURES_DIR / "bovw_tsne_top_classes.png", "t-SNE BoVW - classes mais frequentes")
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED, n_jobs=1)
        emb_umap = reducer.fit_transform(X)
        _scatter_embedding(emb_umap, y, config.FIGURES_DIR / "bovw_umap_all.png", "UMAP dos histogramas BoVW")
    except Exception as exc:
        print(f"UMAP indisponivel: {exc}")


def _scatter_embedding(emb, y, path, title):
    labels = sorted(set(y.tolist()))
    cmap = plt.get_cmap("tab20", len(labels))
    label_to_num = {label: i for i, label in enumerate(labels)}
    colors = [label_to_num[v] for v in y]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap=cmap, s=22, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scatter_embedding_top(emb, y, path, title):
    counts = Counter(y.tolist())
    top = {name for name, _ in counts.most_common(10)}
    fig, ax = plt.subplots(figsize=(9, 7))
    mask_other = np.array([v not in top for v in y])
    ax.scatter(emb[mask_other, 0], emb[mask_other, 1], c="#d0d0d0", s=18, alpha=0.45, label="outras")
    for label in sorted(top):
        mask = np.array([v == label for v in y])
        ax.scatter(emb[mask, 0], emb[mask, 1], s=28, alpha=0.9, label=label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
