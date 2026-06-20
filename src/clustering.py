#!/usr/bin/python3
#
# clustering.py: Aprendizado NÃO supervisionado para investigar padrões
#                espaço-temporais — de forma ACADÊMICA, não "torturando os
#                números até alguma hipótese funcionar":
#                  - testa hipóteses (vários k, vários parâmetros);
#                  - compara métricas (silhouette, Davies-Bouldin, Calinski-H.);
#                  - observa padrões e REJEITA hipóteses fracas;
#                  - declara limitações.
#
#   Features: latitude/longitude + hora cíclica (sin/cos), escalonadas.
#
#   Algoritmos:
#     - KMeans (MiniBatchKMeans no dataset completo) com elbow + 3 métricas;
#     - DBSCAN e HDBSCAN (em amostra, por custo de memória/tempo).
#
#   Saídas: elbow_kmeans.png, silhouette_kmeans.png, mapa_clusters.html,
#           clusters_3d.html, cluster_summary.csv, clustering_metrics.json.

from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# HDBSCAN só existe a partir do scikit-learn 1.3. O projeto fixa 1.8 no
# requirements, mas mantemos um fallback: se a versão instalada não tiver
# HDBSCAN, o pipeline avisa e segue com KMeans + DBSCAN.
try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except ImportError:  # scikit-learn < 1.3
    _HAS_HDBSCAN = False

from config import (
    FIGURES_DIR,
    MAPS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    CLUSTER_DENSITY_SAMPLE,
    MAP_POINT_SAMPLE,
    SP_CENTER,
    PERIODO_LABEL,
    rel_to_root,
)
from features import build_cluster_matrix

_CLUSTER_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324", "#808000",
]


# ---------------------------------------------------------------------------
# Avaliação do KMeans (elbow + métricas)
# ---------------------------------------------------------------------------
def evaluate_kmeans(X: np.ndarray, k_range=range(2, 11),
                    silhouette_sample: int = 20000) -> pd.DataFrame:
    """
    Ajusta MiniBatchKMeans para cada k e calcula:
      - inertia (para o elbow);
      - silhouette (em amostra — métrica O(n²), inviável em 1,8M);
      - Davies-Bouldin (menor é melhor);
      - Calinski-Harabasz (maior é melhor).
    Salva elbow_kmeans.png e silhouette_kmeans.png.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(silhouette_sample, len(X)), replace=False)
    Xs = X[idx]

    rows = []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             n_init=3, batch_size=4096)
        labels_full = km.fit_predict(X)
        labels_s = labels_full[idx]
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Xs, labels_s)),
            "davies_bouldin": float(davies_bouldin_score(Xs, labels_s)),
            "calinski_harabasz": float(calinski_harabasz_score(Xs, labels_s)),
        })
        print(f"  k={k}: silhouette={rows[-1]['silhouette']:.3f} "
              f"DB={rows[-1]['davies_bouldin']:.3f} "
              f"CH={rows[-1]['calinski_harabasz']:.0f}")

    metrics = pd.DataFrame(rows)

    # Elbow (inertia)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(metrics["k"], metrics["inertia"], "o-", color="#2b6cb0")
    ax.set_title("Método do cotovelo (elbow) — KMeans")
    ax.set_xlabel("k (nº de clusters)")
    ax.set_ylabel("Inércia (within-cluster SSE)")
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "elbow_kmeans.png", bbox_inches="tight")
    plt.close(fig)

    # Silhouette + DB + CH
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(metrics["k"], metrics["silhouette"], "o-", color="#2b6cb0")
    axes[0].set_title("Silhouette (↑ melhor)")
    axes[1].plot(metrics["k"], metrics["davies_bouldin"], "o-", color="#dd6b20")
    axes[1].set_title("Davies-Bouldin (↓ melhor)")
    axes[2].plot(metrics["k"], metrics["calinski_harabasz"], "o-", color="#2f855a")
    axes[2].set_title("Calinski-Harabasz (↑ melhor)")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    fig.suptitle("Métricas internas de clusterização vs k")
    fig.savefig(FIGURES_DIR / "silhouette_kmeans.png", bbox_inches="tight")
    plt.close(fig)

    return metrics


def escolher_k(metrics: pd.DataFrame) -> int:
    """Escolhe k pelo maior silhouette no intervalo testado (critério simples e
    transparente). A escolha e suas limitações ficam documentadas."""
    return int(metrics.loc[metrics["silhouette"].idxmax(), "k"])


def fit_final_kmeans(X: np.ndarray, k: int):
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                         n_init=5, batch_size=4096)
    labels = km.fit_predict(X)
    return labels, km


# ---------------------------------------------------------------------------
# Resumo por cluster
# ---------------------------------------------------------------------------
def cluster_summary(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    """Resumo estatístico por cluster: tamanho, centróide, hora média, período
    dominante, top-3 crimes."""
    rows = []
    total = len(df)
    for c, g in df.groupby(label_col):
        top_crimes = g["crime"].value_counts().head(3)
        top_str = "; ".join(f"{n} ({v})" for n, v in top_crimes.items())
        periodo_dom = g["periodo"].value_counts().idxmax()
        rows.append({
            "cluster": int(c),
            "n": int(len(g)),
            "pct": round(100 * len(g) / total, 2),
            "lat_centro": round(float(g["lat"].mean()), 5),
            "lon_centro": round(float(g["lon"].mean()), 5),
            "hora_media": round(float(g["hora"].mean()), 1),
            "hora_mediana": float(g["hora"].median()),
            "periodo_dominante": PERIODO_LABEL.get(periodo_dom, periodo_dom),
            "top3_crimes": top_str,
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.to_csv(REPORTS_DIR / "cluster_summary.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# DBSCAN / HDBSCAN (em amostra)
# ---------------------------------------------------------------------------
def run_density_based(X_sample: np.ndarray) -> dict:
    """
    Roda DBSCAN e HDBSCAN em amostra (custo proibitivo em 1,8M pontos) e
    documenta nº de clusters e ruído para cada parametrização testada.
    """
    out = {"amostra": int(len(X_sample)), "dbscan": [], "hdbscan": []}

    for eps in (0.2, 0.3, 0.5):
        for min_samples in (10, 25):
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X_sample)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            out["dbscan"].append({
                "eps": eps, "min_samples": min_samples,
                "n_clusters": int(n_clusters),
                "ruido_pct": round(100 * n_noise / len(labels), 1),
            })
            print(f"  DBSCAN eps={eps} min_samples={min_samples}: "
                  f"{n_clusters} clusters, {out['dbscan'][-1]['ruido_pct']}% ruído")

    if not _HAS_HDBSCAN:
        out["hdbscan_disponivel"] = False
        print("  HDBSCAN indisponível nesta versão do scikit-learn (<1.3): "
              "etapa pulada; KMeans e DBSCAN seguem normalmente.")
        return out

    out["hdbscan_disponivel"] = True
    for mcs in (100, 250, 500):
        hb = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X_sample)
        labels = hb.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        out["hdbscan"].append({
            "min_cluster_size": mcs,
            "n_clusters": int(n_clusters),
            "ruido_pct": round(100 * n_noise / len(labels), 1),
        })
        print(f"  HDBSCAN min_cluster_size={mcs}: "
              f"{n_clusters} clusters, {out['hdbscan'][-1]['ruido_pct']}% ruído")

    return out


# ---------------------------------------------------------------------------
# Mapa de clusters
# ---------------------------------------------------------------------------
def mapa_clusters(df: pd.DataFrame, summary: pd.DataFrame,
                  label_col: str = "cluster") -> str:
    """Mapa folium: amostra de pontos colorida por cluster + centróides."""
    n = min(MAP_POINT_SAMPLE, len(df))
    s = df.sample(n, random_state=RANDOM_STATE)
    m = folium.Map(location=list(SP_CENTER), zoom_start=11,
                   tiles="CartoDB positron", control_scale=True)

    for lat, lon, c in zip(s["lat"], s["lon"], s[label_col]):
        color = _CLUSTER_COLORS[int(c) % len(_CLUSTER_COLORS)]
        folium.CircleMarker([lat, lon], radius=1.5, color=color,
                            fill=True, fill_opacity=0.5, weight=0).add_to(m)

    # Centróides com popup do resumo
    for _, r in summary.iterrows():
        c = int(r["cluster"])
        color = _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]
        popup = (f"<b>Cluster {c}</b><br>n={r['n']:,} ({r['pct']}%)<br>"
                 f"hora média: {r['hora_media']}h<br>"
                 f"período: {r['periodo_dominante']}<br>"
                 f"top crimes: {r['top3_crimes']}").replace(",", ".")
        folium.Marker(
            [r["lat_centro"], r["lon_centro"]],
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color="black", icon="info-sign"),
        ).add_to(m)

    out = MAPS_DIR / "mapa_clusters.html"
    m.save(str(out))
    return str(out)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def run_all(df_clean: pd.DataFrame | None = None) -> dict:
    from data_loader import load_clean, get_exploratory_df
    from viz3d import clusters_3d

    if df_clean is None:
        df_clean = load_clean()
    exp = get_exploratory_df(df_clean)

    print("Construindo matriz de features (lat, lon, hora_sin, hora_cos)...")
    X, scaler, names = build_cluster_matrix(exp)

    print("Avaliando KMeans (elbow + métricas)...")
    metrics = evaluate_kmeans(X)
    k = escolher_k(metrics)
    print(f"k escolhido (maior silhouette): {k}")

    labels, km = fit_final_kmeans(X, k)
    exp = exp.copy()
    exp["cluster"] = labels

    summary = cluster_summary(exp)
    mapa = mapa_clusters(exp, summary)
    c3d = clusters_3d(exp, label_col="cluster", out_name="clusters_3d.html")

    print("Rodando DBSCAN/HDBSCAN em amostra...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(CLUSTER_DENSITY_SAMPLE, len(X)), replace=False)
    densidade = run_density_based(X[idx])

    resultado = {
        "k_escolhido": k,
        "kmeans_metrics": metrics.to_dict(orient="records"),
        "silhouette_k": float(metrics.loc[metrics["k"] == k, "silhouette"].iloc[0]),
        "densidade": densidade,
        "n_exploratorio": int(len(exp)),
        "features": names,
        "arquivos": {
            "elbow": rel_to_root(FIGURES_DIR / "elbow_kmeans.png"),
            "silhouette": rel_to_root(FIGURES_DIR / "silhouette_kmeans.png"),
            "mapa_clusters": rel_to_root(mapa),
            "clusters_3d": rel_to_root(c3d),
            "cluster_summary": rel_to_root(REPORTS_DIR / "cluster_summary.csv"),
        },
    }
    with open(REPORTS_DIR / "clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    return resultado


if __name__ == "__main__":
    res = run_all()
    print("\n== RESUMO ==")
    print(json.dumps({k: v for k, v in res.items() if k != "kmeans_metrics"},
                     indent=2, ensure_ascii=False))
