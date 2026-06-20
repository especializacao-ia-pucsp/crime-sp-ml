#!/usr/bin/python3
#
# visualization.py: Gráficos da Análise Exploratória de Dados (EDA).
#
#   Gera figuras estáticas (PNG) em reports/figures a partir do dataset limpo:
#     - contagem de crimes por NATUREZA_APURADA (e top-15);
#     - desbalanceamento de classes (escala log);
#     - distribuição por hora do dia e por período;
#     - distribuição geográfica (dispersão amostrada) e densidade (hexbin);
#     - assinatura temporal por tipo de crime (heatmap hora x crime).
#
#   Aggregações usam o dataset completo; apenas a dispersão de pontos usa
#   amostra (limite de renderização). Todos os caps estão explícitos.

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend headless (sem display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURES_DIR,
    RANDOM_STATE,
    PERIODO_ORDER,
    PERIODO_LABEL,
    SP_LAT_MIN,
    SP_LAT_MAX,
    SP_LON_MIN,
    SP_LON_MAX,
)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

_BLUE = "#2b6cb0"
_ORANGE = "#dd6b20"


def _save(fig, nome: str) -> str:
    caminho = FIGURES_DIR / nome
    fig.savefig(caminho)
    plt.close(fig)
    return str(caminho)


def fig_crime_counts(df: pd.DataFrame) -> str:
    """Contagem de TODAS as classes de NATUREZA_APURADA (barra horizontal)."""
    counts = df["crime"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(counts.index[::-1], counts.values[::-1], color=_BLUE)
    ax.set_title("Ocorrências por NATUREZA_APURADA (dataset completo)")
    ax.set_xlabel("Nº de ocorrências")
    ax.ticklabel_format(axis="x", style="plain")
    return _save(fig, "eda_contagem_por_crime.png")


def fig_top15(df: pd.DataFrame) -> str:
    """Top-15 tipos de crime mais frequentes."""
    counts = df["crime"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=_BLUE)
    ax.set_title("Top 15 tipos de crime")
    ax.set_xlabel("Nº de ocorrências")
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v, i, f" {v:,}".replace(",", "."), va="center", fontsize=8)
    return _save(fig, "eda_top15_crimes.png")


def fig_class_imbalance(df: pd.DataFrame) -> str:
    """Desbalanceamento de classes em escala logarítmica."""
    counts = df["crime"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(counts.index[::-1], counts.values[::-1], color=_ORANGE)
    ax.set_xscale("log")
    ax.set_title("Desbalanceamento de classes (escala log)\n"
                 f"classe mais comum / mais rara ≈ {counts.max() / counts.min():,.0f}x"
                 .replace(",", "."))
    ax.set_xlabel("Nº de ocorrências (log)")
    return _save(fig, "eda_desbalanceamento_classes.png")


def fig_hora_dist(df: pd.DataFrame) -> str:
    """
    Distribuição de ocorrências por hora do dia (0-23).
    IMPORTANTE: receba SÓ linhas com HORA EXATA. A hora aproximada por período
    (pontos médios 3/9/15/21) criaria picos artificiais nessas 4 horas.
    """
    counts = df["hora"].value_counts().sort_index()
    counts = counts.reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index, counts.values, color=_BLUE)
    ax.set_title("Distribuição de ocorrências por hora do dia")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Nº de ocorrências")
    ax.set_xticks(range(0, 24))
    return _save(fig, "eda_distribuicao_por_hora.png")


def fig_periodo_dist(df: pd.DataFrame) -> str:
    """Distribuição por período do dia."""
    counts = df["periodo"].value_counts().reindex(PERIODO_ORDER, fill_value=0)
    labels = [PERIODO_LABEL[p] for p in counts.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, counts.values, color=_BLUE)
    ax.set_title("Distribuição de ocorrências por período do dia")
    ax.set_ylabel("Nº de ocorrências")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
    return _save(fig, "eda_distribuicao_por_periodo.png")


def fig_geo_scatter(df: pd.DataFrame, sample: int = 40000) -> str:
    """Dispersão geográfica (amostra) — limite de renderização do scatter."""
    n = min(sample, len(df))
    s = df.sample(n, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(s["lon"], s["lat"], s=2, alpha=0.15, color=_BLUE, linewidths=0)
    ax.set_title(f"Distribuição geográfica das ocorrências\n(amostra de {n:,} pontos)"
                 .replace(",", "."))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(SP_LON_MIN, SP_LON_MAX)
    ax.set_ylim(SP_LAT_MIN, SP_LAT_MAX)
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, "eda_geo_dispersao.png")


def fig_geo_hexbin(df: pd.DataFrame) -> str:
    """Densidade geográfica (hexbin sobre o dataset completo)."""
    fig, ax = plt.subplots(figsize=(8, 7.5))
    hb = ax.hexbin(df["lon"], df["lat"], gridsize=80, cmap="inferno",
                   bins="log", mincnt=1,
                   extent=(SP_LON_MIN, SP_LON_MAX, SP_LAT_MIN, SP_LAT_MAX))
    ax.set_title(f"Densidade de ocorrências (hexbin, n={len(df):,})".replace(",", "."))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(hb, ax=ax, label="log(contagem)")
    return _save(fig, "eda_densidade_hexbin.png")


def fig_hora_crime_heatmap(df: pd.DataFrame, top_n: int = 12) -> str:
    """
    Heatmap hora x crime, normalizado por linha (cada crime soma 1 ao longo do
    dia). Mostra a ASSINATURA TEMPORAL de cada tipo de crime — base da narrativa
    de que o tempo carrega informação que a localização sozinha não dá.
    IMPORTANTE: receba SÓ linhas com HORA EXATA (ver fig_hora_dist).
    """
    top = df["crime"].value_counts().head(top_n).index
    sub = df[df["crime"].isin(top)]
    pivot = (
        sub.pivot_table(index="crime", columns="hora", values="lat",
                        aggfunc="count", fill_value=0)
        .reindex(columns=range(24), fill_value=0)
        .reindex(index=top)
    )
    # normaliza por linha (distribuição horária relativa de cada crime)
    pivot_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot_norm.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(pivot_norm.index)))
    ax.set_yticklabels(pivot_norm.index, fontsize=8)
    ax.set_title("Assinatura temporal por tipo de crime\n"
                 "(fração das ocorrências de cada crime por hora — normalizado por linha)")
    ax.set_xlabel("Hora do dia")
    fig.colorbar(im, ax=ax, label="fração das ocorrências do crime")
    return _save(fig, "eda_heatmap_hora_x_crime.png")


def generate_all(df_clean: pd.DataFrame | None = None) -> dict:
    """
    Gera todas as figuras de EDA, escolhendo o frame correto para cada uma:
      - contagem/top15/desbalanceamento: dataset COMPLETO (todas as linhas).
      - hora e heatmap hora×crime: SÓ HORA EXATA (evita picos artificiais da
        aproximação por período).
      - período: frame exploratório (a aproximação preserva o período, então a
        contagem por período fica completa e correta).
      - dispersão/densidade geográfica: frame exploratório (mais pontos; a
        geo não é afetada pela aproximação da hora).
    """
    from data_loader import load_clean, get_exploratory_df, get_supervised_df
    if df_clean is None:
        df_clean = load_clean()
    exp_df = get_exploratory_df(df_clean)
    exact_df = get_supervised_df(df_clean, min_classe=0)  # geo + hora exata, todas as classes

    paths = {}
    paths["contagem"] = fig_crime_counts(df_clean)
    paths["top15"] = fig_top15(df_clean)
    paths["desbalanceamento"] = fig_class_imbalance(df_clean)
    paths["hora"] = fig_hora_dist(exact_df)
    paths["periodo"] = fig_periodo_dist(exp_df)
    paths["geo_scatter"] = fig_geo_scatter(exp_df)
    paths["hexbin"] = fig_geo_hexbin(exp_df)
    paths["heatmap_hora_crime"] = fig_hora_crime_heatmap(exact_df)
    return paths


if __name__ == "__main__":
    out = generate_all()
    for k, v in out.items():
        print(f"{k:20} -> {v}")
