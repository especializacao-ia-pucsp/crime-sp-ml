#!/usr/bin/python3
#
# viz3d.py: Visualização 3D interativa (Plotly) — espaço + tempo + tipo.
#
#   Eixos: X = longitude normalizada, Y = latitude normalizada,
#          Z = hora normalizada; cor = NATUREZA_APURADA (ou cluster).
#
#   A 3D ajuda a enxergar SIMULTANEAMENTE espaço, tempo e tipo de crime — mas
#   é uma ferramenta de intuição, não substitui métricas.
#
#   Os HTMLs carregam o plotly.js via CDN (arquivos leves; requer internet ao
#   abrir). Para versão offline, troque include_plotlyjs="cdn" por True.

from __future__ import annotations

import pandas as pd
import plotly.express as px

from config import FIGURES_DIR, VIZ3D_SAMPLE, RANDOM_STATE
from features import add_space_time_normalized


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=RANDOM_STATE)


def crimes_3d(df: pd.DataFrame, top_n: int = 8, sample: int = VIZ3D_SAMPLE) -> str:
    """
    Dispersão 3D long/lat/hora colorida por tipo de crime (top_n classes para
    legibilidade; demais agrupadas). Amostra por limite de renderização.
    """
    top = df["crime"].value_counts().head(top_n).index
    d = df.copy()
    d["crime_plot"] = d["crime"].where(d["crime"].isin(top), other="OUTROS")
    d = _sample(d, sample)
    d = add_space_time_normalized(d)

    fig = px.scatter_3d(
        d, x="lon_norm", y="lat_norm", z="hora_norm",
        color="crime_plot", opacity=0.55,
        labels={"lon_norm": "Longitude (norm)", "lat_norm": "Latitude (norm)",
                "hora_norm": "Hora (norm)", "crime_plot": "Crime"},
        title=("Crimes em 3D — espaço (lon/lat) × tempo (hora) × tipo<br>"
               f"<sup>amostra de {len(d):,} ocorrências; cor = tipo de crime</sup>"
               .replace(",", ".")),
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, b=0, t=60))
    out = FIGURES_DIR / "crimes_3d_long_lat_tempo.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return str(out)


def clusters_3d(df: pd.DataFrame, label_col: str = "cluster",
                sample: int = VIZ3D_SAMPLE,
                out_name: str = "clusters_3d.html") -> str:
    """Dispersão 3D long/lat/hora colorida pelo rótulo de cluster."""
    d = _sample(df, sample)
    d = add_space_time_normalized(d)
    d[label_col] = d[label_col].astype(str)

    fig = px.scatter_3d(
        d, x="lon_norm", y="lat_norm", z="hora_norm",
        color=label_col, opacity=0.6,
        labels={"lon_norm": "Longitude (norm)", "lat_norm": "Latitude (norm)",
                "hora_norm": "Hora (norm)", label_col: "Cluster"},
        title=("Clusters em 3D — espaço (lon/lat) × tempo (hora)<br>"
               f"<sup>amostra de {len(d):,} ocorrências; cor = cluster</sup>"
               .replace(",", ".")),
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, b=0, t=60))
    out = FIGURES_DIR / out_name
    fig.write_html(str(out), include_plotlyjs="cdn")
    return str(out)


def generate_all(df_clean: pd.DataFrame | None = None) -> dict:
    from data_loader import load_clean, get_exploratory_df
    if df_clean is None:
        df_clean = load_clean()
    exp = get_exploratory_df(df_clean)
    return {"crimes_3d": crimes_3d(exp)}


if __name__ == "__main__":
    out = generate_all()
    for k, v in out.items():
        print(f"{k:12} -> {v}")
