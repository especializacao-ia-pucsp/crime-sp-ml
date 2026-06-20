#!/usr/bin/python3
#
# maps.py: Mapas interativos (folium) — o coração narrativo do projeto.
#
#   Trilha SEM tempo:
#     - mapa_crimes_sem_tempo.html .......... pontos só por lat/long
#     - mapa_crimes_heatmap_sem_tempo.html .. densidade (dado completo, binado)
#     - mapa_crimes_paulista_foco.html ...... foco no entorno da Av. Paulista
#     - mapa_crimes_sem_paulista.html ....... mesma densidade EXCLUINDO a Paulista
#   Trilha COM tempo:
#     - mapa_crimes_por_periodo.html ........ HeatMapWithTime (4 períodos)
#     - mapa_crimes_por_hora.html ........... HeatMapWithTime (24 horas, hora exata)
#     - mapa_crimes_por_tipo.html ........... 1 camada por tipo de crime
#
#   Convenção de volume:
#     - CAMADAS DE PONTOS usam amostra (MAP_POINT_SAMPLE) — teto de renderização.
#     - HEATMAPS usam o dado COMPLETO agregado numa grade (~111m por célula).

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, HeatMapWithTime

from config import (
    MAPS_DIR,
    MAP_POINT_SAMPLE,
    RANDOM_STATE,
    SP_CENTER,
    PAULISTA_CENTER,
    PAULISTA_LAT_MIN,
    PAULISTA_LAT_MAX,
    PAULISTA_LON_MIN,
    PAULISTA_LON_MAX,
    PERIODO_ORDER,
    PERIODO_LABEL,
)

# Paleta para tipos de crime (cores distinguíveis)
_CRIME_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
]


def _base_map(center=SP_CENTER, zoom=11, tiles="CartoDB positron") -> folium.Map:
    return folium.Map(location=list(center), zoom_start=zoom, tiles=tiles,
                      control_scale=True)


def _in_paulista(df: pd.DataFrame) -> pd.Series:
    return (
        df["lat"].between(PAULISTA_LAT_MIN, PAULISTA_LAT_MAX)
        & df["lon"].between(PAULISTA_LON_MIN, PAULISTA_LON_MAX)
    )


def _grid_weighted(df: pd.DataFrame, step: float = 0.0025,
                   min_count: int = 1, top: int | None = None) -> list[list[float]]:
    """
    Agrega ocorrências numa grade regular e retorna [lat, lon, peso_norm] usando
    o DADO COMPLETO. `step` é o lado da célula em graus (0.0025° ≈ 280m).

    Para conter o tamanho do HTML (sobretudo nos mapas animados com 24 quadros),
    descartamos a cauda de células com pouquíssimas ocorrências (`min_count`) e,
    opcionalmente, mantemos só as `top` células mais quentes. Isso preserva as
    manchas relevantes e remove ruído de 1 ocorrência.
    """
    if len(df) == 0:
        return []
    rlat = (df["lat"] / step).round() * step
    rlon = (df["lon"] / step).round() * step
    g = (
        pd.DataFrame({"rlat": rlat.round(5), "rlon": rlon.round(5)})
        .groupby(["rlat", "rlon"]).size().reset_index(name="n")
    )
    if min_count > 1:
        g = g[g["n"] >= min_count]
    if top is not None and len(g) > top:
        g = g.nlargest(top, "n")
    if len(g) == 0:
        return []
    peso = g["n"] / g["n"].max()
    return np.column_stack([g["rlat"], g["rlon"], peso]).tolist()


def _title(m: folium.Map, html: str) -> None:
    """Adiciona uma caixa de título/legenda sobre o mapa."""
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
        f'background:rgba(255,255,255,0.9);padding:8px 12px;border-radius:6px;'
        f'font-family:sans-serif;font-size:13px;max-width:520px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.3)">{html}</div>'
    ))


# ---------------------------------------------------------------------------
# Trilha SEM tempo
# ---------------------------------------------------------------------------
def mapa_sem_tempo(df: pd.DataFrame) -> str:
    """Pontos por lat/long, SEM tempo — ilustra a concentração visual."""
    n = min(MAP_POINT_SAMPLE, len(df))
    s = df.sample(n, random_state=RANDOM_STATE)
    m = _base_map()
    fg = folium.FeatureGroup(name=f"Ocorrências (amostra {n:,})".replace(",", "."))
    for lat, lon in zip(s["lat"], s["lon"]):
        folium.CircleMarker([lat, lon], radius=1.5, color="#2b6cb0",
                            fill=True, fill_opacity=0.4, weight=0).add_to(fg)
    fg.add_to(m)
    _title(m, "<b>Mapa sem critério de tempo</b><br>"
              f"Amostra de {n:,} ocorrências plotadas só por latitude/longitude. "
              "Áreas de alta circulação (ex.: região da Av. Paulista/Centro) tendem "
              "a concentrar pontos. Sozinho, este mapa não basta para decisão."
              .replace(",", "."))
    out = MAPS_DIR / "mapa_crimes_sem_tempo.html"
    m.save(str(out))
    return str(out)


def mapa_heatmap_sem_tempo(df: pd.DataFrame) -> str:
    """Densidade (dado completo, binado), SEM tempo."""
    m = _base_map()
    HeatMap(_grid_weighted(df, min_count=2), radius=9, blur=12, min_opacity=0.3,
            name="Densidade (dado completo)").add_to(m)
    _title(m, "<b>Densidade sem tempo (dado completo)</b><br>"
              f"{len(df):,} ocorrências agregadas em grade (~111m). A mancha mais "
              "quente costuma cair sobre o Centro/Av. Paulista — efeito de alta "
              "circulação, não necessariamente de risco por morador."
              .replace(",", "."))
    out = MAPS_DIR / "mapa_crimes_heatmap_sem_tempo.html"
    m.save(str(out))
    return str(out)


def mapa_paulista_foco(df: pd.DataFrame) -> str:
    """Foco no entorno aproximado da Av. Paulista."""
    sub = df[_in_paulista(df)]
    m = _base_map(center=PAULISTA_CENTER, zoom=15)
    if len(sub):
        HeatMap(_grid_weighted(sub, step=0.0006), radius=12, blur=15,
                min_opacity=0.3).add_to(m)
    folium.Rectangle(
        bounds=[[PAULISTA_LAT_MIN, PAULISTA_LON_MIN],
                [PAULISTA_LAT_MAX, PAULISTA_LON_MAX]],
        color="#dd6b20", fill=False, weight=2).add_to(m)
    _title(m, "<b>Foco: entorno da Av. Paulista</b><br>"
              f"{len(sub):,} ocorrências no retângulo aproximado. É uma vizinhança "
              "aproximada, não a avenida exata. Não force conclusão causal."
              .replace(",", "."))
    out = MAPS_DIR / "mapa_crimes_paulista_foco.html"
    m.save(str(out))
    return str(out)


def mapa_sem_paulista(df: pd.DataFrame) -> str:
    """Densidade EXCLUINDO a Paulista — comparação honesta."""
    sub = df[~_in_paulista(df)]
    m = _base_map()
    HeatMap(_grid_weighted(sub, min_count=2), radius=9, blur=12,
            min_opacity=0.3).add_to(m)
    _title(m, "<b>Densidade EXCLUINDO o entorno da Paulista</b><br>"
              f"{len(sub):,} ocorrências. Removendo a Paulista, outras manchas "
              "(corredores comerciais, terminais, vias arteriais) seguem visíveis: "
              "a concentração não se resume a uma única região."
              .replace(",", "."))
    out = MAPS_DIR / "mapa_crimes_sem_paulista.html"
    m.save(str(out))
    return str(out)


# ---------------------------------------------------------------------------
# Trilha COM tempo
# ---------------------------------------------------------------------------
def mapa_por_periodo(df: pd.DataFrame) -> str:
    """HeatMapWithTime com 4 quadros (madrugada/manhã/tarde/noite)."""
    frames, labels = [], []
    for p in PERIODO_ORDER:
        sub = df[df["periodo"] == p]
        frames.append(_grid_weighted(sub, step=0.004, min_count=2, top=2500))
        labels.append(PERIODO_LABEL[p])
    m = _base_map()
    HeatMapWithTime(frames, index=labels, radius=10, min_opacity=0.2,
                    max_opacity=0.85, auto_play=False).add_to(m)
    _title(m, "<b>Densidade por período do dia</b><br>"
              "Use o controle de tempo (rodapé) para alternar madrugada→noite. "
              "Há indícios de que as manchas se deslocam conforme o período — "
              "linguagem de hipótese, sem afirmar causalidade.")
    out = MAPS_DIR / "mapa_crimes_por_periodo.html"
    m.save(str(out))
    return str(out)


def mapa_por_hora(df_exact: pd.DataFrame) -> str:
    """HeatMapWithTime com 24 quadros por hora (HORA EXATA)."""
    frames, labels = [], []
    for h in range(24):
        sub = df_exact[df_exact["hora"] == h]
        frames.append(_grid_weighted(sub, step=0.004, min_count=2, top=2200))
        labels.append(f"{h:02d}h")
    m = _base_map()
    HeatMapWithTime(frames, index=labels, radius=10, min_opacity=0.2,
                    max_opacity=0.85, auto_play=False).add_to(m)
    _title(m, "<b>Densidade por hora do dia (hora exata)</b><br>"
              "Sequência de 0h a 23h. Sugere deslocamento das concentrações ao "
              "longo do dia — interpretar como hipótese, não conclusão.")
    out = MAPS_DIR / "mapa_crimes_por_hora.html"
    m.save(str(out))
    return str(out)


def mapa_por_tipo(df: pd.DataFrame, top_n: int = 6) -> str:
    """Uma camada de densidade por tipo de crime (LayerControl)."""
    top = df["crime"].value_counts().head(top_n).index
    m = _base_map()
    for i, crime in enumerate(top):
        sub = df[df["crime"] == crime]
        fg = folium.FeatureGroup(name=crime, show=(i == 0))
        HeatMap(_grid_weighted(sub, step=0.003, min_count=2, top=3500),
                radius=9, blur=12, min_opacity=0.3).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _title(m, "<b>Densidade por tipo de crime</b><br>"
              "Ative/desative as camadas (canto superior direito) para comparar a "
              "geografia de cada tipo. Cada crime tem sua própria distribuição.")
    out = MAPS_DIR / "mapa_crimes_por_tipo.html"
    m.save(str(out))
    return str(out)


def generate_all(df_clean: pd.DataFrame | None = None) -> dict:
    from data_loader import (load_clean, get_geo_df, get_exploratory_df,
                             get_supervised_df)
    if df_clean is None:
        df_clean = load_clean()
    geo = get_geo_df(df_clean)                            # só geo válida (sem exigir tempo)
    exp = get_exploratory_df(df_clean)                    # geo + período/hora
    exact = get_supervised_df(df_clean, min_classe=0)     # geo + hora exata

    paths = {}
    # Mapas SEM tempo / sem dependência temporal -> frame geo-only (mais pontos).
    paths["sem_tempo"] = mapa_sem_tempo(geo)
    paths["heatmap_sem_tempo"] = mapa_heatmap_sem_tempo(geo)
    paths["paulista_foco"] = mapa_paulista_foco(geo)
    paths["sem_paulista"] = mapa_sem_paulista(geo)
    paths["por_tipo"] = mapa_por_tipo(geo)
    # Mapas COM tempo -> frames que carregam o sinal temporal.
    paths["por_periodo"] = mapa_por_periodo(exp)
    paths["por_hora"] = mapa_por_hora(exact)
    return paths


if __name__ == "__main__":
    out = generate_all()
    for k, v in out.items():
        print(f"{k:18} -> {v}")
