#!/usr/bin/python3
#
# features.py: Montagem de matrizes de features para análise não supervisionada
#              (clustering) e exploratória (one-hot de NATUREZA_APURADA).
#
#   O modelo supervisionado (modeling.py) usa um Pipeline sklearn com seu
#   próprio scaler sobre as colunas cruas [lat, lon, hora, hora_sin, hora_cos];
#   por isso este módulo foca no que o clustering e a EDA precisam:
#     - matriz de clustering escalonada (espaço + tempo cíclico);
#     - one-hot de NATUREZA_APURADA para análises exploratórias.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Features espaço-temporais usadas no clustering. Usamos a codificação CÍCLICA
# da hora (sin/cos) em vez da hora linear: assim 23h e 0h ficam próximos, o que
# a hora linear (0..23) não captura.
CLUSTER_FEATURES = ["lat", "lon", "hora_sin", "hora_cos"]


def scale_columns(df: pd.DataFrame, cols: list[str], kind: str = "standard",
                  scaler=None):
    """
    Escalona as colunas indicadas. Retorna (array_escalonado, scaler_ajustado).

    kind: "standard" (StandardScaler) ou "minmax" (MinMaxScaler).
    Passe um `scaler` já ajustado para apenas transformar (ex.: aplicar o mesmo
    scaler de treino em novos dados).
    """
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas ausentes para escalonamento: {faltando}")

    X = df[cols].to_numpy(dtype=float)
    if scaler is None:
        scaler = MinMaxScaler() if kind == "minmax" else StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    return Xs, scaler


def build_cluster_matrix(df: pd.DataFrame, features: list[str] | None = None,
                         kind: str = "standard", scaler=None):
    """
    Constrói a matriz de clustering escalonada.

    Retorna (X_scaled, scaler, feature_names).
    Por padrão usa CLUSTER_FEATURES (lat, lon, hora_sin, hora_cos).
    """
    features = features or CLUSTER_FEATURES
    Xs, scaler = scale_columns(df, features, kind=kind, scaler=scaler)
    return Xs, scaler, list(features)


def one_hot_natureza(df: pd.DataFrame, col: str = "crime",
                     top_n: int | None = None) -> pd.DataFrame:
    """
    One-hot encoding de NATUREZA_APURADA (coluna `crime`) para EDA/clustering.

    Se top_n for informado, mantém apenas as top_n classes mais frequentes e
    agrupa o restante em "OUTROS_AGRUPADOS" (evita explosão de colunas).
    """
    if col not in df.columns:
        raise KeyError(f"Coluna '{col}' ausente para one-hot.")

    serie = df[col].astype("string")
    if top_n is not None:
        principais = serie.value_counts().head(top_n).index
        serie = serie.where(serie.isin(principais), other="OUTROS_AGRUPADOS")

    dummies = pd.get_dummies(serie, prefix="nat")
    return dummies


def add_space_time_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta colunas normalizadas (MinMax) de longitude/latitude/hora para a
    visualização 3D (eixos comparáveis em [0,1]). Não altera as originais.
    """
    out = df.copy()
    for src, dst in [("lon", "lon_norm"), ("lat", "lat_norm")]:
        v = out[src].to_numpy(dtype=float)
        rng = v.max() - v.min()
        out[dst] = (v - v.min()) / rng if rng else 0.0
    if "hora_norm" not in out.columns:
        out["hora_norm"] = out["hora"].astype(float) / 23.0
    return out
