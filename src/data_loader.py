#!/usr/bin/python3
#
# data_loader.py: Carga e limpeza final do dataset criminal de São Paulo.
#
#   O pipeline em `pipeline.py` já produziu o CSV processado
#   (data/processed/SPDadosCriminais_Preprocessed.csv.bz2). Este módulo é a
#   ÚNICA porta de entrada para os notebooks e demais módulos: ele lê esse CSV
#   e aplica os ajustes finais que a análise e a modelagem exigem:
#
#     - remove a coluna de índice serializada ("Unnamed: 0");
#     - converte LATITUDE/LONGITUDE para float e marca como válidas apenas as
#       coordenadas dentro do bounding box do município de São Paulo
#       (descarta os ~195k registros com lat/long = 0 ou fora da cidade);
#     - converte HORA_OCORRENCIA_BO em hora inteira [0,23] (NaN quando ausente);
#     - normaliza DESCR_PERIODO (caixa mista + mojibake) para um rótulo canônico
#       {madrugada, manha, tarde, noite, indeterminado};
#     - corrige o mojibake que duplica uma classe de NATUREZA_APURADA;
#     - deriva features de tempo (sin/cos cíclico, período do dia).
#
#   Os resultados são cacheados em pickle para acelerar execuções repetidas.
#
#   Funções públicas principais:
#     - load_clean()         -> DataFrame completo limpo, com flags
#     - get_supervised_df()  -> subset p/ modelo (geo válida + HORA EXATA)
#     - get_exploratory_df() -> subset p/ EDA/clustering (geo válida; período
#                               textual pode virar hora aproximada)

from __future__ import annotations

import unicodedata
import numpy as np
import pandas as pd

from config import (
    PROCESSED_CSV,
    CLEAN_CACHE,
    SP_LAT_MIN,
    SP_LAT_MAX,
    SP_LON_MIN,
    SP_LON_MAX,
    PERIODO_CANONICO,
    PERIODO_HORA_APROX,
    hora_para_periodo,
)

# Colunas originais relevantes do CSV processado
COL_DATA = "DATA_OCORRENCIA_BO"
COL_HORA = "HORA_OCORRENCIA_BO"
COL_PERIODO = "DESCR_PERIODO"
COL_LAT = "LATITUDE"
COL_LON = "LONGITUDE"
COL_CRIME = "NATUREZA_APURADA"
COL_BAIRRO = "BAIRRO"

REQUIRED_COLUMNS = [COL_HORA, COL_PERIODO, COL_LAT, COL_LON, COL_CRIME]


# ---------------------------------------------------------------------------
# Helpers de normalização
# ---------------------------------------------------------------------------
def _strip_accents_upper(text: str) -> str:
    """Uppercase sem diacríticos (mesma ideia de utils.convert_to_plain_uppercase)."""
    decoded = unicodedata.normalize("NFKD", str(text))
    plain = "".join(ch for ch in decoded if not unicodedata.combining(ch))
    return plain.upper().strip()


def _normaliza_periodo(valor) -> float | str:
    """
    Converte um valor bruto de DESCR_PERIODO no rótulo canônico.

    Trata caixa mista e o mojibake "Pela manh�" casando por PREFIXO sobre a
    string em maiúsculas e sem acento. Retorna np.nan quando o valor é nulo
    ou não reconhecido.
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return np.nan
    chave = _strip_accents_upper(valor)
    # O mojibake vira um caractere de substituição; removemos qualquer coisa
    # que não seja A-Z/espaço para casar o prefixo de forma robusta.
    chave = "".join(ch if ("A" <= ch <= "Z" or ch == " ") else " " for ch in chave)
    chave = " ".join(chave.split())
    for prefixo, canonico in PERIODO_CANONICO.items():
        if chave.startswith(prefixo):
            return canonico
    return np.nan


def _limpa_natureza(serie: pd.Series) -> pd.Series:
    """
    Corrige variações de travessão em NATUREZA_APURADA. O caso real é
    "LESAO CORPORAL CULPOSA – OUTRAS" (en-dash U+2013), que deve coincidir com
    "LESAO CORPORAL CULPOSA - OUTRAS" (hífen U+002D). Unificamos todos os
    travessões/replacement-char em hífen e colapsamos espaços.
    """
    limpa = serie.astype("string")
    for dash in ("–", "—", "‒", "−", "�"):  # –, —, ‒, −, �
        limpa = limpa.str.replace(dash, "-", regex=False)
    limpa = limpa.str.replace(r"\s+", " ", regex=True).str.strip()
    return limpa


# ---------------------------------------------------------------------------
# Carga / construção do dataframe limpo
# ---------------------------------------------------------------------------
def _build_clean() -> pd.DataFrame:
    """Lê o CSV processado e aplica toda a limpeza/derivação. (Pesado.)"""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado em {PROCESSED_CSV}. "
            "Gere-o com src/pipeline.py a partir dos .xlsx da SSP/SP."
        )

    df = pd.read_csv(PROCESSED_CSV, dtype=str)

    # Coluna de índice serializada pelo to_csv original
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # Validação defensiva de colunas essenciais
    faltando = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if faltando:
        raise KeyError(
            f"Colunas essenciais ausentes no dataset: {faltando}. "
            f"Colunas presentes: {list(df.columns)}"
        )

    # --- Coordenadas -------------------------------------------------------
    lat = pd.to_numeric(df[COL_LAT].str.replace(",", ".", regex=False), errors="coerce")
    lon = pd.to_numeric(df[COL_LON].str.replace(",", ".", regex=False), errors="coerce")
    df["lat"] = lat
    df["lon"] = lon
    df["geo_valid"] = (
        lat.between(SP_LAT_MIN, SP_LAT_MAX) & lon.between(SP_LON_MIN, SP_LON_MAX)
    )

    # --- Hora exata --------------------------------------------------------
    hora_dt = pd.to_datetime(df[COL_HORA], format="%H:%M:%S", errors="coerce")
    df["hora"] = hora_dt.dt.hour.astype("Int64")
    df["minuto"] = hora_dt.dt.minute.astype("Int64")
    df["tem_hora_exata"] = df["hora"].notna()

    # --- Período textual normalizado --------------------------------------
    df["periodo_texto"] = df[COL_PERIODO].map(_normaliza_periodo)

    # --- Natureza (target) limpa ------------------------------------------
    df["crime"] = _limpa_natureza(df[COL_CRIME])

    # --- Data (apenas parse; usada só para sanidade/EDA leve) -------------
    df["data"] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df["ano"] = df["data"].dt.year.astype("Int64")

    return df


def load_clean(force: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """
    Retorna o DataFrame completo limpo (todas as linhas), com colunas derivadas
    e flags. Usa cache em pickle quando possível.

    Parâmetros:
      force (bool): se True, ignora o cache e reconstrói.
      use_cache (bool): se False, não lê nem escreve cache.
    """
    if use_cache and not force and CLEAN_CACHE.exists():
        try:
            if CLEAN_CACHE.stat().st_mtime >= PROCESSED_CSV.stat().st_mtime:
                return pd.read_pickle(CLEAN_CACHE)
        except Exception:
            pass  # cache corrompido/incompatível -> reconstrói

    df = _build_clean()

    if use_cache:
        try:
            df.to_pickle(CLEAN_CACHE)
        except Exception:
            pass  # cache é otimização; falha não é fatal

    return df


# ---------------------------------------------------------------------------
# Subsets prontos para cada etapa
# ---------------------------------------------------------------------------
def add_cyclical_hour(df: pd.DataFrame, hora_col: str = "hora") -> pd.DataFrame:
    """Adiciona hora_norm, hora_sin e hora_cos a partir de uma coluna de hora."""
    out = df.copy()
    h = out[hora_col].astype(float)
    out["hora_norm"] = h / 23.0
    out["hora_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["hora_cos"] = np.cos(2 * np.pi * h / 24.0)
    return out


def get_supervised_df(df: pd.DataFrame | None = None, min_classe: int = 50) -> pd.DataFrame:
    """
    Subset para o MODELO SUPERVISIONADO: geo válida + HORA EXATA.

    - Descarta período textual e "hora incerta" (sem aproximação -> sem
      vazamento de informação temporal incerta no modelo).
    - Remove classes raríssimas (< min_classe ocorrências) para permitir split
      estratificado e avaliação estável; o descarte é pequeno e documentado.

    Colunas chave de saída: lat, lon, hora, hora_norm, hora_sin, hora_cos,
    periodo (derivado da hora exata) e crime (target).
    """
    if df is None:
        df = load_clean()

    sub = df[df["geo_valid"] & df["tem_hora_exata"]].copy()
    sub["hora"] = sub["hora"].astype(int)
    sub["periodo"] = sub["hora"].map(hora_para_periodo)
    sub = add_cyclical_hour(sub, "hora")

    if min_classe and min_classe > 0:
        contagem = sub["crime"].value_counts()
        manter = contagem[contagem >= min_classe].index
        sub = sub[sub["crime"].isin(manter)].copy()

    cols = ["lat", "lon", "hora", "minuto", "hora_norm", "hora_sin", "hora_cos",
            "periodo", "crime", "ano", "data", COL_BAIRRO]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def get_exploratory_df(df: pd.DataFrame | None = None,
                       usar_periodo_aproximado: bool = True) -> pd.DataFrame:
    """
    Subset para EDA/CLUSTERING: exige apenas geo válida.

    - Onde há hora exata, usa-a.
    - Onde NÃO há hora exata mas há período textual válido (manha/tarde/noite/
      madrugada), opcionalmente preenche a hora pelo PONTO MÉDIO do período
      (PERIODO_HORA_APROX). Isto é uma APROXIMAÇÃO exploratória, marcada em
      `hora_aproximada=True`, e NUNCA deve ser usada no modelo supervisionado.
    - Descarta "indeterminado" e linhas que continuam sem hora.

    Retorna colunas: lat, lon, hora, hora_norm, hora_sin, hora_cos, periodo,
    crime, hora_aproximada.
    """
    if df is None:
        df = load_clean()

    sub = df[df["geo_valid"]].copy()

    hora = sub["hora"].astype("float")
    aproximada = pd.Series(False, index=sub.index)

    if usar_periodo_aproximado:
        sem_hora = hora.isna()
        per = sub["periodo_texto"]
        pode_aprox = sem_hora & per.isin(list(PERIODO_HORA_APROX.keys()))
        hora_aprox = per.map(PERIODO_HORA_APROX)
        hora = hora.where(~pode_aprox, hora_aprox)
        aproximada = pode_aprox

    sub["hora"] = hora
    sub["hora_aproximada"] = aproximada
    sub = sub[sub["hora"].notna()].copy()
    sub["hora"] = sub["hora"].astype(int)
    sub["periodo"] = sub["hora"].map(hora_para_periodo)
    sub = add_cyclical_hour(sub, "hora")

    cols = ["lat", "lon", "hora", "hora_norm", "hora_sin", "hora_cos",
            "periodo", "crime", "hora_aproximada", COL_BAIRRO]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def get_geo_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Subset GEO-ONLY: todas as linhas com coordenada válida, SEM exigir tempo.

    Usado nos mapas "sem critério de tempo" (latitude/longitude apenas), que não
    devem descartar ocorrências geocodificadas só porque lhes falta hora exata
    ou período textual — ao contrário de get_exploratory_df (que exige um sinal
    temporal). Cobre ~1,88M linhas vs ~1,81M do exploratório.
    """
    if df is None:
        df = load_clean()
    sub = df[df["geo_valid"]].copy()
    cols = ["lat", "lon", "crime", COL_BAIRRO]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def resumo_dataset(df: pd.DataFrame | None = None) -> dict:
    """Estatísticas rápidas para logging/README/relatórios."""
    if df is None:
        df = load_clean()
    sup = get_supervised_df(df)
    exp = get_exploratory_df(df)
    return {
        "linhas_totais": int(len(df)),
        "geo_validas": int(df["geo_valid"].sum()),
        "com_hora_exata": int(df["tem_hora_exata"].sum()),
        "linhas_supervisionado": int(len(sup)),
        "classes_supervisionado": int(sup["crime"].nunique()),
        "linhas_exploratorio": int(len(exp)),
        "classes_totais": int(df["crime"].nunique()),
    }


if __name__ == "__main__":
    import json
    print("Carregando e limpando o dataset (pode levar ~1 min na 1ª vez)...")
    info = resumo_dataset(load_clean(force=True))
    print(json.dumps(info, indent=2, ensure_ascii=False))
