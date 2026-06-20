#!/usr/bin/python3
#
# config.py: Configuração central do projeto crime-sp-ml.
#
#   Centraliza caminhos (via pathlib, sem hardcode de máquina), constantes de
#   reprodutibilidade (RANDOM_STATE), limites de amostragem para renderização,
#   bounding boxes geográficos e os mapeamentos de limpeza usados em todo o
#   projeto. Importar daqui evita repetir "números mágicos" pelos módulos.

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (derivados da localização deste arquivo: <root>/src/config.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Dataset limpo (fonte de verdade do projeto)
PROCESSED_CSV = PROCESSED_DIR / "SPDadosCriminais_Preprocessed.csv.bz2"
# Cache do dataframe já parseado/limpo (acelera execuções repetidas; gitignored)
CLEAN_CACHE = PROCESSED_DIR / "crimes_clean_cache.pkl"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MAPS_DIR = REPORTS_DIR / "maps"
MODELS_DIR = ROOT / "models"

# Garante que os diretórios de saída existam
for _d in (REPORTS_DIR, FIGURES_DIR, MAPS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def rel_to_root(caminho) -> str:
    """
    Converte um caminho para uma string relativa à raiz do projeto, em estilo
    POSIX (barras /). Usado ao gravar caminhos em artefatos (JSONs de métricas)
    para não vazar caminho absoluto de máquina (ex.: C:\\GIT\\...). Se o caminho
    estiver fora da raiz, devolve o caminho como está.
    """
    p = Path(caminho)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()

# ---------------------------------------------------------------------------
# Reprodutibilidade e amostragem
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Limite de pontos para CAMADAS DE PONTOS em mapas folium e no 3D Plotly.
# Não é amostragem estatística: é um teto de RENDERIZAÇÃO (o navegador não
# desenha 1M+ marcadores). Heatmaps usam o dado completo. Documentado no README.
MAP_POINT_SAMPLE = 6000
VIZ3D_SAMPLE = 12000

# Amostra para clustering baseado em distância (DBSCAN/HDBSCAN), cujo custo de
# memória/tempo é proibitivo em 1,2M pontos. KMeans roda no completo
# (MiniBatchKMeans). Ver clustering.py para a justificativa.
CLUSTER_DENSITY_SAMPLE = 60000

# ---------------------------------------------------------------------------
# Bounding boxes geográficos
# ---------------------------------------------------------------------------
# Município de São Paulo (filtro de coordenadas válidas)
SP_LAT_MIN, SP_LAT_MAX = -24.10, -23.30
SP_LON_MIN, SP_LON_MAX = -46.90, -46.30

# Centro aproximado da cidade (para inicializar mapas folium)
SP_CENTER = (-23.5505, -46.6333)

# Retângulo aproximado da Av. Paulista + entorno (usado para foco/exclusão na
# narrativa do projeto). NÃO é a avenida exata: é uma vizinhança aproximada.
PAULISTA_LAT_MIN, PAULISTA_LAT_MAX = -23.575, -23.553
PAULISTA_LON_MIN, PAULISTA_LON_MAX = -46.668, -46.640
PAULISTA_CENTER = (-23.5613, -46.6558)

# ---------------------------------------------------------------------------
# Mapeamentos de limpeza / normalização
# ---------------------------------------------------------------------------
# DESCR_PERIODO vem em caixa mista e com mojibake (ex.: "Pela manh�").
# Normalizamos por PREFIXO (uppercase, sem acento) para um rótulo canônico.
# "indeterminado" é descartado no modelo supervisionado.
PERIODO_CANONICO = {
    "PELA MANH": "manha",
    "A TARDE": "tarde",
    "A NOITE": "noite",
    "DE MADRUGADA": "madrugada",
    "EM HORA INCERTA": "indeterminado",
}

# Ordem lógica dos períodos para gráficos
PERIODO_ORDER = ["madrugada", "manha", "tarde", "noite"]

# Hora aproximada (ponto médio) de cada período — usado APENAS na trilha
# exploratória/clustering quando não há hora exata. Deixar claro que é
# aproximação e nunca usar isto no modelo supervisionado.
PERIODO_HORA_APROX = {
    "madrugada": 3,   # ~00h-06h
    "manha": 9,       # ~06h-12h
    "tarde": 15,      # ~12h-18h
    "noite": 21,      # ~18h-24h
}

# Faixas de hora -> período (para derivar período a partir da hora exata)
def hora_para_periodo(hora: int) -> str:
    """Mapeia uma hora inteira [0,23] para o período do dia canônico."""
    if 0 <= hora < 6:
        return "madrugada"
    if 6 <= hora < 12:
        return "manha"
    if 12 <= hora < 18:
        return "tarde"
    return "noite"

# Rótulos legíveis dos períodos (para textos/slides)
PERIODO_LABEL = {
    "madrugada": "Madrugada (00h–06h)",
    "manha": "Manhã (06h–12h)",
    "tarde": "Tarde (12h–18h)",
    "noite": "Noite (18h–24h)",
}
