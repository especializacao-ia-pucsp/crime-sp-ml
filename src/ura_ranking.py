#!/usr/bin/python3
#
# ura_ranking.py: Camada de aplicação — ranking de crimes prováveis para a URA
#                 do 190 a partir de (latitude, longitude, hora).
#
#   Funções públicas:
#     - predict_top_crimes(lat, lon, horario, top_k=3) -> lista ranqueada
#     - gerar_opcoes_ura(lat, lon, horario) -> texto da URA (no máx. 2 crimes +
#       "falar com atendente")
#
#   IMPORTANTE (ética/escopo): isto é um PROTÓTIPO ACADÊMICO de ORDENAÇÃO de
#   opções de menu, não decisão policial automática e não substitui o atendente
#   humano. Ver reports/proposta_atualizada.md (seção de ética).

from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib

from config import MODELS_DIR, REPORTS_DIR

_MODEL = None
_LE = None

# Casos de referência (local + hora) usados nos exemplos e na documentação.
CASOS_REFERENCIA = [
    ("Av. Paulista — 19h (noite, comercial)", -23.5613, -46.6558, 19),
    ("Centro/Sé — 3h (madrugada)", -23.5505, -46.6333, 3),
    ("Zona Leste — 8h (manhã)", -23.5487, -46.4583, 8),
    ("Zona Sul — 21h (noite, periférica)", -23.6680, -46.7082, 21),
]


def _parse_horario(horario) -> int:
    """Aceita int (0-23), 'HH:MM', 'HH:MM:SS' ou 'HH' e retorna a hora inteira."""
    if isinstance(horario, (int, np.integer)):
        h = int(horario)
    elif isinstance(horario, float) and not np.isnan(horario):
        h = int(horario)
    else:
        s = str(horario).strip()
        h = int(s.split(":")[0]) if ":" in s else int(float(s))
    if not 0 <= h <= 23:
        raise ValueError(f"Hora fora do intervalo [0,23]: {horario!r}")
    return h


def _features(lat: float, lon: float, hora: int) -> np.ndarray:
    """Monta o vetor de features na MESMA ordem de modeling.FEATURES."""
    hora_sin = np.sin(2 * np.pi * hora / 24.0)
    hora_cos = np.cos(2 * np.pi * hora / 24.0)
    return np.array([[lat, lon, hora_sin, hora_cos]], dtype="float32")


def load_model(model_dir: Path = MODELS_DIR):
    """Carrega (uma vez) o classificador e o label encoder persistidos."""
    global _MODEL, _LE
    if _MODEL is None or _LE is None:
        model_path = model_dir / "crime_classifier.joblib"
        le_path = model_dir / "label_encoder.joblib"
        if not model_path.exists() or not le_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_dir}. Rode src/modeling.py "
                "(ou run_all.py) para treiná-lo primeiro."
            )
        _MODEL = joblib.load(model_path)
        _LE = joblib.load(le_path)
    return _MODEL, _LE


def predict_top_crimes(latitude: float, longitude: float, horario,
                       top_k: int = 3) -> list[dict]:
    """
    Retorna os `top_k` crimes mais prováveis para (latitude, longitude, hora).

    Exemplo de retorno:
      [{"rank": 1, "crime": "ROUBO - OUTROS", "probabilidade": 0.42}, ...]
    """
    model, le = load_model()
    hora = _parse_horario(horario)
    X = _features(latitude, longitude, hora)
    proba = model.predict_proba(X)[0]
    ordem = np.argsort(proba)[::-1][:top_k]
    return [
        {"rank": i + 1,
         "crime": str(le.inverse_transform([idx])[0]),
         "probabilidade": round(float(proba[idx]), 4)}
        for i, idx in enumerate(ordem)
    ]


def gerar_opcoes_ura(latitude: float, longitude: float, horario,
                     n_crimes: int = 2) -> str:
    """
    Gera o texto de menu da URA do 190 com NO MÁXIMO `n_crimes` (padrão 2)
    opções de crime mais provável + a opção de falar com atendente.

    Mantém o menu curto de propósito (não despeja uma lista gigante) e SEMPRE
    oferece o atendente humano. É priorização/ordenação, não decisão.
    """
    top = predict_top_crimes(latitude, longitude, horario, top_k=n_crimes)
    linhas = ["Polícia Militar de São Paulo, 190.",
              "Para agilizar seu atendimento, selecione o tipo de ocorrência:"]
    for i, item in enumerate(top, start=1):
        linhas.append(f"{i}. {_humaniza(item['crime'])}.")
    linhas.append(f"{len(top) + 1}. Falar com atendente.")
    return " ".join(linhas)


def _humaniza(crime: str) -> str:
    """Deixa o rótulo do crime mais natural para fala da URA."""
    mapa = {
        "ROUBO - OUTROS": "Roubo",
        "FURTO - OUTROS": "Furto",
        "FURTO DE VEICULO": "Furto de veículo",
        "ROUBO DE VEICULO": "Roubo de veículo",
        "LESAO CORPORAL DOLOSA": "Agressão / lesão corporal",
        "LESAO CORPORAL CULPOSA POR ACIDENTE DE TRANSITO": "Acidente de trânsito com vítima",
        "TRAFICO DE ENTORPECENTES": "Tráfico de drogas",
        "ROUBO DE CARGA": "Roubo de carga",
    }
    return mapa.get(crime, crime.capitalize())


def salvar_exemplos_md(caminho=None) -> str:
    """Gera reports/ura_examples.md com o ranking e o texto da URA para os casos
    de referência (reproduzível: usa o modelo persistido)."""
    caminho = caminho or (REPORTS_DIR / "ura_examples.md")
    linhas = [
        "# Exemplos de URA do 190",
        "",
        "A ideia da URA é simples: a partir de **(latitude, longitude, hora)** da "
        "ligação, o modelo estima quais tipos de ocorrência são mais prováveis "
        "naquele ponto e horário e usa isso para **ordenar as opções do menu**. "
        "O menu mostra no máximo **2 tipos** e termina **sempre** com a opção de "
        "*falar com atendente* — quem liga nunca fica preso ao menu.",
        "",
        "Importante: isto ordena opções, não classifica a ocorrência nem decide "
        "o despacho. As probabilidades abaixo aparecem só para inspeção; o cidadão "
        "vê apenas o texto do menu.",
        "",
    ]
    for nome, lat, lon, h in CASOS_REFERENCIA:
        linhas.append(f"## {nome}")
        linhas.append(f"`lat={lat}, lon={lon}, hora={h}`")
        linhas.append("")
        linhas.append("Ranking (top-3):")
        linhas.append("")
        linhas.append("| rank | crime | probabilidade |")
        linhas.append("|---|---|---|")
        for r in predict_top_crimes(lat, lon, h, top_k=3):
            linhas.append(f"| {r['rank']} | {r['crime']} | {r['probabilidade']:.1%} |")
        linhas.append("")
        linhas.append("**Texto da URA:**")
        linhas.append("")
        linhas.append(f"> {gerar_opcoes_ura(lat, lon, h)}")
        linhas.append("")
    linhas.append("---")
    linhas.append("O ponto a observar é a mudança de ordem conforme o contexto: "
                  "de manhã na Zona Leste o *furto de veículo* sobe; à noite na "
                  "região central o *roubo* aparece em primeiro. É essa reordenação "
                  "por local e horário que justifica o protótipo — não acertar o "
                  "crime exato, e muito menos prever ocorrências raras.")
    texto = "\n".join(linhas) + "\n"
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    return str(caminho)


if __name__ == "__main__":
    for nome, lat, lon, h in CASOS_REFERENCIA:
        print(f"\n### {nome} (lat={lat}, lon={lon}, hora={h})")
        for r in predict_top_crimes(lat, lon, h, top_k=3):
            print(f"   {r['rank']}. {r['crime']} -> {r['probabilidade']:.1%}")
        print("   URA:", gerar_opcoes_ura(lat, lon, h))
    print("\nMD salvo em:", salvar_exemplos_md())
