#!/usr/bin/python3
#
# modeling.py: Modelo supervisionado para a URA do 190.
#
#   Tarefa: dado (latitude, longitude, hora) -> distribuição de probabilidade
#   sobre os tipos de crime (NATUREZA_APURADA). A saída ordenada alimenta a URA.
#
#   ENTRADA do modelo: lat, lon, hora_sin, hora_cos. NUNCA usamos o próprio
#   crime como feature (seria vazamento). A hora é codificada de forma cíclica
#   (sin/cos) para que 23h e 0h fiquem próximas.
#
#   Modelos comparados (os principais usam priores naturais, SEM class_weight,
#   porque a URA quer a base-rate real do local/horário):
#     - DummyClassifier (baselines: classe mais frequente e priores)
#     - LogisticRegression (baseline linear)
#     - RandomForestClassifier (não linear, compacto)
#     - HistGradientBoostingClassifier (boosting)
#     - HistGradientBoostingClassifier com class_weight="balanced" APENAS para
#       mostrar o trade-off do desbalanceamento (melhora balanced acc/macro-F1,
#       mas piora top-1) — não é candidato a melhor modelo.
#
#   Seleção do melhor modelo por top-1 (accuracy), não por top-3 (dominado pela
#   base-rate). Ver a docstring de train_all.
#
#   Métricas: accuracy, balanced accuracy, macro/weighted F1, top-2 e top-3,
#   classification report e matriz de confusão (normalizada).
#
#   Persistência: models/crime_classifier.joblib (melhor pipeline),
#   models/label_encoder.joblib, models/preprocessing_pipeline.joblib (scaler).

from __future__ import annotations

import json
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_STATE, rel_to_root

# Features de ENTRADA do modelo (target nunca entra aqui)
FEATURES = ["lat", "lon", "hora_sin", "hora_cos"]


# ---------------------------------------------------------------------------
# Preparação X/y
# ---------------------------------------------------------------------------
def build_xy(df: pd.DataFrame):
    """Retorna (X, y_encoded, label_encoder). X só com FEATURES."""
    faltando = [c for c in FEATURES + ["crime"] if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas ausentes para modelagem: {faltando}")
    X = df[FEATURES].astype("float32").to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df["crime"].astype(str))
    return X, y, le


def _make_pipeline(clf) -> Pipeline:
    """Pipeline padrão: StandardScaler + classificador."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _model_zoo() -> dict:
    """
    Modelos a comparar. Para o caso da URA (ranquear os crimes MAIS prováveis)
    usamos PRIORES NATURAIS — sem class_weight — porque queremos a base-rate
    real do local/horário. Mantemos UMA variante `balanced` apenas para
    DOCUMENTAR o trade-off do desbalanceamento (melhora macro-F1/balanced acc,
    mas piora top-1 e a ordenação para a URA).
    """
    return {
        # Baselines
        "dummy_most_frequent": _make_pipeline(
            DummyClassifier(strategy="most_frequent")),
        "dummy_prior": _make_pipeline(  # base-rate honesto (proba = priores)
            DummyClassifier(strategy="prior")),
        # Modelos "naturais" (sem balanceamento) — candidatos para a URA
        "logistic_regression": _make_pipeline(
            LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=-1)),
        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=16, min_samples_leaf=200,
                max_leaf_nodes=2500, max_samples=0.5, n_jobs=-1,
                random_state=RANDOM_STATE)),
        "hist_gradient_boosting": _make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE)),
        # Variante balanceada (apenas para mostrar o impacto do desbalanceamento)
        "hist_gradient_boosting_balanced": _make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, class_weight="balanced",
                random_state=RANDOM_STATE)),
    }


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def _evaluate(name, pipe, X_test, y_test, classes) -> dict:
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        labels_idx = np.arange(len(classes))
        metrics["top2_accuracy"] = float(
            top_k_accuracy_score(y_test, proba, k=2, labels=labels_idx))
        metrics["top3_accuracy"] = float(
            top_k_accuracy_score(y_test, proba, k=3, labels=labels_idx))
    else:
        metrics["top2_accuracy"] = None
        metrics["top3_accuracy"] = None
    return metrics


def _confusion_png(y_test, y_pred, classes, nome="confusion_matrix.png") -> str:
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [c[:24] for c in classes]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de confusão (normalizada por linha) — melhor modelo")
    fig.colorbar(im, ax=ax, label="fração da classe verdadeira")
    fig.savefig(FIGURES_DIR / nome, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return str(FIGURES_DIR / nome)


# ---------------------------------------------------------------------------
# Treino completo + seleção + persistência
# ---------------------------------------------------------------------------
def train_all(df: pd.DataFrame, test_size: float = 0.25,
              selecionar_por: str = "accuracy") -> dict:
    """
    Treina e compara os modelos, escolhe um e persiste.

    A seleção usa `accuracy` (top-1) e NÃO top-3. Motivo: o top-3 é dominado pela
    base-rate (as 3 classes mais comuns já cobrem ~81% das ocorrências), então
    quase todos os modelos empatam nessa métrica com o baseline de prioris. O
    top-1 é onde o uso de local+hora gera ganho mensurável sobre o baseline, e é
    a métrica mais honesta para comparar os modelos aqui.
    """
    X, y, le = build_xy(df)
    classes = list(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    print(f"Treino: {len(X_tr):,} | Teste: {len(X_te):,} | classes: {len(classes)}"
          .replace(",", "."))

    resultados = {}
    pipes = {}
    for name, pipe in _model_zoo().items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
        m = _evaluate(name, pipe, X_te, y_te, classes)
        m["fit_seconds"] = round(time.time() - t0, 1)
        resultados[name] = m
        pipes[name] = pipe
        t2 = m['top2_accuracy']
        t3 = m['top3_accuracy']
        print(f"  {name:32} acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
              f"f1m={m['f1_macro']:.3f} "
              f"top2={t2 if t2 is None else round(t2,3)} "
              f"top3={t3 if t3 is None else round(t3,3)} ({m['fit_seconds']}s)")

    # Seleção: melhor entre os modelos NÃO triviais (exclui baselines dummy)
    candidatos = {k: v for k, v in resultados.items()
                  if not k.startswith("dummy") and v.get(selecionar_por) is not None}
    best_name = max(candidatos, key=lambda k: candidatos[k][selecionar_por])
    best_pipe = pipes[best_name]
    print(f"\nMelhor modelo (por {selecionar_por}): {best_name}")

    # Classification report e matriz de confusão do melhor
    y_pred_best = best_pipe.predict(X_te)
    report = classification_report(y_te, y_pred_best, target_names=classes,
                                   zero_division=0, output_dict=True)
    cm_path = _confusion_png(y_te, y_pred_best, classes)

    # Persistência
    joblib.dump(best_pipe, MODELS_DIR / "crime_classifier.joblib", compress=3)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib", compress=3)
    # O scaler do pipeline vencedor, isolado (atende à lista de entregáveis)
    joblib.dump(best_pipe.named_steps["scaler"],
                MODELS_DIR / "preprocessing_pipeline.joblib", compress=3)

    metrics_out = {
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "n_classes": len(classes),
        "classes": classes,
        "features": FEATURES,
        "melhor_modelo": best_name,
        "selecionado_por": selecionar_por,
        "modelos": resultados,
        "classification_report_melhor": report,
        "arquivos": {
            "modelo": rel_to_root(MODELS_DIR / "crime_classifier.joblib"),
            "label_encoder": rel_to_root(MODELS_DIR / "label_encoder.joblib"),
            "scaler": rel_to_root(MODELS_DIR / "preprocessing_pipeline.joblib"),
            "confusion_matrix": rel_to_root(cm_path),
        },
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    return metrics_out


if __name__ == "__main__":
    from data_loader import load_clean, get_supervised_df
    df = get_supervised_df(load_clean())
    out = train_all(df)
    print("\n== model_metrics.json salvo ==")
    print("melhor:", out["melhor_modelo"])
    for k, v in out["modelos"].items():
        print(f"  {k:24} {v}")
