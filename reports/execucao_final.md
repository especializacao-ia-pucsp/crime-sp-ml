# Execução final — log da revisão

Registro do que foi rodado na revisão final, com resultados reais.

## Comandos executados

| Comando | Resultado |
|---|---|
| `pip install -r requirements.txt` | OK — dependências já satisfeitas (só aviso de nova versão do pip) |
| `python src/clustering.py` | OK — regenerou figuras, `cluster_summary.csv` e `clustering_metrics.json` (caminhos relativos, flag `hdbscan_disponivel`) |
| `python src/modeling.py` | OK — retreinou e regenerou `model_metrics.json` (caminhos relativos, seleção por top-1) |
| `python src/ura_ranking.py` | OK — regenerou `ura_examples.md` |
| `python run_all.py --rapido` | OK — concluído em 148s; melhor modelo por `accuracy`: random_forest |
| `python run_all.py` (completo) | OK — concluído em ~359s; regenerou EDA, 7 mapas, 3D, clustering, modelo e exemplos da URA |
| `jupyter nbconvert --execute` nos 6 notebooks | OK — 0 erros; header sem caminho absoluto |

O `run_all.py` completo foi executado uma última vez nesta revisão e terminou
sem erro. Os números do modelo são os mesmos do `--rapido` (treino determinístico
com semente fixa): nada de métrica foi alterado.

## Resultado do modelo (idêntico em --rapido e no run completo)

```
Treino: 930.909 | Teste: 310.303 | classes: 20
  dummy_prior            acc=0.362  top2=0.695  top3=0.812
  logistic_regression    acc=0.404  top2=0.695  top3=0.813
  random_forest          acc=0.457  top2=0.714  top3=0.827   <- escolhido (top-1)
  hist_gradient_boosting acc=0.443  top2=0.704  top3=0.819
  hist_gb_balanced       acc=0.153  bal_acc=0.163  top3=0.370
```

## Erros encontrados e correções feitas (nesta revisão)

- **Inconsistência 24 vs 20 classes.** A base tem 24 naturezas; o modelo usa 20
  após filtrar hora exata/coordenada válida e remover classes raras. Texto
  corrigido em README, proposta e apresentação para explicar isso.
- **Caminhos absolutos nos JSONs.** `model_metrics.json` e
  `clustering_metrics.json` gravavam `C:\GIT\...`. Adicionado `config.rel_to_root`
  e os módulos passaram a gravar caminhos relativos. JSONs regenerados.
- **Seleção do modelo por top-3.** Trocada para top-1 (`accuracy`), porque o
  top-3 é dominado pela base-rate. Justificativa no código e no README.
- **HDBSCAN sem fallback.** Import protegido por try/except; se indisponível, o
  pipeline avisa e segue com KMeans + DBSCAN.
- **Tom dos textos.** README, proposta, slides e exemplos da URA reescritos em
  linguagem mais direta e acadêmica, sem frases promocionais.

## Reprodução pela banca

Para regenerar tudo do zero (a partir da base já limpa em `data/processed/`):

```bash
pip install -r requirements.txt
python run_all.py
```

Leva ~6 minutos nesta máquina. Como o treino tem semente fixa, os números saem
iguais aos registrados aqui; mapas e gráficos 3D mudam apenas o identificador
interno de render do HTML, não o conteúdo.

## Ambiente
- Python 3.12, Windows. scikit-learn 1.8.0 (mesma versão dos `.joblib`).
