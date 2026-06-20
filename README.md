# crime-sp-ml — análise espaço-temporal de crimes em São Paulo

Trabalho de especialização em IA usando as ocorrências criminais da SSP/SP
(2022–2026, município de São Paulo). A base já vinha coletada e limpa; aqui o
foco é a análise.

## Pergunta do trabalho

Um mapa feito só com latitude e longitude acende quase sempre as mesmas regiões
de maior circulação — Centro, Avenida Paulista, corredores comerciais — para a
maioria dos tipos de crime. Isso mostra onde há mais registro e movimento, não o
risco de quem mora ali. A pergunta é: **incluir o horário ajuda a separar melhor
os tipos de ocorrência e a ordenar opções de um atendimento?**

## O que foi feito

- Preparação das variáveis a partir da base limpa (coordenadas, hora, período).
- Análise exploratória (contagens, distribuição por hora, assinatura
  hora × tipo de crime).
- Mapas sem o horário e com o horário (densidade por período e por hora) e uma
  visualização 3D.
- Clusterização (KMeans, DBSCAN, HDBSCAN) para levantar hipóteses.
- Um modelo que recebe latitude, longitude e hora e devolve a probabilidade de
  cada tipo de crime, aplicado a um protótipo de URA do 190 (ordenar as opções
  do menu).

## Dados usados

- Fonte: Portal da Transparência da SSP/SP, anos 2022 a 2026 (corte em
  29/04/2026), filtrados para o município de São Paulo.
- 2.084.529 ocorrências após a limpeza; **24 naturezas de crime**.
- Campos centrais: `NATUREZA_APURADA` (o que prever), `LATITUDE`/`LONGITUDE`,
  `HORA_OCORRENCIA_BO` e `DESCR_PERIODO`.
- O modelo só usa registros com **hora exata** e coordenada válida. Depois de
  remover as classes com pouquíssimos exemplos nesse recorte (necessário para o
  split estratificado), sobram **20 classes** no experimento supervisionado.

## Como executar

```bash
pip install -r requirements.txt
python run_all.py            # gera tudo em reports/ e models/
python run_all.py --rapido   # só dados + modelo (pula mapas/3D/clustering)
```

Módulos isolados (a partir de `src/`): `visualization.py`, `maps.py`,
`viz3d.py`, `clustering.py`, `modeling.py`, `ura_ranking.py`. Usar o modelo:

```python
import sys; sys.path.insert(0, "src")
from ura_ranking import predict_top_crimes, gerar_opcoes_ura
predict_top_crimes(-23.5613, -46.6558, "19:00", top_k=3)
gerar_opcoes_ura(-23.5487, -46.4583, 8)
```

## Principais resultados

Modelo escolhido: **RandomForest** (entrada: latitude, longitude, hora).

| Métrica | RandomForest | Base-rate (`dummy_prior`) |
|---|---|---|
| Acerto da 1ª opção (top-1) | 0,457 | 0,362 |
| Top-2 | 0,714 | 0,695 |
| Top-3 | 0,827 | 0,812 |
| Balanced accuracy | 0,071 | 0,050 |
| Macro-F1 | 0,065 | 0,027 |

Clusterização: KMeans com k = 9 e silhouette ≈ 0,32. DBSCAN e HDBSCAN não deram
uma partição estável (variam muito com o parâmetro: de 1 cluster a dezenas, ou
56–80% de pontos classificados como ruído).

## Leitura correta dos resultados

- O ganho que dá para defender está no **top-1**: 0,457 contra 0,362 do
  baseline de prioris, ~9,5 pontos percentuais. É aí que latitude/longitude/hora
  fazem diferença.
- O **top-3 alto (0,827) engana**: o baseline que só repete as classes mais
  frequentes já chega a 0,812. As três naturezas mais comuns cobrem ~81% das
  ocorrências, então top-3 alto é mais consequência da distribuição do que mérito
  do modelo. Por isso a seleção do modelo usa top-1, não top-3.
- **Balanced accuracy e macro-F1 são baixos** (0,071 e 0,065). O modelo acerta
  as classes frequentes e erra as raras; ele não serve para prever crimes raros.
- O **silhouette ≈ 0,32** indica grupos pouco separados. O clustering é
  exploratório: levanta hipóteses, não prova estrutura.
- A utilidade do protótipo de URA é **ordenar as opções prováveis** do menu por
  local e horário, não acertar a ocorrência nem identificar crime raro.

## Estrutura do repositório

```
data/raw/         dados brutos da SSP/SP
data/processed/   base limpa (CSV comprimido) — entrada de tudo
docs/             apresentacao (.pptx/.csv)
src/              pipeline de tratamento de dados e módulos (config, data_loader, 
                  features, visualization, maps, viz3d, clustering, modeling, 
                  ura_ranking, utils)
notebooks/        01_ingestao … 06_visualizacoes (chamam src/)
reports/          figures/, maps/, métricas (.json/.csv), proposta, 
                  ura_examples, cluster_interpretation
models/           crime_classifier.joblib, label_encoder.joblib,
                  preprocessing_pipeline.joblib
run_all.py        roda tudo
```

## Limitações

- Desbalanceamento forte (Furto/Roubo ≈ 75%): o modelo concentra as previsões
  nessas classes; macro-F1 baixo.
- Os mapas de calor medem concentração de registros e circulação, não risco
  individual; a Paulista aparece por ser área de alta circulação.
- Parte das ocorrências não tem hora exata; a aproximação por período só é usada
  na exploração/clustering, nunca no modelo.
- Nenhum achado é causal — usamos "há indícios", "sugere".

## Ética

É um protótipo acadêmico. A URA só **ordena opções de menu** e sempre oferece
"falar com atendente"; não decide despacho nem direciona policiamento. Modelos
treinados em dados históricos podem reproduzir vieses de notificação e de
registro, e por isso um uso real exigiria auditoria e transparência.

## Arquivos gerados

- `reports/figures/` — EDA, matriz de confusão, 3D (HTML).
- `reports/maps/` — mapas folium (sem tempo, por período/hora, Paulista, clusters).
- `reports/model_metrics.json`, `clustering_metrics.json`, `cluster_summary.csv`.
- `reports/ura_examples.md`, `cluster_interpretation.md`, `checklist_entrega.md`, 
  `execucao_final.md`.
- `models/*.joblib`.
