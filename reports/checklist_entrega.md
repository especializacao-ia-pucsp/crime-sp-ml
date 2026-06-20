# Checklist de entrega

Marcado após a revisão final. Detalhes da execução em
[execucao_final.md](execucao_final.md).

- [x] Código executa (`run_all.py`, módulos de `src/`)
- [x] `run_all.py` completo rodado nesta revisão (~6 min, sem erro) — `execucao_final.md`
- [x] Métricas do texto conferem com `model_metrics.json` / `clustering_metrics.json`
- [x] README atualizado (enxuto, leitura honesta dos resultados)
- [x] Proposta atualizada (tom acadêmico, mudança de escopo explicada)
- [x] Apresentação gerada (`apresentacao.pptx` + `.pdf`)
- [x] Caminhos relativos nos JSONs (sem caminho absoluto de máquina)
- [x] Inconsistência 24/20 classes resolvida (24 na base, 20 no experimento)
- [x] Limitações explícitas (desbalanceamento, macro-F1 baixo, silhouette baixo)
- [x] Ética explícita (ordenação de menu, não decisão; atendente sempre presente)
- [x] Seleção do modelo por top-1 (não por top-3) com justificativa
- [x] `NATUREZA_APURADA` não é usada como feature do modelo (sem vazamento)
- [x] `LabelEncoder` salvo; `predict_proba` e top-k funcionando
- [x] HDBSCAN com fallback (segue com KMeans + DBSCAN se indisponível)
- [x] Notebooks re-executados nesta revisão (01–06), 0 erros — ver `execucao_final.md`
- [x] `requirements.txt` instalável e enxuto
