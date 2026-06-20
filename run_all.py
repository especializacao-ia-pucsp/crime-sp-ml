#!/usr/bin/python3
#
# run_all.py: Orquestra o projeto crime-sp-ml de ponta a ponta, de forma
#             idempotente, a partir do dataset já processado.
#
#   Etapas (na ordem):
#     1. Carga + limpeza (gera cache)         -> src/data_loader.py
#     2. Análise exploratória (figuras PNG)   -> src/visualization.py
#     3. Mapas folium (sem/com tempo)         -> src/maps.py
#     4. Visualização 3D (Plotly)             -> src/viz3d.py
#     5. Clusterização (KMeans/DBSCAN/HDBSCAN)-> src/clustering.py
#     6. Modelo supervisionado + métricas     -> src/modeling.py
#     7. Exemplos de URA                      -> src/ura_ranking.py
#
#   Uso:
#     python run_all.py            # roda tudo
#     python run_all.py --rapido   # pula mapas/3D/clustering (só dados+modelo)
#
#   Os artefatos vão para reports/ e models/.

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

# Torna os módulos de src/ importáveis sem instalação
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def _banner(txt: str):
    print("\n" + "=" * 70 + f"\n  {txt}\n" + "=" * 70)


def main(rapido: bool = False):
    t0 = time.time()

    import data_loader
    import visualization
    import modeling
    import ura_ranking

    _banner("1/7 — Carga e limpeza do dataset")
    df = data_loader.load_clean()
    info = data_loader.resumo_dataset(df)
    for k, v in info.items():
        print(f"   {k:26}: {v:,}".replace(",", "."))

    _banner("2/7 — Análise exploratória (figuras)")
    figs = visualization.generate_all(df)
    print(f"   {len(figs)} figuras em reports/figures/")

    if not rapido:
        import maps
        import viz3d
        import clustering

        _banner("3/7 — Mapas folium")
        mp = maps.generate_all(df)
        print(f"   {len(mp)} mapas em reports/maps/")

        _banner("4/7 — Visualização 3D")
        v3 = viz3d.generate_all(df)
        print(f"   {len(v3)} html(s) 3D em reports/figures/")

        _banner("5/7 — Clusterização")
        cl = clustering.run_all(df)
        print(f"   k={cl['k_escolhido']} | silhouette={cl['silhouette_k']:.3f}")
    else:
        print("\n[--rapido] pulando mapas, 3D e clustering.")

    _banner("6/7 — Modelo supervisionado")
    sup = data_loader.get_supervised_df(df)
    met = modeling.train_all(sup)
    melhor = met["modelos"][met["melhor_modelo"]]
    print(f"   melhor: {met['melhor_modelo']} | acc={melhor['accuracy']:.3f} "
          f"top3={melhor['top3_accuracy']:.3f} bal_acc={melhor['balanced_accuracy']:.3f}")

    _banner("7/7 — Exemplos de URA do 190")
    for nome, lat, lon, h in ura_ranking.CASOS_REFERENCIA:
        print(f"   {nome}: {ura_ranking.gerar_opcoes_ura(lat, lon, h)}")
    print("   ->", ura_ranking.salvar_exemplos_md())

    _banner(f"CONCLUÍDO em {time.time() - t0:.0f}s")
    print("Artefatos: reports/figures, reports/maps, reports/*.json/.csv, models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline completo crime-sp-ml")
    ap.add_argument("--rapido", action="store_true",
                    help="pula mapas/3D/clustering (só dados + modelo)")
    args = ap.parse_args()
    main(rapido=args.rapido)
