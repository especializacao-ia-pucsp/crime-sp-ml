# Interpretação dos clusters

Leitura do `cluster_summary.csv` (KMeans, k = 9, sobre latitude, longitude e
hora em seno/cosseno). São hipóteses de exploração, não conclusões.

## Achados

1. **Os quatro maiores clusters ficam na mesma região central e se diferenciam
   pelo horário.** Os clusters 2, 5, 3 e 8 (juntos ~59% das ocorrências) têm
   centróides quase no mesmo ponto (lat ≈ −23,54; lon ≈ −46,64), mas hora média
   de 20,5h, 15h, 9,4h e 3h — noite, tarde, manhã e madrugada. Ou seja: o que a
   localização junta, o horário separa. É a evidência mais direta a favor da
   pergunta do trabalho.

2. **As regiões periféricas repetem a mesma lógica.** A zona leste aparece em
   dois clusters (0 e 7), um de manhã e outro à noite; a zona sudoeste, em três
   (6, 1, 4), manhã/tarde/noite. Cada região é fatiada por período do dia.

3. **Furto e Roubo aparecem no topo de todos os clusters.** Nenhum grupo
   "pertence" a um crime específico ou raro. O agrupamento reflete sobretudo a
   densidade dessas duas classes — o mesmo desbalanceamento que limita o modelo.

4. **Há uma diferenciação fraca na periferia noturna.** O cluster 4 (sudoeste,
   noite) é o único em que Roubo supera Furto no topo e em que Roubo de veículo
   entra no top-3. É sutil, mas coerente com o mapa por horário.

## Limitações

1. Silhouette ≈ 0,32: muita sobreposição entre os grupos; as fronteiras são
   frágeis e mudam com a inicialização.
2. KMeans impõe o número de grupos e assume formato aproximadamente esférico.
   DBSCAN e HDBSCAN, que poderiam achar grupos por densidade, foram instáveis
   (de 1 cluster a dezenas, ou 56–80% dos pontos virando ruído).
3. Como Furto/Roubo dominam, o resultado diz mais sobre onde/quando essas duas
   classes acontecem do que sobre crimes específicos.

## Próximos passos

1. Rodar o clustering separando por tipo de crime, ou com reamostragem, para ver
   se crimes menos frequentes formam grupos próprios.
2. Acrescentar variáveis de contexto (uso do solo, transporte, iluminação) e
   reavaliar se os grupos ficam mais separados.
