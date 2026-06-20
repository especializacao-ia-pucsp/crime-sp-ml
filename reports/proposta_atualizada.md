# Proposta atualizada

## Título
Análise espaço-temporal de ocorrências criminais em São Paulo com clusterização
e um modelo preditivo para ordenar opções de uma URA de emergência (190).

## O que mudou em relação à proposta original
A proposta inicial terminava na coleta, limpeza e visualização dos dados da
SSP/SP. Quando começamos a olhar os mapas, percebemos que plotar só por
latitude/longitude acende sempre as mesmas regiões (Centro, Paulista), para
quase todo tipo de crime. Isso é pouco útil para decidir qualquer coisa. A
partir daí o trabalho passou a investigar se o **horário** acrescenta informação
e a testar isso de duas formas: agrupando as ocorrências (clusterização) e
treinando um modelo que usa local e hora para ordenar os tipos de crime mais
prováveis. O caso de uso escolhido foi ordenar as opções de uma URA do 190.

## Problema
Mapas de criminalidade feitos apenas com coordenadas concentram a leitura nas
áreas de maior circulação. Eles misturam tipos e horários muito diferentes num
mesmo calor visual e medem volume de registro, não risco. A pergunta é direta:
incluir o horário ajuda a separar melhor os tipos de ocorrência?

## Objetivos
- Geral: verificar se latitude, longitude e horário, juntos, organizam melhor os
  tipos de ocorrência do que só a localização.
- Específicos: preparar as variáveis a partir da base limpa; descrever o
  desbalanceamento; comparar mapas com e sem o horário; aplicar clusterização;
  treinar e comparar modelos de classificação; entregar funções de aplicação
  (`predict_top_crimes`, `gerar_opcoes_ura`).

## Base de dados
- SSP/SP, Portal da Transparência, 2022–2026 (corte 29/04/2026), município de SP.
- 2.084.529 ocorrências após a limpeza; 24 naturezas de crime.
- Campos centrais: `NATUREZA_APURADA` (alvo), `LATITUDE`/`LONGITUDE`,
  `HORA_OCORRENCIA_BO`, `DESCR_PERIODO`.
- A base limpa contém 24 naturezas. No experimento supervisionado, depois dos
  filtros de coordenada válida, hora exata e remoção das classes raras (poucas
  ocorrências, que inviabilizam a avaliação estratificada), o modelo final
  trabalha com 20 dessas classes. É uma decisão metodológica, não um descarte
  de dado: as 4 classes deixadas de fora têm exemplos de menos para treinar e
  medir com confiança.

## Metodologia
- **Preparação.** Coordenadas convertidas para número e filtradas pelo retângulo
  do município (descarte das que vinham como zero ou fora da cidade). Hora
  convertida para inteiro e também para seno/cosseno, para que 23h e 0h fiquem
  próximas. O período textual (manhã/tarde/noite) foi normalizado, mas só é usado
  na exploração; o modelo usa apenas hora exata.
- **Por que clustering.** Antes de classificar, quisemos ver se existe alguma
  estrutura espaço-temporal nos dados — agrupamentos naturais por região e
  horário. O clustering aqui é exploratório: serve para gerar hipóteses sobre
  como as ocorrências se distribuem, não para provar que há grupos bem definidos.
- **Por que classificação.** A pergunta do trabalho tem um lado prático: se local
  e hora carregam informação, deve ser possível usá-los para ordenar os tipos de
  crime mais prováveis. O modelo materializa isso e permite medir o ganho com
  métricas, comparando com um baseline que só usa a frequência das classes.
- **Avaliação.** Split estratificado com semente fixa; accuracy, balanced
  accuracy, macro/weighted F1, top-2 e top-3, matriz de confusão. O tipo de crime
  nunca entra como variável de entrada (seria vazamento).

## Limitações
- A base é muito desbalanceada (Furto e Roubo somam ~75%). O modelo acerta as
  classes comuns e vai mal nas raras; isso aparece no macro-F1 e na balanced
  accuracy baixos, e não tem como esconder.
- O top-3 fica alto, mas o baseline de prioris já chega perto: as três classes
  mais frequentes cobrem ~81% dos casos. O ganho defensável está no top-1.
- O clustering não separou bem (silhouette ≈ 0,32); DBSCAN/HDBSCAN foram
  instáveis. Tratamos os resultados como hipótese.
- Os dados são de registro: refletem subnotificação e padrões de circulação, não
  o risco real por morador.

## Ética
O protótipo de URA apenas ordena as opções do menu por local e horário e sempre
mantém "falar com atendente". Não decide despacho, não substitui o atendente e
não direciona policiamento. Como o modelo aprende de dados históricos, ele pode
reproduzir vieses de quem registra e de onde se patrulha mais; qualquer uso além
do acadêmico exigiria auditoria desses vieses e transparência sobre os dados.

## Próximos passos
- Acrescentar variáveis de contexto (uso do solo, transporte, iluminação) para
  explicar melhor a variação espacial.
- Tratar o desbalanceamento (reamostragem/calibração) e medir o ganho real.
- Validação temporal (treinar no passado, testar no futuro) antes de pensar em
  qualquer uso operacional.
