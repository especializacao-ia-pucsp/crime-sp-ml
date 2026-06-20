# Exemplos de URA do 190

A ideia da URA é simples: a partir de **(latitude, longitude, hora)** da ligação, o modelo estima quais tipos de ocorrência são mais prováveis naquele ponto e horário e usa isso para **ordenar as opções do menu**. O menu mostra no máximo **2 tipos** e termina **sempre** com a opção de *falar com atendente* — quem liga nunca fica preso ao menu.

Importante: isto ordena opções, não classifica a ocorrência nem decide o despacho. As probabilidades abaixo aparecem só para inspeção; o cidadão vê apenas o texto do menu.

## Av. Paulista — 19h (noite, comercial)
`lat=-23.5613, lon=-46.6558, hora=19`

Ranking (top-3):

| rank | crime | probabilidade |
|---|---|---|
| 1 | ROUBO - OUTROS | 46.8% |
| 2 | FURTO - OUTROS | 39.1% |
| 3 | FURTO DE VEICULO | 8.0% |

**Texto da URA:**

> Polícia Militar de São Paulo, 190. Para agilizar seu atendimento, selecione o tipo de ocorrência: 1. Roubo. 2. Furto. 3. Falar com atendente.

## Centro/Sé — 3h (madrugada)
`lat=-23.5505, lon=-46.6333, hora=3`

Ranking (top-3):

| rank | crime | probabilidade |
|---|---|---|
| 1 | FURTO - OUTROS | 43.5% |
| 2 | ROUBO - OUTROS | 43.0% |
| 3 | LESAO CORPORAL DOLOSA | 5.4% |

**Texto da URA:**

> Polícia Militar de São Paulo, 190. Para agilizar seu atendimento, selecione o tipo de ocorrência: 1. Furto. 2. Roubo. 3. Falar com atendente.

## Zona Leste — 8h (manhã)
`lat=-23.5487, lon=-46.4583, hora=8`

Ranking (top-3):

| rank | crime | probabilidade |
|---|---|---|
| 1 | FURTO - OUTROS | 26.8% |
| 2 | FURTO DE VEICULO | 26.6% |
| 3 | ROUBO - OUTROS | 25.4% |

**Texto da URA:**

> Polícia Militar de São Paulo, 190. Para agilizar seu atendimento, selecione o tipo de ocorrência: 1. Furto. 2. Furto de veículo. 3. Falar com atendente.

## Zona Sul — 21h (noite, periférica)
`lat=-23.668, lon=-46.7082, hora=21`

Ranking (top-3):

| rank | crime | probabilidade |
|---|---|---|
| 1 | ROUBO - OUTROS | 42.4% |
| 2 | FURTO - OUTROS | 18.8% |
| 3 | ROUBO DE VEICULO | 13.6% |

**Texto da URA:**

> Polícia Militar de São Paulo, 190. Para agilizar seu atendimento, selecione o tipo de ocorrência: 1. Roubo. 2. Furto. 3. Falar com atendente.

---
O ponto a observar é a mudança de ordem conforme o contexto: de manhã na Zona Leste o *furto de veículo* sobe; à noite na região central o *roubo* aparece em primeiro. É essa reordenação por local e horário que justifica o protótipo — não acertar o crime exato, e muito menos prever ocorrências raras.
