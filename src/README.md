Instruções para executar o pipeline de dados:

```
usage: pipeline.py [-h] source destination

Importa e processa os dados criminais disponibilizados pela SSP/SP

positional arguments:
  source       Nome do XLSX que será importado. Aceita coringas como * e ?
               para múltiplos arquivos, mas lembre de escapá-los usando a
               contrabarra (\).
  destination  Caminho e nome do arquivo CSV que será gerado

options:
  -h, --help   show this help message and exit
```

Aqui também estão separados os módulos de execução de aprendizado de máquina
que devem ser chamados pelo `run_all.py` que está na raiz do repositório.

Lembre de antes baixar os dados da SSP/SP e executar o `pipeline.py` para
gerar o arquivo com os dados já tratados para serem utilizados pelo
`run_all.py`.
