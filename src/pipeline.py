#!/usr/bin/python3
#
#   pipeline.py:    Lê os arquivos de dados criminais disponibilizados pela
#                   SSP/SP, remove os dados que não são interessantes para
#                   este projeto e executa uma série de transformações:
#
#                   - corrige tipos
#                   - unifica colunas
#                   - padroniza o formato de escrita
#                   - corrige dados inválidos
#                   - preenche dados faltantes
#
#                   As funções estão na ordem em que são executadas pelo
#                   pipeline e os comentários descrevem o que estão fazendo
#                   e qual o motivo. São funções específicas para esse tipo
#                   de dados e este pipeline, estão separadas mais para
#                   criterio de organização. Ler os comentários e código
#                   dessas funções sequencialmente mostra uma visão completa
#                   e detalhada da execução do pipeline.
#
#                   Se quiser ter uma visão de alto nível das etapas do
#                   pipeline, a função `run` mostra essa visão, chamando cada
#                   uma das outras na ordem correta.
#

import argparse
import glob
import pandas as pd
from utils import (
    import_excel_data,
    convert_string_to_time,
    convert_something_to_int,
    convert_to_plain_uppercase,
)

# define o DataFrame como global
df_original = pd.DataFrame


def carga_dos_dados(file_path):
    """
    O primeiro passo é a carga dos arquivos fornecidos pela SSP/SP.

    Como esses arquivos são razoavelmente grandes e demoram para ser
    baixados, o execução do pipeline lê apenas arquivos locais, que devem
    salvos com antecedência no diretório `../data/raw` (ou em outro que
    seja informado em tempo de execução.
    """
    global df_original

    print("1. Carregando os dados dos arquivos da SSP/SP:")
    df_original = import_excel_data(file_path, skip_sheets=1)


def selecionando_o_municipio_de_Sao_Paulo():
    """
    Antes de começar qualquer trabalho de ajuste e limpeza de dados, vamos
    reduzir o `df_original` ao nosso escopo definido: a cidade de São Paulo.

    Infelizmente existem dois campos com a informação da cidade, alguns dos
    arquivos da SSP/SP têm `CIDADE` e outros usam `NOME_MUNICIPIO`; mas,
    felizmente, todos eles têm o campo `COD IBGE`, e o código do IBGE para
    o município de São Paulo é 3550308, segundo o próprio site do instituto
    (https://www.ibge.gov.br/explica/codigos-dos-municipios.php)
    """
    global df_original

    print("2. Filtrando pelo município de São Paulo: ", end="")
    df_original = df_original[df_original["COD IBGE"] == 3550308]
    print("OK")


def removendo_as_colunas_desnecessarias_ou_redundantes():
    """
    Juntando todos os arquivos, essas são as colunas do DataFrame unificado:

        ```
        NOME_DEPARTAMENTO                          object
        NOME_SECCIONAL                             object
        NOME_DELEGACIA                             object
        NOME_MUNICIPIO                             object
        NUM_BO                                     object
        ANO_BO                                      int64
        DATA_REGISTRO                      datetime64[ns]
        DATA_OCORRENCIA_BO                         object
        HORA_OCORRENCIA_BO                         object
        DESC_PERIODO                               object
        DESCR_TIPOLOCAL                            object
        DESCR_SUBTIPOLOCAL                         object
        BAIRRO                                     object
        LOGRADOURO                                 object
        NUMERO_LOGRADOURO                          object
        LATITUDE                                   object
        LONGITUDE                                  object
        NOME_DELEGACIA_CIRCUNSCRICAO               object
        NOME_DEPARTAMENTO_CIRCUNSCRICAO            object
        NOME_SECCIONAL_CIRCUNSCRICAO               object
        NOME_MUNICIPIO_CIRCUNSCRICAO               object
        RUBRICA                                    object
        DESCR_CONDUTA                              object
        NATUREZA_APURADA                           object
        MES_ESTATISTICA                             int64
        ANO_ESTATISTICA                             int64
        CMD                                        object
        BTL                                        object
        CIA                                        object
        COD IBGE                                   object
        NOME_DELEGACIA_CIRCUNSCRIÇÃO               object
        NOME_DEPARTAMENTO_CIRCUNSCRIÇÃO            object
        NOME_SECCIONAL_CIRCUNSCRIÇÃO               object
        NOME_MUNICIPIO_CIRCUNSCRIÇÃO               object
        CIDADE                                     object
        DATA_COMUNICACAO_BO                        object
        DESCR_PERIODO                              object
        ```

    As quatro colunas com `CIRCUNSCRIÇÃO` no nome tem duas versões, uma
    com e outra sem os acentos no nome.

    Também estão duplicadas: `DESC_PERIODO` e `DESCR_PERIODO`; e as menos
    evidentes: `NOME_MUNICIPIO` e `CIDADE`.

    Agora começar os ajustes e limpeza de dados.

    A limpeza começa removendo as colunas com dados que não serão utilizados:
    """
    global df_original

    to_drop = []

    # CIDADE, NOME_MUNICIPIO e COD IBGE não tem mais utilidade agora que os
    # dados já foram filtrados por município:
    to_drop.extend(["CIDADE", "NOME_MUNICIPIO", "COD IBGE"])

    # Para data e hora utilizaremos os dados que estão em:
    # DATA_OCORRENCIA_BO, HORA_OCORRENCIA_BO e DESC[R]_PERIODO. As outras
    # colunas de data e hora devem ser removidas:
    to_drop.extend(
        [
            "ANO_BO",
            "DATA_REGISTRO",
            "DATA_COMUNICACAO_BO",
            "MES_ESTATISTICA",
            "ANO_ESTATISTICA",
        ]
    )

    # Os dados da referentes a Polícia Civil (delegacia, número do BO, etc)
    # não serão utilizados e podem ser removidos:
    to_drop.extend(
        [
            "NOME_DELEGACIA_CIRCUNSCRICAO",
            "NOME_DEPARTAMENTO_CIRCUNSCRICAO",
            "NOME_SECCIONAL_CIRCUNSCRICAO",
            "NOME_MUNICIPIO_CIRCUNSCRICAO",
            "NOME_DELEGACIA_CIRCUNSCRIÇÃO",
            "NOME_DEPARTAMENTO_CIRCUNSCRIÇÃO",
            "NOME_SECCIONAL_CIRCUNSCRIÇÃO",
            "NOME_MUNICIPIO_CIRCUNSCRIÇÃO",
            "NOME_DEPARTAMENTO",
            "NOME_SECCIONAL",
            "NOME_DELEGACIA",
            "NUM_BO",
        ]
    )

    # As colunas `RUBRICA` e `NATUREZA_APURADA` contém o mesmo tipo de
    # informação, qual crime foi cometido. Porém a `RUBRICA` tem uma
    # variabilidade bem maior (maiúsculas, minúsculas, ordem dos termos,
    # etc), `NATUREZA_APURADA` é tem os dados mais padronizados.
    to_drop.extend(["RUBRICA"])

    print("3. Removendo colunas que não serão utilizadas: ", end="")
    df_original.drop(columns=to_drop, inplace=True, errors="ignore")
    print("OK")


def unificando_colunas_DESC_PERIODO_e_DESCR_PERIODO():
    """
    Depois das remoções de colunas que não serão utilizadas, algumas
    transformações, para tornar os dados mais fáceis de trabalhar.
    """
    global df_original

    # Unificar as colunas `DESCR_PERIODO` e `DESC_PERIODO` na
    # `DESCR_PERIODO` que fica no mesmo padrão das outras colunas
    # de descrição
    print("4. Unificando as descrições do período: ", end="")
    if "DESC_PERIODO" in df_original.columns:
        if "DESCR_PERIODO" in df_original.columns:
            df_original["DESCR_PERIODO"] = df_original["DESCR_PERIODO"].fillna(
                df_original["DESC_PERIODO"]
            )
        else:
            df_original["DESCR_PERIODO"] = df_original["DESC_PERIODO"]
        df_original.drop(columns="DESC_PERIODO", inplace=True)
    print("OK")


def padronizando_os_dados_da_HORA_OCORRENCIA_BO():
    # A coluna `HORA_OCORRENCIA_BO` não é lá muito padronizada e tem dados
    # no formato HH:MM:SS e HH:MM:SS.s+ o primeiro está convertido para
    # datetime.time automaticamente, o segundo precisa ser convertido.
    # Essa transformação preserva os NaN
    global df_original

    print("5. Convertendo os dados de horário para o tipo correto: ", end="")
    df_original["HORA_OCORRENCIA_BO"] = df_original["HORA_OCORRENCIA_BO"].apply(
        convert_string_to_time
    )
    print("OK")


def padronizando_o_tipo_do_NUMERO_LOGRADOURO():
    # A coluna `NUMERO_LOGRADOURO` está com os números no tipo float, mas
    # número de rua são inteiros. Não existe a casa 23.7 da rua.
    global df_original

    print("6. Convertendo os números da rua para o tipo correto: ", end="")
    df_original["NUMERO_LOGRADOURO"] = df_original["NUMERO_LOGRADOURO"].apply(
        convert_something_to_int
    )
    print("OK")


def padronizando_caixa_e_removendo_diacriticos():
    # Padroniza as colunas `BAIRRO`, `LOGRADOURO` e `NATUREZA_APURADA` para
    # maiúsculas e sem acentos.
    global df_original

    print("7. Remover acentos e mudar para maiúsculas: ", end="")
    to_convert = ["BAIRRO", "LOGRADOURO", "NATUREZA_APURADA"]
    for col_to_convert in to_convert:
        df_original[col_to_convert] = df_original[col_to_convert].apply(
            convert_to_plain_uppercase
        )
    print("OK")


def removendo_informacao_de_numero_do_LOGRADOURO():
    # `LOGRADOURO` será utilizado para gerar outros dados, importante reforçar
    # a padronização. Algumas vezes o `LOGRADOURO` está junto com o
    # `NUMERO_LOGRADOURO`, na mesma coluna.
    global df_original

    print("8. Remover informação de número da rua do local errado: ", end="")
    df_original["LOGRADOURO"] = df_original["LOGRADOURO"].str.replace(
        r"([A-Z0-9-_ ]+),(.*)", r"\1", regex=True
    )
    print("OK")


def preenchendo_LATITUDE_e_LONGITUDE_faltantes():
    # `LATITUDE` e `LONGITUDE` são importantes para as nossas análises,
    # infelizmente uma série de linhas não tem essas informações.
    #
    # Várias linhas tem o valor `NUL,L`, aqui substituímos por NaN
    global df_original

    print("9. Preencher os dados faltantes de LATITUDE e LONGITUDE: ")
    df_original.replace("NUL,L", pd.NA, inplace=True)

    # As linhas são agrupadas pelo `LOGRADOURO` e os que estiverem
    # com NaN na `LATITUDE` ou `LONGITUDE` são substituídas pela
    # última informação válida coletada no mesmo `LOGRADOURO`
    print("\t- Com base no LOGRADOURO: ", end="")
    df_original[["LATITUDE", "LONGITUDE"]] = df_original.groupby("LOGRADOURO")[
        ["LATITUDE", "LONGITUDE"]
    ].ffill()
    print("OK")

    # As linhas são agrupadas pelo `BAIRRO` e os que estiverem
    # com NaN na `LATITUDE` ou `LONGITUDE` são substituídas pela
    # última informação válida coletada no mesmo `BAIRRO`
    print("\t- Com base no BAIRRO: ", end="")
    df_original[["LATITUDE", "LONGITUDE"]] = df_original.groupby("BAIRRO")[
        ["LATITUDE", "LONGITUDE"]
    ].ffill()
    print("OK")

    # O que mesmo assim está como NaN na `LATITUDE` ou `LONGITUDE`
    # é dropado
    df_original = df_original.dropna(subset=["LATITUDE", "LONGITUDE"])


def escrevendo_arquivo_de_saida(file_path):
    global df_original

    print("10. Escrevendo os dados processados: ")
    print(f"\t{file_path}: ", end="")
    df_original.to_csv(file_path)
    print("OK")


def run(source_path, dest_path):

    # 1. Carregando os dados dos arquivos da SSP/SP
    carga_dos_dados(source_path)

    # 2. Filtrando pelo município de São Paulo
    selecionando_o_municipio_de_Sao_Paulo()

    # 3. Removendo colunas que não serão utilizadas
    removendo_as_colunas_desnecessarias_ou_redundantes()

    # 4. Unificando as descrições do período
    unificando_colunas_DESC_PERIODO_e_DESCR_PERIODO()

    # 5. Convertendo os dados de horário para o tipo correto
    padronizando_os_dados_da_HORA_OCORRENCIA_BO()

    # 6. Convertendo os números da rua para o tipo correto
    padronizando_o_tipo_do_NUMERO_LOGRADOURO()

    # 7. Remover acentos e mudar para maiúsculas
    padronizando_caixa_e_removendo_diacriticos()

    # 8. Remover informação de número da rua do local errado
    removendo_informacao_de_numero_do_LOGRADOURO()

    # 9. Preencher os dados faltantes de LATITUDE e LONGITUDE
    preenchendo_LATITUDE_e_LONGITUDE_faltantes()

    # 10. Escrevendo os dados processados
    escrevendo_arquivo_de_saida(dest_path)


def main():

    # Para sempre imprimir o help quando houver um erro e também para deixar
    # o erro mais destacado
    class OurArgumentParser(argparse.ArgumentParser):
        def error(self, error_text):
            print(f"\n \033[41m\033[1;37m *** ERRO! {error_text} *** \033[0m \n")
            self.print_help()
            exit(1)

    # Lê os argumentos de entrada do programa
    parser = OurArgumentParser(description="""
                        Importa e processa os dados criminais 
                        disponibilizados pela SSP/SP
                        """)
    parser.add_argument(
        "source",
        nargs=1,
        help="""
                        Nome do XLSX que será importado. Aceita coringas como 
                        * e ? para múltiplos arquivos, mas lembre de escapá-los
                        usando a contrabarra (\\).
                        """,
    )
    parser.add_argument(
        "destination",
        nargs=1,
        help="""
                        Caminho e nome do arquivo CSV que será gerado
                        """,
    )
    args = parser.parse_args()

    # Verifica se o arquivo de origem existe
    source_path = args.source[0]
    if len(glob.glob(source_path)) == 0:
        parser.error(f"{source_path} não encontrado!")

    # Verifica se é possível escrever no arquivo de destino
    dest_path = args.destination[0]
    try:
        with open(dest_path, "w"):
            pass
    except Exception as e:
        parser.error(f"Não é possível escrever em {dest_path}!")

    # Executa o pipeline propriamente dito
    run(source_path, dest_path)


if __name__ == "__main__":
    main()
