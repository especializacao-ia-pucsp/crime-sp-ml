#!/usr/bin/python3
#
# utils.py: Este modulo tem uma série de funções razoavelmente genéricas
#           utilizadas originalmente para processar os arquivos de dados
#           criminais da SSP/SP.
#
#           Pode ser importado com:
#
#              import utils
#
#           Ou podem ser importadas funções individuais:
#
#               from utils import import_excel_data
#
#           Todas as funções tem doc strings descrevendo o seu uso,
#           parâmetros de entrada e de saída.


import glob
import pandas as pd
import unicodedata
from datetime import datetime, date, time
from joblib import Parallel, delayed


def import_excel_data(pattern_path: str, skip_sheets: int = 0):
    """
    Importa em um dataframe o conteúdo dos arquivos Excel da SSP/SP

    Parâmetros:
      pattern_path (str): O padrão que será utilizado para o nome dos arquivos
      skip_sheets (int): A partir de qual aba serão importados os dados

    Retorno
      concat_df (pd.DataFrame): Dataframe com o conteúdo dos Excels indicados
    """
    file_paths = glob.glob(pattern_path)

    def load_excel_file(file_path):
        # Aqui é lido e interpretado o arquivo Excel. Ele está separado
        # do `pd.read_excel` porque assim ele é carregado na memória apenas
        # uma vez.
        print(f"\t{file_path}")
        xlsf = pd.ExcelFile(file_path)

        # O parâmetro `sheet_name` pode ser uma lista com as abas
        # dentro do arquivo XLSX mas é mais fácil fazer a concatenação
        # uma de cada vez
        parallel_sheets = Parallel(
            n_jobs=len(xlsf.sheet_names[skip_sheets:]), require="sharedmem"
        )(
            delayed(pd.read_excel)(xlsf, header=0, sheet_name=sheet_name)
            for sheet_name in xlsf.sheet_names[skip_sheets:]
        )
        concat_sheets_df = pd.concat(parallel_sheets, ignore_index=True)

        return concat_sheets_df

    parallel_dfs = Parallel(n_jobs=-1)(
        delayed(load_excel_file)(file_path) for file_path in file_paths
    )
    concat_df = pd.concat(parallel_dfs, ignore_index=True)

    return concat_df


def convert_string_to_time(original_time):
    """
    Converte strings no formato HH:MM:SS* para datetime.time

    Parâmetros:
      original_time: A entrada pode ser qualquer tipo, mas apenas os dados
      do tipo str serão convertidos

    Retorno:
      datetime.time: Se a entrada foi uma string no formato HH:MM:SS*
    """
    hhmmss = original_time
    if isinstance(original_time, str):
        hhmmss = time(int(original_time[0:2]),
                      int(original_time[3:5]),
                      int(original_time[6:8]))
    return hhmmss


def convert_something_to_int(original_data):
    """
    Converte a entrada de vários tipos diferentes em inteiros.

    Parâmetros:
        original_data: A entrada pode ser qualquer de qualquer tipo

    Retorno:
        ret_int: Se a entrada já for um inteiro, retorna ele mesmo.
                 Se a entrada for um float, é convertido para inteiro.
                 Se for qualquer outra coisa, retorna 0 (zero).
    """
    if isinstance(original_data, int):
        return original_data

    if isinstance(original_data, float):
        ret_int = int(original_data)
    else:
        ret_int = int(0.0)

    return ret_int


def convert_to_plain_uppercase(original_text):
    """
    Converte as strings de entrada em maiúsculas e sem acento.

    Parâmetros:
        original_text: A entrada deve ser algum tipo conversível em string

    Retorno:
        plain_text: Texto plano em maiúsculas e sem diacríticos
    """
    decoded_text = unicodedata.normalize("NFKD", str(original_text))
    plain_text = "".join(
        [char for char in decoded_text if not unicodedata.combining(char)]
    )
    return plain_text.upper()


def main():
    print("Este módulo não deve ser chamado diretamente!")
    exit(1)


if __name__ == "__main__":
    main()
