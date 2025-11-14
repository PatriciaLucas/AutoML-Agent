
import numpy as np
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
import os

def remover_valores_aleatorios(df, coluna="Close", proporcao=0.1):
    """
    Remove aleatoriamente uma proporção de valores da coluna especificada no DataFrame.
    Plota a série resultante com os valores faltantes visíveis.
    """
    df_modificado = df.copy()

    n = len(df_modificado)
    k = int(n * proporcao)

    # Seleciona posições (inteiros) aleatórias
    posicoes = np.random.choice(n, size=k, replace=False)

    # Apaga os valores com base nas posições
    df_modificado.iloc[posicoes, df.columns.get_loc(coluna)] = np.nan
    pd.DataFrame(df_modificado, columns=[coluna])

    return df_modificado


def serialize_output(output):
    if isinstance(output, np.generic):
        return output.item()
    elif isinstance(output, pd.DataFrame):
        return output.reset_index().to_dict(orient="records")
    elif isinstance(output, pd.Series):
        return output.to_dict()
    elif isinstance(output, (list, dict, str, int, float, bool)) or output is None:
        return output
    else:
        return str(output)
    

def to_jsonable(x):
    import numpy as np, pandas as pd
    from datetime import date, datetime
    from pathlib import Path
    from enum import Enum
    if isinstance(x, (np.bool_, np.number)): return x.item()
    if isinstance(x, np.ndarray):            return x.tolist()
    if isinstance(x, pd.DataFrame):          return x.to_dict(orient="records")
    if isinstance(x, pd.Series):             return x.to_list()
    if isinstance(x, (set, tuple)):          return [to_jsonable(v) for v in x]
    if isinstance(x, (datetime, date)):      return x.isoformat()
    if isinstance(x, (Path, Enum)):          return str(x)
    if isinstance(x, dict):                  return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):                  return [to_jsonable(v) for v in x]
    return x


class CaptureStepsHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.logs.append(f"Start chain: {serialized.get('name', '')}")

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"Action: {action.tool}\nInput: {action.tool_input}")

    def on_tool_end(self, output, **kwargs):
        self.logs.append(f"Output: {output}")

    def on_agent_finish(self, finish, **kwargs):
        self.logs.append(f"Finish: {finish.return_values}")

def carregar_csvs_pasta(pasta):
    documentos = []
    metadata = []
    # Para cada csv na pasta
    for arquivo in pasta.glob("*.csv"):
        loader = CSVLoader(file_path=str(arquivo))
        docs = loader.load()
        documentos.extend(docs)
        metadata.extend()

    return documentos, metadata


def append_list_csv(csv_path: str, data: list[str]):
    """
    Adiciona uma nova linha a um arquivo CSV. Cria o arquivo caso não exista. Adiciona os dados 
    por meio do modo 'append'.
    
    Args:
        csv_path: caminho para o csv
        data: dicionário em que as chaves correspondem aos nomes das colunas e os valores são os dados a serem salvos
    """
    header = ["resumos"]
    data_to_df = [[resumo] for resumo in data]
    df = pd.DataFrame(data_to_df, columns=header)
    df.to_csv(csv_path, index=False, encoding='utf-8')
