import numpy as np
import pandas as pd

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
