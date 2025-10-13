from langchain.tools import tool
from pydantic import BaseModel, ConfigDict
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.utils.missing_values import fill_missing_values
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
from AUTODCETS import autodcets
import io
import base64
import emd
import re
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import matplotlib.cm as cm
import numpy as np
import matplotlib.lines as mlines
from typing import List
import pickle
from typing import List, Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("API_KEY")

class SchemaTools(BaseModel):
    coluna: str


class SchemaAutoml(BaseModel):
    input: str

 

class Tools:

    @staticmethod
    def rag(documento):
        global retriever

        # Load documentos
        loader = CSVLoader(file_path=documento)
        docs = loader.load()
        
        # Split documentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)
        
        # Create VectorStore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="docs",
            embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY),
        )
        retriever = vectorstore.as_retriever()
        return retriever


    # Tool para RAG
    @staticmethod
    @tool
    def retrieve_context(query: str):
        """Pesquise sobre o que foi feito durante as etapas do pipeline para previsão de séries temporais."""
        global retriever
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    

    @staticmethod
    @tool(args_schema=SchemaAutoml, return_direct=True)
    def automl(input: str) -> pd.DataFrame:
        """
        Faz previsões de vários passos à frente usando o método AUTODCETS.
        Argumentos:
        - target: nome da coluna a ser prevista.
        - step_ahead: número de passos à frente para prever.
        - max_lags: número máximo de defasagens a serem consideradas no modelo.
        - decomposition: se True, aplica decomposição EMD antes da previsão.
        Retorna um DataFrame com as previsões.

        Formato do input: {target: nome_da_coluna, step_ahead: número_passos, max_lags: valor, decomposition: true/false}
        """
        global df
        
        # Extrair parâmetros com regex
        target = re.search(r'target\s*:\s*["\']?(\w+)["\']?', input).group(1)
        step_ahead = int(re.search(r'step_ahead\s*:\s*(\d+)', input).group(1))
        max_lags = int(re.search(r'max_lags\s*:\s*(\d+)', input).group(1))
        decomposition = re.search(r'decomposition\s*:\s*(\w+)', input).group(1).lower() == "true"

        train, test = df.head(int(df.shape[0] - step_ahead)), df.tail(step_ahead)

        model_automl = autodcets.AUTODCETS(
            params_MEOHP={'npop': 4, 'ngen': 2, 'size_train': 200, 'size_test': 50},
            feature_selection=True,
            distributive_version=False,
            save_model=True,
            path_model='model',
            decomposition=decomposition,
            max_lags=max_lags,
            test_size=0,
            optimize_hiperparams=True
        )
        model_automl.fit(train, target)

        if decomposition:
            forecast = model_automl.predict_ahead(step_ahead=step_ahead)[0]
        else:
            forecast = model_automl.predict_ahead_multivariate(step_ahead=step_ahead)[target]

        real = test[target]

        output = pd.concat([real.reset_index(drop=True), forecast.reset_index(drop=True)], axis=1)
        output.columns = ["real "+target, "previsto "+target]

        # Serializa o modelo em binário
        model_bytes = pickle.dumps(model_automl)

        # Converte para base64 para retornar como string (compatível com agente)
        model_b64 = base64.b64encode(model_bytes).decode("utf-8")

        return {
            "predicoes": output.to_dict(),  
            "modelo": model_b64 
        }


    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def impute_values_with_linear_method(coluna: str) -> dict:
        """
        Imputa valores ausentes na coluna usando interpolação linear.
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna um DataFrame imputado.

        Formato do input: nome_da_coluna
        """
        global df
        
        coluna = coluna.strip("'").strip('"')
        df.index = pd.to_datetime(df.index, errors="raise")
        series = TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=None)

        if df.isna().any().any():
            filler = MissingValuesFiller(fill="auto")
            series_filled = filler.transform(series)
        else:
            series_filled = series

        df = series_filled.to_dataframe().to_dict()
        return df
    
    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def impute_values_with_spline_method(coluna: str) -> pd.DataFrame:
        """
        Imputa valores ausentes na coluna no DataFrame df utilizando interpolação spline (ordem 3).
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna um DataFrame imputado.

        Formato do input: nome_da_coluna
        """
        global df
        # 1) Garante índice datetime
        df.index = pd.to_datetime(df.index, errors="raise")

        # 2) Cria TimeSeries preenchendo datas
        series = TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=None)

        # 3) Interpola se necessário com spline (ordem 3)
        if df.isna().any().any():
            series_filled = fill_missing_values(
                series,
                fill="auto",
                method="spline",
                order=3,
                limit_direction="both"
            )
        else:
            series_filled = series
        df = series_filled.to_dataframe()
        return series_filled.to_dataframe()

    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def impute_values_with_backfill_method(coluna: str) -> pd.DataFrame:
        """
        Imputa valores ausentes na coluna no DataFrame df utilizando o método de preenchimento para trás (Backward Fill).
        O preenchimento para trás deve ser utilizado quando se assume que o valor ausente pode ser adequadamente representado pela próxima observação disponível.
        É mais indicado quando os dados variam lentamente e não há forte sazonalidade ou padrões complexos que invalidem essa suposição.
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna um DataFrame imputado.
        Formato do input: nome_da_coluna
        """
        global df
        df = df.fillna(method="bfill")

        return df

    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def impute_values_with_mean_method(coluna: str) -> pd.DataFrame:
        """
        Imputa valores ausentes na coluna no DataFrame df utilizando a média dos dados.
        A imputação pela média deve ser utilizada em séries temporais aproximadamente estacionárias, onde a média é representativa do comportamento dos dados.
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna um DataFrame imputado.

        Formato do input: nome_da_coluna
        """
        global df
        df = df.fillna(df.mean())

        return df

   
    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def impute_values_with_nearest_method(coluna: str) -> pd.DataFrame:
        """
        Imputa valores ausentes na coluna no DataFrame utilizando o valor mais próximo.
        A imputação pelo valor mais próximo é adequada quando os dados apresentam pouca variação entre pontos adjacentes.
        É especialmente útil em dados categóricos ou ordinais, onde faz sentido substituir o valor ausente pelo vizinho mais próximo observado.
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna um DataFrame imputado.

        Formato do input: nome_da_coluna
        """
        global df
        df.index = pd.to_datetime(df.index, errors="raise")

        df = df.interpolate(method='nearest', limit_direction='both')

        return df

    @staticmethod
    @tool(args_schema=SchemaTools)
    def EMD(coluna: str) -> pd.DataFrame:
        """Decompõe uma série temporal usando o método Empirical Mode Decomposition (EMD).
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna o dataframe imf com as componentes do EMD.

        Formato do input: nome_da_coluna
        """
        global df, imf
        serie = df[coluna].values

        imf = emd.sift.sift(serie)
        imf = pd.DataFrame(imf)

        return imf

    @staticmethod
    @tool(args_schema=SchemaTools)
    def testar_estacionariedade(coluna: str) -> str:
        """Determina se uma série temporal é estacionária.
        Argumento: coluna deve conter o nome de uma coluna existente no DataFrame df.
        Retorna uma mensagem indicando se a série é estacionária ou não.

        Formato do input: nome_da_coluna
        """
        global df
        serie = df[coluna]

        try:
            # Teste ADF
            adf_resultado = adfuller(serie)
            adf_stat = adf_resultado[0]
            adf_p = adf_resultado[1]
            adf_conclusao = (
                "estacionária" if adf_p < 0.05 else "não estacionária"
            )

            # Teste KPSS
            kpss_resultado = kpss(serie, regression='c', nlags="auto")
            kpss_stat = kpss_resultado[0]
            kpss_p = kpss_resultado[1]
            kpss_conclusao = (
                "não estacionária" if kpss_p < 0.05 else "estacionária"
            )

            if adf_conclusao == "estacionária" and kpss_conclusao == "estacionária":
                return "A série temporal é provavelmente estacionária."
            elif adf_conclusao == "não estacionária" and kpss_conclusao == "não estacionária":
                return "A série temporal é provavelmente não estacionária."
            else:
                return "A série temporal é provavelmente não estacionária."

        except Exception as e:
            return f"Erro ao aplicar os testes: {str(e)}"


    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def plot_column_base64(coluna: str) -> str:
        """
        Gera um gráfico de linha da coluna especificada no DataFrame df.
        Retorna a imagem em base64.

        Formato do input: {nome_da_coluna}
        """
        global df
        # Garante que o índice é datetime
        df.index = pd.to_datetime(df.index)

        # Geração do gráfico
        plt.figure(figsize=(10, 5))
        df[coluna].plot(title=f"{coluna}", grid=True)
        plt.xlabel("Data")
        plt.ylabel(coluna)

        # Salvando como imagem em memória
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Convertendo para base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64
    
    @staticmethod
    @tool(return_direct=True)
    def plot_real_vs_pred() -> str:
        """
        Gera um gráfico de linhas comparando valores reais e previstos
        de uma série temporal.
        Retorna a imagem em base64.

        Formato do input: essa ferramenta não requer input.
        """
        global df

        output = df
        y_real = output[df.columns[0]].tolist()
        y_pred = output[df.columns[1]].tolist()

        # Verifica se as listas têm o mesmo tamanho
        if len(y_real) != len(y_pred):
            raise ValueError("As listas y_real e y_pred devem ter o mesmo tamanho.")

        # Cria um índice temporal simples (0,1,2,...)
        index = range(len(y_real))

        # Converte para DataFrame
        df_plot = pd.DataFrame({"Real": y_real, "Previsto": y_pred}, index=index)

        # Geração do gráfico
        plt.figure(figsize=(10, 5))
        plt.plot(df_plot.index, df_plot["Real"], label="Real", color="blue")
        plt.plot(df_plot.index, df_plot["Previsto"], label="Previsto", color="orange")
        plt.title("Valores reais vs previstos")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        # Salvando como imagem em memória
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Convertendo para base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    @staticmethod
    @tool(args_schema=SchemaTools, return_direct=True)
    def plot_imf(coluna: str) -> str:
        """
        Gera um gráfico de linha das componentes geradas pela EMD.
        Retorna a imagem em base64.

        Formato do input: {nome_da_coluna}
        """
        global imf
        n_imfs = imf.shape[1]
        fig, axs = plt.subplots(n_imfs, 1, figsize=(12, 2 * n_imfs), sharex=True)

        if n_imfs == 1:
            axs = [axs]  # Garante iterabilidade

        for i, ax in enumerate(axs):
            ax.plot(imf.iloc[:, i], label=f'IMF {i+1}')
            ax.set_ylabel(f'IMF {i+1}')
            ax.legend(loc='upper right')
            ax.grid(True)

        axs[-1].set_xlabel("Índice da série")
        fig.suptitle(f"IMFs da coluna '{coluna}'")
        plt.tight_layout()

        # Salva imagem em buffer e converte para base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64


    @staticmethod
    def get_edges(importance, model):
        arestas = []
        for imp in importance:
            if importance[imp][0] != -1:
                arestas.append((imp, model.target+" "+str(0)))
        return arestas

    @staticmethod
    def get_importance(model):
        G = model.G_list[model.target]
        lags = G.where(G).stack().index.tolist()
        model_target = model.dict_variables[model.target]["trained_model"]
        feature_importance_df = pd.DataFrame(model_target.feature_importances_, columns=['importances'])
        importance = {}
        max_value = max(x[0] for x in lags)
        i=0

        for l in lags:
            importance[str(l[1])+" "+str(l[0])] = [float(feature_importance_df.iloc[i].values[0]), max_value - l[0]]
            i += 1
        importance[model.target+" "+str(0)] = [-1, max_value]
        return importance, max_value

    @staticmethod
    def get_variables(model):
        variables = list(model.G_list[model.target].columns)
        variables.sort(key=lambda x: 0 if x == model.target else 1)
        return variables
    
    @staticmethod
    def renomear_imf_para_target(d: dict, target) -> dict:
        """
        Troca qualquer ocorrência de 'IMF<digitos>' por target nas chaves.
        """
        return {re.sub(r'\bIMF\d+\b', target, k): v for k, v in d.items()}

    
    @staticmethod
    def agrupar_dicts(model, dicts: List[Dict[str, List[float]]], min_value: float = -1) -> Dict[str, List[float]]:
        """
        Agrupa uma lista de dicionários (cada valor é uma lista [v1, v2]).
        Regras:
        - Se a key for repetida, soma o primeiro valor (v1).
        - O primeiro valor final é limitado inferiormente por min_value (default -1).
        - O segundo valor final é calculado como: max_lags - <numero_da_key>
        Retorna: dict com a mesma set de keys e valores [v1_final, v2_final].
        """

        G = model.G_list[model.target]
        lags = G.where(G).stack().index.tolist()
        max_lags = max(x[0] for x in lags)  # valor máximo de lag

        # 1) soma os primeiros valores por key
        sums = {}
        for d in dicts:
            for k, (v1, _) in d.items():
                sums[k] = sums.get(k, 0.0) + float(v1)

        # 2) aplica limite inferior (min_value)
        for k in sums:
            if sums[k] < min_value:
                sums[k] = min_value

        # 3) calcula o segundo valor como max_lags - número da key
        result = {}
        for k in sums:
            m = re.search(r"\s(\d+)$", k)
            if m:
                num = int(m.group(1))
            else:
                num = 0  # fallback
            v2 = max_lags - num
            result[k] = [sums[k], v2]

        return result, max_lags
    

    @staticmethod
    def normalize_importance(data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Normaliza o primeiro valor de cada lista em um dicionário,
        mantendo o segundo valor inalterado. Valores iguais a -1 não são normalizados.

        Args:
            data (Dict[str, List[float]]): Dicionário de entrada com listas [valor, índice].

        Returns:
            Dict[str, List[float]]: Dicionário com valores normalizados.
        """
        # extrair valores que não são -1
        valores = [v[0] for v in data.values() if v[0] != -1]

        from sklearn.preprocessing import normalize
        valores = normalize([valores])

        # normalização linear
        normalized = {}
        i=0
        for k, v in data.items():
            if v[0] == -1:
                normalized[k] = v
            else:
                normalized[k] = [valores[0][i], v[1]]
                i += 1
        return normalized
    
    @staticmethod
    @tool(return_direct=True)
    def desenhar_grafo():

        """ Desenha o grafo de importância das variáveis do modelo.
        Argumento: model deve ser um modelo treinado do AUTODCE-TS.
        Retorna a imagem do grafo em base64.
        """
        
        global modelo

        if modelo.decomposition:
            target_original = modelo.target
            dfs_filtrados = {k: v for k, v in modelo.G_list.items() if "IMF" in k}
            df_final = pd.concat(dfs_filtrados.values()).groupby(level=0).any()
            df_final_filtrados = {k: v for k, v in df_final.items() if "IMF" in k}
            df_final = df_final.drop(df_final.filter(like="IMF").columns, axis=1)
            df_target = pd.concat(df_final_filtrados.values()).groupby(level=0).any()
            df_final[modelo.target] = df_target.values
            modelo.G_list[modelo.target] = df_final
            keys = modelo.G_list.keys()
            imf_keys = [k for k in keys if "IMF" in k]

            importancias_imf = []
            for k in imf_keys:
                modelo.target = k
                imp = Tools.get_importance(modelo)
                importancias_imf.append(Tools.renomear_imf_para_target(imp[0], target_original))

            modelo.target = target_original
            
            importance, max_value = Tools.agrupar_dicts(modelo, importancias_imf)

            print(importance)

            importance = Tools.normalize_importance(importance)
                
        else:
            importance, max_value = Tools.get_importance(modelo)

        variables = Tools.get_variables(modelo)
        arestas = Tools.get_edges(importance, modelo)

        G = nx.DiGraph()
        G.add_nodes_from(importance.keys())
        G.add_edges_from(arestas)

        # Mapeamento de variável para posição no eixo y
        var_to_y = {v: i+1 for i, v in enumerate(variables)}

        # Posições: eixo x = lag, eixo y = variável
        pos = {}
        for f, lag in importance.items():
            var = f.rsplit(" ", 1)[0]
            y = var_to_y[var]
            pos[f] = (lag[1], y)

        # Defini importância para cor
        imp_vals = np.array([v[0] for v in importance.values()])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFEE00FF","#FF5200"])
        colors = []
        for imp in imp_vals:
            if imp < 0:
                colors.append("#4169E1")
            else:
                colors.append(cmap(imp))
        

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # Configuração dos eixos do gráfico
        ax.set_xlabel("Lags", fontsize=12)
        ax.set_ylabel("Variables", fontsize=12)
        
        # Definir ticks e labels
        space_lags = np.linspace(-0.15, 5.0, max_value+1)
        ax.set_xticks(space_lags)
        # list_lags = make_time_labels(max_value+1)
        ax.set_xticklabels([], fontsize=10)
        
        space_nodes = np.linspace(0.8, len(variables)+0.2, len(variables))
        ax.set_yticks(space_nodes)
        ax.set_yticklabels(variables, fontsize=10)

        # Fixar limites dos eixos
        ax.set_xlim(-1, max_value+1)
        ax.set_ylim(-0.3, len(variables)+1)

        # Criar um eixo secundário do NetworkX
        ax_graph = fig.add_axes(ax.get_position(), frameon=False)
        ax_graph.set_xlim(-1.20, max_value+2)
        ax_graph.set_ylim(-0.3, len(variables)+1)


        labels = {n: n.rsplit(" ", 1)[0] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black', ax=ax_graph)
        nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cm.Reds, 
                            node_size=800, edgecolors='black', linewidths=0.8, ax=ax_graph)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20, 
                            connectionstyle='arc3,rad=0.3', width=1, ax=ax_graph)
        

        # Barra de cores
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Feature Importance", rotation=270, labelpad=15)

        # Esconder os eixos do gráfico NetworkX
        ax_graph.set_xticks([])
        ax_graph.set_yticks([])
        ax_graph.set_facecolor('none')

        ax.grid(False)

        endog_legend = mlines.Line2D([], [], 
                                    color='#4169E1', 
                                    marker='o', 
                                    linestyle='None',
                                    markersize=10,
                                    label='Endogenous variable at time t')

        ax.legend(handles=[endog_legend], 
                loc='center left', 
                bbox_to_anchor=(0, 1.05),
                fontsize=7)
        
        plt.savefig('grafo.png', bbox_inches='tight')
        plt.tight_layout()
        # plt.show()

        # Salva imagem em buffer e converte para base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {'imagem': img_base64,
                'importancia': importance}
