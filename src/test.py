import os
import pandas as pd
from dotenv import load_dotenv  
from agent import Agent
from tools import Tools
from prompts import Prompts
import tools
from graph import Graph
from utils import remover_valores_aleatorios
from AUTODCETS import datasets
from IPython.display import Image, display
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()
API_KEY = os.getenv("API_KEY")

name = 'ENERGY_1'
df = pd.read_csv('CLIMATIC_2.csv').head(20)
df = df.drop(columns=['Date'])
df = remover_valores_aleatorios(df, 'ETO', 0.1)


tools.df = df
agent_type = 'pandas'
# model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model = "deepseek-ai/DeepSeek-V3.1"

tools_ob = [Tools.impute_values_with_linear_method, Tools.impute_values_with_spline_method]

chat = Agent(model).build(agent_type, df, tools_ob)

graph_ob = Graph(chat, tools_ob).build_graph()

config = {
    "configurable": {
        "api_key": API_KEY,  # Your actual key
        "model": model,  # Your model
        "thread_id": "1"  # Optional conversation tracking
    }
}

# 1ª Etapa - Imputação de valores faltantes
user_msg ='Faça a imputação de valores faltantes da coluna ETO.'

prompt = Prompts.get_prompt('Etapa 1', user_msg = user_msg)

estado = graph_ob.invoke({"messages": [HumanMessage(content=prompt)]}, config)

print(estado['agent_output'])




# 2ª Etapa - Previsão
user_msg ='Faça a previsão de 3 passos à frente da coluna ETO usando o dataframe df.'

prompt = Prompts.get_prompt('Etapa 3', user_msg = user_msg)

estado2 = graph_ob.invoke({"messages": [HumanMessage(content=prompt)]}, config)

print(estado['agent_output'])

