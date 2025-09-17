from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv 
from typing import NotRequired
import pickle
import base64
import json

from tools import Tools
import tools
from agent import Agent
from prompts import Prompts
import utils



load_dotenv()
API_KEY = os.getenv("API_KEY")

# Definição das tools
tools_list = [Tools.automl, Tools.plot_real_vs_pred, Tools.testar_estacionariedade,
            Tools.desenhar_grafo, Tools.impute_values_with_backfill_method,
            Tools.impute_values_with_mean_method, Tools.impute_values_with_nearest_method, 
            Tools.impute_values_with_linear_method,
            Tools.impute_values_with_spline_method]


# Estados
class State(TypedDict):
    msg: List[BaseMessage]             # lista de mensagens do tipo HumanMessage ou AIMessage
    step: NotRequired[int]             # etapa atual do workflow
    log: NotRequired[str]              # descrição da etapa executada (pensamento e ações do agente pandas)
    tool_output: NotRequired[list]     # saída das tools executadas
    resumos: NotRequired[list]          # histórico das etapas já resumidas. (list)
    avaliacao: NotRequired[str]        # feedback do avaliador: sim ou não
    feedback: NotRequired[str]         # feedback do avaliador
    dataframe: str                     # path do dataframe a ser analisado
    avaliador_count: int               # contador de avaliações negativas seguidas

# Inicialização do dataframe e o modelo
df = pd.DataFrame()
modelo = None


# Definição dos modelos e agentes
model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
agente_pandas = Agent(model).build('pandas', df, tools_list)
repl = next(t for t in agente_pandas.tools if isinstance(t, PythonAstREPLTool)) # pega a tool REPL para atualizar o dataframe no agente pandas

agente_llm   = Agent(model).build('llm')

# Implementação dos nós
def executa_etapa(state: State):
    global df, modelo, model, agente_pandas, repl
    # Pega a etapa atual
    step = state.get("step", 1)
    print(f">>> Entrou no nó executa_etapa - step {step}", flush=True)

    try:
        if step == 1:
            # Carregar o dataframe
            df = pd.read_csv(state["dataframe"]).drop(columns=['Date']).head(1000)
            df = utils.remover_valores_aleatorios(df, coluna="ETO", proporcao=0.01)
            repl.locals["df"] = df
            tools.df = df

            # Pega o prompt da etapa 1 e gera a mensagem para o agente pandas
            prompt = Prompts.get_prompt('Etapa 1')
            messages = [HumanMessage(content=prompt)]

            # Executa o agente pandas e extrai os logs e outputs das tools
            agent_output = agente_pandas.invoke(messages)

            intermediate_steps = agent_output.get("intermediate_steps", [])

            logs = [action_log.log for action_log, _ in intermediate_steps]
            
            tool_outputs = {}
            for action_log, observation in intermediate_steps:
                tool_name = getattr(action_log, "tool", None)
                output = utils.serialize_output(observation)
                tool_outputs[tool_name] = output   # chave = nome da tool, valor = output
            
            tool_output_final = next((v for k, v in tool_outputs.items() if "imput" in k), None)
            novo_df = pd.DataFrame(tool_output_final)
            

            # Atualiza o dataframe global e o dataframe nas tools
            repl.locals["df"] = novo_df
            df = novo_df
            tools.df = novo_df

            new_messages = state["msg"] + [AIMessage(content=logs)]


        elif step == 2:

            # Pega a última mensagem humana para compor o prompt da etapa 2.
            last_human_message = None
            for msg in reversed(state["msg"]):
                # caso 1: já é HumanMessage
                if isinstance(msg, HumanMessage):
                    last_human_message = msg.content
                    break
                # caso 2: veio como dict serializado
                if isinstance(msg, dict) and msg.get("type") == "human":
                    last_human_message = msg.get("content")
                    break

            # Pega o prompt da etapa 2 e gera a mensagem para o agente pandas
            prompt = Prompts.get_prompt('Etapa 2', user_msg = last_human_message)
            messages = [HumanMessage(content=prompt)]

            # Executa o agente pandas e extrai os logs e outputs das tools
            # agent_output = agente_pandas.invoke(messages)

            # intermediate_steps = agent_output.get("intermediate_steps", [])

            # logs = [action_log.log for action_log, _ in intermediate_steps]
            
            # tool_outputs = {}
            # for action_log, observation in intermediate_steps:
            #     tool_name = getattr(action_log, "tool", None)
            #     output = utils.serialize_output(observation)
            #     tool_outputs[tool_name] = output   # chave = nome da tool, valor = output
            
            # # Pega a saída da tool automl
            # dict_automl = next((v for k, v in tool_outputs.items() if "automl" in k), None)
            # modelo = pickle.loads(base64.b64decode(dict_automl['modelo']))
            # tool_output_final = dict_automl['predicoes']
            # novo_df = pd.DataFrame(dict_automl['predicoes'])
            # new_messages = state["msg"] + [AIMessage(content=logs)]

            #>>>>>>>>>>>>>> Para testar sem executar o automl
            tool_output_final = {'predicoes': {'real': {0: 5.8, 1: 5.5, 2: 5.8, 3: 5.8, 4: 5.6},
            'previsto': {0: 5.086650089557475,
            1: 4.981711405165194,
            2: 4.960316273071086,
            3: 4.714007595126947,
            4: 4.5939236324712045}}}
            novo_df = pd.DataFrame(tool_output_final["predicoes"])
            modelo = pickle.load(open("model.pickle", 'rb'))
            logs = ""
            new_messages = state["msg"] + messages
            
            # Atualiza o dataframe global e o dataframe nas tools
            repl.locals["df"] = novo_df
            df = novo_df
            tools.df = novo_df
            tools.modelo = modelo

        elif step == 3:

            # Pega o prompt da etapa 3 e gera a mensagem para o agente pandas
            prompt = Prompts.get_prompt('Etapa 3')
            messages = [HumanMessage(content=prompt)]

            # Executa o agente pandas e extrai os logs e outputs das tools
            agent_output = agente_pandas.invoke(messages)

            intermediate_steps = agent_output.get("intermediate_steps", [])

            logs = [action_log.log for action_log, _ in intermediate_steps]
            
            tool_outputs = {}
            for action_log, observation in intermediate_steps:
                tool_name = getattr(action_log, "tool", None)
                output = utils.serialize_output(observation)
                tool_outputs[tool_name] = output   # chave = nome da tool, valor = output
            
            # Pega a saída da tool plot_real_vs_pred
            tool_output_final = next((v for k, v in tool_outputs.items() if "plot_real_vs_pred" in k), None)

            new_messages = state["msg"] + [AIMessage(content=logs)]

        elif step == 4:

            # Pega o prompt da etapa 4 e gera a mensagem para o agente pandas
            prompt = Prompts.get_prompt('Etapa 4', modelo = modelo)
            messages = [HumanMessage(content=prompt)]

            # Executa o agente pandas e extrai os logs e outputs das tools
            agent_output = agente_pandas.invoke(messages)

            intermediate_steps = agent_output.get("intermediate_steps", [])

            logs = [action_log.log for action_log, _ in intermediate_steps]
            
            tool_outputs = {}
            for action_log, observation in intermediate_steps:
                tool_name = getattr(action_log, "tool", None)
                output = utils.serialize_output(observation)
                tool_outputs[tool_name] = output   # chave = nome da tool, valor = output
            
            # Pega a saída da tool desenhar_grafo
            tool_output_final = next((v for k, v in tool_outputs.items() if "desenhar_grafo" in k), None)

            new_messages = state["msg"] + [AIMessage(content=logs)]

    except Exception as e:
        logs = f"Erro na execução da etapa {step}: {e}"
        tool_output_final = ""
        new_messages = state["msg"] + [AIMessage(content=logs)]

    return {
                "messages": new_messages,
                "log": logs,
                "tool_output": state.get("tool_output", []) + [tool_output_final]
            }

def avalia_etapa(state: State):
    print(">>> Entrou no nó avalia_etapa", flush=True)
    global agente_llm
    
    prompt_avalia = Prompts.get_prompt('Avaliação')
    out = agente_llm.invoke(prompt_avalia)

    json_out = json.loads(out.content)

    avaliacao = json_out['avaliacao']
    feedback = json_out['feedback']

    if avaliacao == "não":
        state['avaliador_count'] = state['avaliador_count'] + 1

    if state['avaliador_count'] > 4:
        avaliacao = "sim"
        feedback = ""

        state['avaliador_count'] = 0

    print("Avaliação: " + avaliacao)
    print("Feedback: " + feedback)
    
    state["avaliacao"] = avaliacao
    state['feedback'] = feedback

    if avaliacao == "sim":
        state['feedback'] = ""
    
    return state


def proxima_etapa(state: State):
    step = state.get("step") + 1
    return {"step": step}


def resume_etapa(state: State):
    print(">>> Entrou no nó resume_etapa", flush=True)
    
    global agente_llm

    # json to text
    def _to_text(o):
        try:
            return json.dumps(o, ensure_ascii=False, indent=2)
        except Exception:
            return str(o)

    try:
        step = state.get("step")

        #   get logs and tool_outputs from state
        raw_logs = state.get("log", "")
        if isinstance(raw_logs, list):
            steps_str = "\n".join(str(x) for x in raw_logs if x is not None)
        else:
            steps_str = str(raw_logs) if raw_logs is not None else ""

        #  colect tool outputs
        outputs_list = state.get("tool_output", [])
        if isinstance(outputs_list, list) and outputs_list:
            # outputs comes as a list of lists
            outputs_str = "\n\n---\n\n".join(_to_text(o) for o in outputs_list)
        else:
            outputs_str = ""

        #  get prompts for resumo
        prompt = Prompts.get_prompt(
            'Resumo',
            steps=steps_str,
            outputs=outputs_str
        )

        # agents invoke
        agent_output = agente_llm.invoke([HumanMessage(content=prompt)])

        # extract resumo text
        resumo_txt = (
            agent_output.get("output")
            if isinstance(agent_output, dict)
            else str(agent_output)
        ) or "Sem dados para resumir."


        # update state
        new_messages = state.get("msg", []) + [
            AIMessage(content=f"[Resumo - etapa {step}]\n{resumo_txt}")
        ]
        resumos = state.get("resumos", [])
        resumos.append(resumo_txt)

        state["msg"] = new_messages
        state["resumos"] = resumos
        
        return state
    except Exception as e:
        # exception handling
        falha = f"Falha no nó resume_etapa (step={state.get('step')}): {e}"
        new_messages = state.get("msg", []) + [AIMessage(content=falha)]
        state["msg"] = new_messages
        resumos = state.get("resumos", [])
        resumos.append(falha)
        state["resumos"] = resumos
        print(state["resumos"])

        return state


def finaliza(state: State):
    """Finaliza o workflow e retorna o resumo completo."""
    print(">>> Entrou no nó finaliza", flush=True)
    return state



# Nós de roteamento
def roteador_avalia_etapa(state: State):
    """Roteia para a mesma etapa ou vai para o resumo."""
    avaliacao = state.get("avaliacao")
    # step = state.get("step", 1)

    if avaliacao == 'não':
        return "refazer"
    else:
        return 'resumir'


def roteador_resume_etapa(state: State):
    """Roteia para a próxima etapa ou finaliza."""
    step = state.get("step", 1)

    if step < 4:
        return "proxima"
    else:
        return "final"


# Construção do grafo
builder = StateGraph(State)

# Adicionando nós
builder.add_node("executa_etapa", executa_etapa)
builder.add_node("tools", ToolNode(tools=tools_list))
builder.add_node("avalia_etapa", avalia_etapa)
builder.add_node("proxima_etapa", proxima_etapa)
builder.add_node("resume_etapa", resume_etapa)
builder.add_node("finaliza", finaliza)

# Adicionando arestas
builder.add_edge(START, "executa_etapa")
builder.add_conditional_edges("executa_etapa", tools_condition)
builder.add_edge("tools", "executa_etapa")
builder.add_edge("executa_etapa", "avalia_etapa")
builder.add_conditional_edges(
    "avalia_etapa",
    roteador_avalia_etapa,
    {
        "refazer": "executa_etapa",
        "resumir": "resume_etapa"
    }
)
builder.add_conditional_edges(
    "resume_etapa",
    roteador_resume_etapa,
    {
        "proxima": "proxima_etapa",
        "final": "finaliza"
    }
)
builder.add_edge("proxima_etapa", "executa_etapa")
builder.add_edge("finaliza", END)


# Para incluir memória inclua checkpointer no compile.
# checkpointer = MemorySaver()

# No langsmith, não use o checkpointer, pois a própria ferramenta já salva o histórico.
graph = builder.compile()

# Compilando o workflow
# graph = builder.compile(checkpointer=checkpointer) 


# Desabilite o app.invoke para executar com o langsmith
final_state = graph.invoke(
    {"msg": [HumanMessage(content="Faça a previsão de 5 passos à frente para a coluna ETO.")],
    'step': 1,
    "log": "",
    "tool_output": [],
    "resumo": [],
    "avaliacao": "sim",
    "feedback": "",
    "dataframe": 'https://raw.githubusercontent.com/PatriciaLucas/AutoML/refs/heads/main/Datasets/CLIMATIC_2.csv',
    "avaliador_count": 0
    },
    config={"configurable": {"api_key": API_KEY, "thread_id": 42}}
)
