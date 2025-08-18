from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Estados
class State(TypedDict):
    msg: str              # mensagem do usuário
    step: str             # etapa atual do workflow
    log: str              # descrição da etapa executada (pensamento e ações do agente pandas)
    tool_output: list     # saída das tools executadas
    resumo: list          # histórico das etapas já resumidas.


# Implementação dos nós
def executa_etapa(state: State):

    return state

def avalia_etapa(state: State):
    
    return state

def proxima_etapa(state: State):

    return

def resume_etapa(state: State):
    
    return state

def finaliza(state: State):
    
    return state

# Nós de roteamento
def roteador_avalia_etapa(state: State):

    return 'resumir'

def roteador_resume_etapa(state: State):
    
    return 'final'


tools = []


# Construção do grafo
builder = StateGraph(State)

# Adicionando nós
builder.add_node("executa_etapa", executa_etapa)
builder.add_node("tools", ToolNode(tools=tools))
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

# Compilando o workflow
graph = builder.compile() 
