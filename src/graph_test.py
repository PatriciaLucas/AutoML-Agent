from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import numpy as np
import pandas as pd


class Graph:
    def __init__(self, agent, tools):
        self.agent = agent
        self.tools = tools

    def serialize_output(self, output):
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

    def executa_etapa(self, state):
        agent_output = self.agent.invoke(state["messages"])
        intermediate_steps = agent_output.get("intermediate_steps", [])
        logs = [log.log for log, _ in intermediate_steps]
        tool_outputs = [self.serialize_output(output) for _, output in intermediate_steps]

        return {
            "logs": state.get("logs", []) + logs,
            "all_tool_outputs": state.get("all_tool_outputs", []) + tool_outputs
        }

    def build_graph(self):
        class State(TypedDict):
            messages: list
            logs: list
            all_tool_outputs: list
            agent_output: str

        builder = StateGraph(State)
        builder.add_node("chatbot", self.executa_etapa)
        # builder.add_node("tools", ToolNode(tools=self.tools))

        builder.add_edge(START, "chatbot")
        # builder.add_conditional_edges("chatbot", tools_condition)
        # builder.add_edge("tools", "chatbot")
        builder.add_edge("chatbot", END)

        return builder.compile(checkpointer=MemorySaver())