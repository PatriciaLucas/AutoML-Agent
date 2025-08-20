from langchain_community.chat_models import ChatDeepInfra
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd


class Agent:
    def __init__(self, model):

        self.llm = ChatDeepInfra(
            model=model,
            temperature=0
        )

    def build(self, name, df=None, tools=None, prefix = None, sufix = None):

        if name == 'pandas':
            return create_pandas_dataframe_agent(
            llm=self.llm,
            df=pd.DataFrame(df),
            verbose=True,
            allow_dangerous_code=True,
            return_intermediate_steps=True,
            agent_type="zero-shot-react-description",
            extra_tools=tools,
            max_interations=10,
            prefix=prefix,
            sufix=sufix
            )
        else: # Para agentes avaliadores e resumidores
            return self.llm
