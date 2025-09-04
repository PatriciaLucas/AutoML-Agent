from langchain_community.chat_models import ChatDeepInfra
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent
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
                                                df=df,
                                                verbose=True,
                                                allow_dangerous_code=True,
                                                return_intermediate_steps=True,
                                                agent_type="zero-shot-react-description",
                                                handle_parsing_errors=True,
                                                extra_tools=tools,
                                                max_interations=10,
                                                prefix=prefix,
                                                sufix=sufix
                                                )
        elif name == 'react':
            return initialize_agent(
                                    tools=tools, 
                                    llm=self.llm,
                                    agent="zero-shot-react-description",
                                    verbose=True
                                    )
        
        else:
            return self.llm
