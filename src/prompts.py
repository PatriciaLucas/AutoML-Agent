
class Prompts:

    def get_prompt(name: str, **kwargs) -> str:
        match name:
            case 'Etapa 1': # Imputação de valores faltantes: 
                #Verificar a possibilidade de relembrar que o agente é um assistente de análise de séries temporais, caso a resposta não seja satisfatória
                #Vamos testar se listar as tools contribui para a resposta do agente 
                prompt = "Agora foque em imputação de valores faltantes. Dada as séries temporais do dataframe df, verifique se há valores faltantes, caso haja, realize a imputação utilizando a tool mais apropriada para a série. Você tem as seguintes tools disponíveis: "

            case 'Etapa 2': # Decomposição:
                #Verificar a possibilidade de relembrar que o agente é um assistente de análise de séries temporais, caso a resposta não seja satisfatória
                #Vamos testar se listar as tools contribui para a resposta do agente 
                prompt = f"Sua tarefa é decidir se a Decomposição por Modos Empíricos (EMD) é benéfica para uma série temporal univariada. Para isso você deve verificar se a série é estacionária ou não, você tem a tool 'testar_estacionariedade' para fazer a verificação. Caso ela não seja estacionária, decomponha a série utilizando a tool 'EMD'. "

            case 'Etapa 3': # Previsão:
                #verificar a possibilidade de tratamento de exceção caso o usuário não passe o nome da coluna 
                target = kwargs.get("target", "")
                step_ahead = kwargs.get("step_ahead", 5)
                prompt = f"Faça uma previsão de {step_ahead} passos à frente da coluna {target} usando o dataframe df. O número máximo defasagens deve ser definido por você. Leve em consideração que quanto maior a entrada, maior será o custo computacional "

            case 'Etapa 4': # Explicação:
                #POSSÍVEL EXEMPLO A USAR CASO O RESULTADO DA EXPLICAÇÃO ESTEJA RUIM
                # Entrada : {coluna_alvo} = "OT"
                # Saída de testar_estacionariedade: “A série temporal é provavelmente não estacionária.”
                # Saída de EMD: DataFrame com 6 colunas.

                # Resposta final esperada (resumo):

                # A análise da série OT indicou, pelos testes ADF e KPSS, que ela é provavelmente não estacionária. Por esse motivo, o agente aplicou o método EMD, que é adequado para lidar com sinais não lineares e de frequência variável. 
                state = kwargs.get("state", "")
                tools_outputs = state.get("tool_output", "")
                prompt = f"""A partir das saídas das ferramentas utilizadas, elabore um resumo explicativo que contenha os passos realizados pelo agente e as motivações para a realização desses passos. Exemplo: O
                Saídas das ferramentas: {tools_outputs}
                """


            case 'Avaliação': # Recebe as ações do Agente Pandas e solicita ao Agente Avaliador que aprove ou não essas ações de acordo com a etapa que foi solicitada.
                
                #action = 
                #step =
                prompt = f"""Você é um avaliador de ações em um processo de análise de series temporais. Seu trabalho é decidir se uma ação deve ser aplicada ou não em uma determinada etapa. 
                            Responda apenas Sim ou Não. 
                            A ação {actions} deve ser executada para a etapa {etapa}?"""



            case 'Resumo': # Recebe as ações do Agente Pandas e solicita ao Agente Resumidor que resuma essas ações. 
                prompt = "...."
        

        return prompt
