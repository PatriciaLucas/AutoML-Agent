class Prompts:

    def get_prompt(name: str, **kwargs) -> str:
        match name:
            case 'Etapa 1': # Imputação de valores faltantes:
                #Verificar a possibilidade de relembrar que o agente é um assistente de análise de séries temporais, caso a resposta não seja satisfatória
                #Vamos testar se listar as tools contribui para a resposta do agente 
                prompt = f"""Você é um assistente de pré-processamento de dados de séries temporais. 
                Foque em imputação de valores faltantes.
                Verifique se há valores faltantes em todo o dataframe.
                Caso haja valores faltantes, realize a imputação utilizando a tool mais apropriada.
                Use o formato: {{nome da coluna}} para qualquer tool, sem aspas.
                """


            case 'Etapa 2': # Previsão:
                #verificar a possibilidade de tratamento de exceção caso o usuário não passe o nome da coluna 
                user_msg = kwargs.get("user_msg", "")
                prompt = f"""Você é um assistente de previsão de séries temporais. Foque apenas na tarefa de previsão e solucione o problema: {user_msg}.
                O número máximo de defasagens e o uso da decomposição deve ser definido por você.
                O número máximo de defasagens deve ser definido de forma a equilibrar a precisão e o custo computacional, 
                então escolha valores entre 1 e 30.
                Para chamar a tool automl use o formato: {{target: 'nome da coluna', step_ahead: 5, max_lags: 15, decomposition: true}}, 
                não use aspas nos nomes dos parâmetros.
                """

            case 'Etapa 3': # Visualização real x previsto
                
                prompt = f'''Foque na tarefa de visualização de dados.
                 O dataframe contém os valores previstos e reais.
                 Gere a figura na Base64 que contenha o gráfico comparando os valores reais e previstos.
                 
                 '''

            case 'Etapa 4': # Visualização grafo causal:
                modelo = kwargs.get("modelo", "")
                prompt = f"""Use a tool desenhar_grafo para gerar o grafo de importância das variáveis do modelo {modelo}.
                """


            case 'Avaliação': # Recebe as ações do Agente Pandas e solicita ao Agente Avaliador que aprove ou não essas ações de acordo com a etapa que foi solicitada.
                
                action = ""
                step = ""
                prompt = f"""Você é um avaliador de ações em um processo de análise de series temporais. Seu trabalho é decidir se uma ação deve ser aplicada ou não em uma determinada etapa. 
                            Responda apenas Sim ou Não. 
                            A ação {actions} deve ser executada para a etapa {etapa}?"""



            case 'Resumo':  # Recebe as ações do Agente Pandas e solicita ao Agente Resumidor que resuma essas ações.
                # Espera receber via kwargs:
                # - steps: string com linhas "THOUGHT: ..." e "ACTION: tool(args)" (na ordem em que ocorreram)
                # - outputs: string com os outputs brutos das tools (concatenados, na mesma ordem)
                steps = kwargs.get("steps", "")
                outputs = kwargs.get("outputs", "")
                prompt = f"""
                    Você é o Agente Resumidor. Receberá dois blocos abaixo:

                    - steps: linhas alternadas no padrão 'THOUGHT: ...' e 'ACTION: tool(args)'
                    - outputs: respostas brutas das tools (texto/tabelas/descrições), na ordem em que ocorreram

                    OBJETIVO
                    Escreva um relatório em pt-BR exatamente no formato:

                    Steps:
                    1) Nome do Step em Português: <título curto em pt-BR>
                    Motivo: <copie/sintetize fielmente o THOUGHT correspondente, sem inventar>
                    Ferramenta Escolhida: <nome da tool indicada em ACTION ou 'nenhuma'>
                    Motivo da escolha da Ferramenta: <explique brevemente por que essa tool foi usada; se não houver, escreva 'não aplicável'>
                    Resultado do "Step": <resuma o output associado a este step; se não houver output, escreva '—'>

                    2) Nome do Step em Português: ...
                    Motivo: ...
                    Ferramenta Escolhida: ...
                    Motivo da escolha da Ferramenta: ...
                    Resultado do "Step": ...

                    REGRAS
                    - Mantenha a ordem cronológica dos steps.
                    - Não invente ferramentas, dados ou resultados.
                    - Associe outputs aos steps na mesma ordem (1º output para o 1º step que teve ACTION, e assim por diante).
                    - Seja conciso e técnico; evite floreios.
                    - Se não houver THOUGHT ou ACTION para algum step, preencha com '—' no campo faltante.
                    - Se não houver nada para resumir, responda apenas: "Sem dados para resumir."

                    DADOS
                    Intermediate Steps:
                    {steps}

                    Outputs:
                    {outputs}
                    """.strip()
        

        return prompt
