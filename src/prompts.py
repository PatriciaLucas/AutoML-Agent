
class Prompts:

    def get_prompt(name: str, **kwargs) -> str:
        match name:
            case 'Etapa 1': # Imputação de valores faltantes:
                #Verificar a possibilidade de relembrar que o agente é um assistente de análise de séries temporais, caso a resposta não seja satisfatória
                #Vamos testar se listar as tools contribui para a resposta do agente               
                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente especializado em pré-processamento de séries temporais, com foco na **imputação de valores faltantes**.

                Regras:
                1. Verifique todas as colunas do dataframe para detectar valores faltantes.
                2. Caso existam valores faltantes, utilize a tool mais apropriada para imputação.
                3. A tool 'testar_estacionariedade' não pode ser usada nesta etapa.

                Se atente a observação a seguir: {feedback}
                """
                # prompt = prompt + feedback


            case 'Etapa 2': # Previsão:
                #verificar a possibilidade de tratamento de exceção caso o usuário não passe o nome da coluna 
                user_msg = kwargs.get("user_msg", "")
                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente especializado em previsão de séries temporais. 
                Sua única tarefa é gerar previsões para o problema descrito pelo usuário: {user_msg}.

                Regras:
                1. Analise a série temporal e defina automaticamente, sempre com uma breve justificativa:
                - O número máximo de defasagens (`max_lags`), escolhendo entre 1 e 50 para equilibrar precisão e custo computacional.
                - Se a decomposição (`decomposition`) deve ser usada ou não.
                2. Utilize a tool `testar_estacionariedade` caso seja necessário decidir sobre o uso da decomposição.

                Se atente a observação a seguir: {feedback}
                """
                   

            case 'Etapa 3': # Visualização real x previsto

                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente especializado em visualização de séries temporais.

                Regras:
                1. Foque apenas na tarefa de visualização de dados.
                2. O dataset possui duas colunas referentes aos valores reais e previstos.
                2. Gere um gráfico comparando os valores reais e previstos.
                3. Retorne a figura **em Base64** pronta para uso, sem explicações ou texto adicional.
                
                Se atente a observação a seguir: {feedback}
                """


            case 'Etapa 4': # Visualização grafo causal:
                modelo = kwargs.get("modelo", "")
                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente de análise de modelos.

                Regras:
                1. Responda **somente com texto estruturado**, sem dicionários, código ou JSON.
                2. Não use ferramentas de execução (como python_repl_ast, shell ou python).
                3. O texto deve ser objetivo e formatado com seções claras.

                Sua tarefa:
                1. Extraia informações geradas pelo AutoML {modelo} com a ferramenta extrair_informacao_automl.

                A tool retornará as seguintes informações técnicas dentro de um dicionário:
                - O dicionário 'modelo_dict' contém uma chave para cada variável. Cada variável foi prevista por um modelo específico, com seus hiperparâmetros associados.
                - O dicionário 'grafo' contém todas as variáveis com as variáveis selecionadas e seus lags.

                Sua tarefa:
                Com base nessas informações, sua resposta final deve ser o resumo textual estruturado no formato abaixo:

                Modelos:
                - Variável: descrição curta do modelo e seus hiperparâmetros.
                - Variável: descrição curta do modelo e seus hiperparâmetros.

                Variáveis selecionadas:
                - Variável: variáveis selecionadas e seus lags.
                - Variável: variáveis selecionadas e seus lags.

                Se atente a observação a seguir: {feedback}
                """
                


            case 'Avaliação': # Recebe as ações do Agente Pandas e solicita ao Agente Avaliador que aprove ou não essas ações de acordo com a etapa que foi solicitada.
                
                log = kwargs.get("log", "")
                step = kwargs.get("step", "")
                error = kwargs.get("error", "")
                tool_list = kwargs.get("tool_list", "")
                human_msg = kwargs.get("human_msg", "")

                if step == 1:
                    stepStr = "Imputação de valores faltantes"
                elif step == 2:
                    stepStr = "Previsão"
                elif step == 3:
                    stepStr = "Geração do gráfico real x previsto"
                elif step == 4:
                    stepStr = "Extração de informações geradas pelo AutoML"
                else:
                    stepStr = "Etapa não prevista"

                print("Step: " + stepStr)

                if error == 0:
                    prompt = f"""Retorne **apenas** um objeto JSON **válido**, sem blocos de código, sem texto explicativo, e sem markdown.
                                 O formato deve ser **exatamente**:
                                 {{"avaliacao": "sim", "feedback": "Você não possui feedback."}}
                                """
                    # prompt = f"""Você é um avaliador de ações executadas por um agente em várias etapas de pipeline de previsão de séries temporais.
                    #             Seu trabalho é decidir se a ação tomada pelo agente é aceitável para resolver a etapa atual.

                    #             Durante a etapa {stepStr}, o agente executou as seguintes ações: {log}.
                                
                    #             Se {stepStr} == "Previsão", considere a seguinte mensagem do usuário para entender o contexto: {human_msg}.

                    #             Retorne **apenas** um objeto JSON **válido**, sem blocos de código, sem texto explicativo, e sem markdown.
                    #             O formato deve ser **exatamente** um desses dois:

                    #             {{"avaliacao": "sim", "feedback": "Você não possui feedback."}}
                    #             ou
                    #             {{"avaliacao": "não", "feedback": "Forneça um feedback curto explicando o que está faltando ou o que foi feito incorretamente."}}

                    #             Não adicione nada antes ou depois do JSON.

                    #             """
                
                if error == 1:
                    prompt = f"""
                                Você é um corretor de erros de um agente que utiliza as seguintes tools: {tool_list}.

                                Durante a etapa {stepStr}, o agente executou as seguintes ações: {log}, 
                                mas ocorreu um erro na última ação, interrompendo a execução.

                                Sua tarefa é:
                                1. Identificar qual foi a última *Action* realizada antes do erro.
                                2. Analisar o formato da chamada feita a essa tool e verificar se está de acordo com o formato esperado.
                                3. Explicar brevemente ao agente o formato da chamada para a tool.

                                Retorne **apenas** um objeto JSON **válido**, sem blocos de código, sem texto explicativo, e sem markdown.
                                O formato deve ser **exatamente** um desses dois:

                                {{
                                "avaliacao": "não",
                                "feedback": "Descreva aqui o que o agente fez de errado e como corrigir a chamada da tool."
                                }}

                                Não adicione nada antes ou depois do JSON.
                                """



            case 'Resumo':  # Recebe as ações do Agente Pandas e solicita ao Agente Resumidor que resuma essas ações.
                # Espera receber via kwargs:
                # - steps: string com linhas "THOUGHT: ..." e "ACTION: tool(args)" (na ordem em que ocorreram)
                # - outputs: string com os outputs brutos das tools (concatenados, na mesma ordem)
                steps = kwargs.get("steps", "")
                outputs = kwargs.get("outputs", "")
                prompt = f"""
                    Você é o Agente Resumidor. Receberá dois blocos abaixo:
                    - etapas: linhas alternadas no padrão 'THOUGHT: ...' e 'ACTION: tool(args)'
                    
                    OBJETIVO
                    Escreva um relatório em pt-BR exatamente no formato:

                    Nome do etapa realizada em Português: <título curto em pt-BR>
                    Descrição da Ação: <copie/sintetize fielmente o THOUGHT correspondente, sem inventar ou 'nenhum'>
                    Ferramenta Escolhida: <nome da tool indicada em ACTION ou 'nenhuma'>
                    Resultado: <resuma o output associado a esta etapa, sem inventar>

                    REGRAS
                    - Mantenha a ordem cronológica das estapas.
                    - Não invente ferramentas, dados ou resultados.
                    - Seja conciso e técnico; evite floreios.
                    - Se não houver THOUGHT ou ACTION para alguma etapa, preencha com '—' no campo faltante.
                    - Se não houver nada para resumir, responda apenas: Sem dados para resumir.

                    DADOS:
                    {steps}
                    """.strip()
            
            case 'ResumoFinal':  # Recebe os resumos das etapas e o modelo para gerar uma resposta final para o usuário.
                resumo = kwargs.get("resumo")
                defasagens = kwargs.get("defasagens")
                target = kwargs.get("target")
                modelo_dict = kwargs.get("modelo_dict")
                previsoes = kwargs.get("previsoes")
                metricas = kwargs.get("metricas")
                
                prompt = f"""

                Você receberá as seguintes informações técnicas:
                - Etapa de pré-processamento: {resumo[0]}.
                - Etapa de previsão: {resumo[1]}.
                - Listas de variáveis/defasagens e suas importâncias utilizadas para prever a variável {target}: {defasagens}
                - O dicionário {modelo_dict} possui uma chave para cada variável. Cada variável foi prevista por um modelo e seus hiperparâmetros.
                - As previsões feita pelo modelo: {previsoes}
                - As métricas calculadas com as previsões e valores reais: {metricas}

                Sua tarefa:
                - Gerar uma explicação em tom amigável com seções claras para o usuário final, incluindo:
                - Explique o que foi feito em cada etapa de pré-processamento e previsão de forma conectada (não apenas em lista).
                - Informe ao usuário que o AutoML usado na previsão é o AutoDCE-TS.
                - Se a decomposição EMD foi aplicada, explique de maneira simples o que isso significa, em quantas IMFs foi decomposta a série e como os componentes (IMFs) foram usados.
                - Informe ao usuário que ele terá acesso ao arquivo modelo.picle. Nesse arquivo ele poderá acessar o modelo escolhido para previsão de cada variável, 
                seus hiperparâmetros acessando modelo.dict_variables['nome da variável]. As variáveis selecionadas podem ser acessadas em modelo.G_list['nome da variável].
                - Liste todas as variáveis e defasagens utilizadas na previsão da variável {target}, bem como suas repectivas importâncias para o modelo. Ao invés de dizer "nome_variável 1 (valor)", 
                diga nome_variável no tempo t-1: valor. Descarte a variável cuja importancia é -1.
                - Liste o modelo usado para previsão da variável {target} e seus hiperparâmetros.
                - Informe as métricas calculadas.
                - Explique ao usuário que ele terá acesso a duas imagens: (grafico.jpg) gráfico de comparação dos valores reais e previstos e (grafo.jpg) grafo causal com as variáveis selecionadas e 
                  suas respectivas importâncias para o modelo.
                - Informe ao usuário as previsões feitas pelo modelo como uma lista.
                - Use um texto fluido, como se estivesse conversando diretamente com o usuário, sem apenas repetir os dados recebidos.

                """
            
            case 'Memoria':
                user_msg = kwargs.get("user_msg", "")
                
                prompt = f""""  Você é o Agente de Memória em um sistema multiagente de análise de séries temporais. Considere a seguinte mensagem do usuário: {user_msg}. Você deve determinar se a mensagem se refere a:
                -execuções passadas (se o usário quer relembrar, consultar ou questionar algo que o sistema já fez anteriormente) OU
                -se a mensagem se refere a uma nova execução de previsão ou análise.
                Retorne *apenas* um objeto JSON *válido*, sem blocos de código, sem texto explicativo, e sem markdown.
                O formato deve ser *exatamente* um destes dois:

                {{"interpretacao": "passado"}} ou {{"interpretacao": "nova_execucao"}}

                Não adicione nada antes ou depois do JSON."""

        return prompt
