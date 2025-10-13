
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
                1. Defina automaticamente, sempre com uma breve justificativa:
                - O número máximo de defasagens (`max_lags`), escolhendo entre 5 e 50 para equilibrar precisão e custo computacional.
                - Se a decomposição (`decomposition`) deve ser usada ou não.
                2. Utilize a tool `testar_estacionariedade` caso seja necessário decidir sobre a estacionariedade.

                Se atente a observação a seguir: {feedback}
                """
                   

            case 'Etapa 3': # Visualização real x previsto

                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente especializado em visualização de séries temporais.

                Regras:
                1. Foque apenas na tarefa de visualização de dados.
                2. O dataframe contém colunas com valores **reais** e **previstos**.
                3. Gere um gráfico comparando os valores reais e previstos.
                4. Retorne a figura **em Base64** pronta para uso, sem explicações ou texto adicional.
                
                Se atente a observação a seguir: {feedback}
                """


            case 'Etapa 4': # Visualização grafo causal:
                modelo = kwargs.get("modelo", "")
                feedback = kwargs.get("feedback", "")

                prompt = f"""
                Você é um assistente de análise de modelos.

                Regras:
                1. Use exclusivamente a tool `desenhar_grafo`.
                2. Gere o **grafo de importância das variáveis** do modelo {modelo}.
                3. Retorne a figura **em Base64** pronta para uso, sem explicações ou texto adicional.

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
                    stepStr = "Visualização real x previsto"
                elif step == 4:
                    stepStr = "Visualização grafo causal"
                else:
                    stepStr = "Etapa não prevista"

                if error == 0:
                    prompt = f"""Você é um avaliador de ações executadas por um agente em várias etapas de pipeline de previsão de séries temporais.
                                Seu trabalho é decidir se a ação tomada pelo agente é aceitável para resolver a etapa atual.

                                Durante a etapa {stepStr}, o agente executou as seguintes ações: {log}.
                                
                                Se {stepStr} == "Previsão", considere a seguinte mensagem do usuário para entender o contexto: {human_msg}.

                                Retorne **apenas** um objeto JSON **válido**, sem blocos de código, sem texto explicativo, e sem markdown.
                                O formato deve ser **exatamente** um desses dois:

                                {{"avaliacao": "sim", "feedback": "Você não possui feedback."}}
                                ou
                                {{"avaliacao": "não", "feedback": "Forneça um feedback curto explicando o que está faltando ou o que foi feito incorretamente."}}

                                Não adicione nada antes ou depois do JSON.

                                """
                
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
            
            case 'ResumoFinal':  # Recebe os resumos das etapas e o modelo para gerar uma resposta final para o usuário.
                resumo = kwargs.get("resumo")
                defasagens = kwargs.get("defasagens")
                target = kwargs.get("target")
                modelo_dict = kwargs.get("modelo_dict")
                previsoes = kwargs.get("previsoes")
                # n_estimators = kwargs.get("n_estimators")
                # min_samples_leaf = kwargs.get("min_samples_leaf")
                # max_features = kwargs.get("max_features")
                # decomposition = kwargs.get("decomposition")
                # graph = kwargs.get("graph")
                # variaveis = kwargs.get("variaveis")
                
                prompt = f"""

                Você receberá as seguintes informações técnicas:
                - Etapa de pré-processamento: {resumo[0]}.
                - Etapa de previsão: {resumo[1]}.
                - Listas de variáveis/defasagens e suas importâncias utilizadas para prever a variável {target}: {defasagens}
                - O dicionário {modelo_dict} possui uma chave para cada variável. Cada variável foi prevista por um modelo e seus hiperparâmetros.
                - As previsões feita pelo modelo: {previsoes}

                Sua tarefa:
                - Gerar uma explicação em tom amigável.
                - Explique o que foi feito em cada etapa de pré-processamento e previsão de forma conectada (não apenas em lista).
                - Informe ao usuário que o AutoML usado na previsão é o AutoDCE-TS.
                - Se a decomposição EMD foi aplicada, explique de maneira simples o que isso significa, em quantas IMFs foi decomposta a série e como os componentes (IMFs) foram usados.
                - Informe ao usuário que ele terá acesso ao arquivo modelo.picle. Nesse arquivo ele poderá acessar o modelo escolhido para previsão de cada variável, 
                seus hiperparâmetros acessando modelo.dict_variables['nome da variável]. As variáveis selecionadas podem ser acessadas em modelo.G_list['nome da variável].
                - Liste todas as variáveis e defazagens utilizadas na previsão da variável {target}, bem como suas repectivas importâncias para o modelo. Ao invés de dizer "ETO 1 (valor)", 
                diga ETO no tempo t-1: valor. Descarte a variável cuja importancia é -1.
                - Liste o modelo usado para previsão da variável {target} e seus hiperparâmetros.
                - Explique ao usuário que ele terá acesso a duas imagens: (grafico.jpg) gráfico de comparação dos valores reais e previstos e (grafo.jpg) grafo causal com as variáveis selecionadas e 
                  suas respectivas importâncias para o modelo.
                - Informe ao usuário as previsões feitas pelo modelo como uma lista.
                - Use um texto fluido, como se estivesse conversando diretamente com o usuário, sem apenas repetir os dados recebidos.

                """

        return prompt
