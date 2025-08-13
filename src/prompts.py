class Prompt:

    def get_prompt(name: str, **kwargs) -> str:
        match name:
            case 'Etapa 1': # Imputação de valores faltantes: 
                prompt = f"...."


            case 'Etapa 2': # Decomposição:
                prompt = f"...."



            case 'Etapa 3': # Previsão:
                prompt = f"...."


            case 'Etapa 4': # Explicação:
                prompt = f"...."


            case 'Avaliação': # Recebe as ações do Agente Pandas e solicita ao Agente Avaliador que aprove ou não essas ações de acordo com a etapa que foi solicitada.
                prompt = f"...."



            case 'Resumo': # Recebe as ações do Agente Pandas e solicita ao Agente Resumidor que resuma essas ações. 
                prompt = "...."
        

        return prompt
