from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline
import pandas as pd
import time

path_modelo = "../Model/Modelo"

modelo = BertForSequenceClassification.from_pretrained(path_modelo)
tokenizer = BertTokenizerFast.from_pretrained(path_modelo)
classificador = pipeline("text-classification", model=modelo, tokenizer=tokenizer)

espacamento = 60

df_perguntas = pd.read_csv("../Dados/dados.csv")
materias = df_perguntas['materia'].unique().tolist()
del df_perguntas

quebrarWhile = False
opcao = None

def pegarMateria():
    materia = None
    
    while materia == None:
        for i in range(len(materias)):
            print(f"{i} - {materias[i]}".rjust(espacamento), flush=True)
        
        materia = int(input())

        if (materia >= len(materias) and materia < 0):
            materia = None
            print("\033[41mNenhuma matéria encontrada com esse numero\033[0m")
    
    return materia

while quebrarWhile == False:

    # Menu de interacao com o usuario
    print("* Menu *".center(espacamento, '-'))
    print("1 - Fazer pergunta ".ljust(espacamento, '-'))
    print("0 - Sair ".ljust(espacamento, '-'), flush=True)
    opcao = int(input())

    # Opcao de saida quebra o while
    if opcao == 0: quebrarWhile = True

    # Opcao para fazer uma pergunta e predize-la
    elif opcao == 1:
        print(" Digite sua pergunta ".center(espacamento, '-'), flush=True)
        texto = input()
        print(f"Pergunta: {texto}")
        print("".center(espacamento, '-'))
        print(" Agora nos diga o numero da matéria ".center(espacamento, '-'), flush=True)

        # Chama uma funcao para imprimir as materias possiveis dinamicamente
        indice_materia = pegarMateria()
        
        # Registra o tempo de inicio para o calculo do tempo gasto pela classificacao
        tempo_inicio = time.perf_counter()

        # Classifica o texto e nos retorna a label prevista e seu score
        resultado = classificador(texto)[0]

        # Registra o tempo ao fim da classificacao para calculo do tempo de resposta
        tempo_fim = time.perf_counter()

        print("".center(espacamento, "-"))
        print(f"Matéria original: {materias[indice_materia]}")
        print(f"Matéria prevista: {resultado['label']}")
        print(f"\033[33mPontuação da previsão: {resultado['score']}\033[0m")
        print(f"\033[36mTempo de resposta: {tempo_fim - tempo_inicio} segundos\033[0m")

        if resultado['label'] == materias[indice_materia]:
            print("\033[32mA IA acertou a matéria!\033[0m")
        else:
            print("\033[31mA IA errou a matéria, talvez precise de um melhor treinamento\033[0m")
        

    else: print(f"Opção {opcao} não existe no Menu")
    print("\n")

print("Fim da execução!")