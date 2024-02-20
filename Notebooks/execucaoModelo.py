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
            print("Nenhuma matéria encontrada com esse numero")
    
    return materia

while quebrarWhile == False:
    print("* Menu *".center(espacamento, '-'))
    print("1 - Fazer pergunta ".ljust(espacamento, '-'))
    print("0 - Sair ".ljust(espacamento, '-'), flush=True)
    opcao = int(input())

    if opcao == 0: quebrarWhile = True
    elif opcao == 1:
        print(" Digite sua pergunta ".center(espacamento, '-'), flush=True)
        texto = input()

        print("".center(espacamento, '-'))
        print(" Agora nos diga o numero da matéria ".center(espacamento, '-'), flush=True)
        indice_materia = pegarMateria()

        tempo_inicio = time.perf_counter()
        resultado = classificador(texto)[0]
        tempo_fim = time.perf_counter()

        print("".center(espacamento, "-"))
        print(f"Matéria original: {materias[indice_materia]}")
        print(f"Matéria prevista: {resultado['label']}")
        print(f"Tempo de resposta: {tempo_fim - tempo_inicio} segundos")

        if resultado['label'] == materias[indice_materia]:
            print("A IA acertou a matéria!")
        else:
            print("A IA errou a matéria, talvez precise de um melhor treinamento")
        

    else: print(f"Opção {opcao} não existe no Menu")
    print("\n")

print("Fim da execução!")