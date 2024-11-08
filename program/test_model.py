import os

# Configurações para reduzir os logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import csv  # Importação adicionada para manipulação de CSV

# Definições de hiperparâmetros
IMAGE_WIDTH = 30  # Largura das imagens após redimensionamento
IMAGE_HEIGHT = 30  # Altura das imagens após redimensionamento
N = 43  # Número de categorias/classes de sinais de trânsito

"""
Este script permite testar o modelo treinado para classificação de sinais de trânsito.
O usuário pode selecionar diferentes configurações de diretórios através de um menu interativo.
O script carrega o modelo treinado, processa as imagens de teste e realiza as previsões,
salvando os resultados em arquivos CSV individuais para cada classe na pasta 'result'.
"""

# Dicionário de mapeamento das classes para os nomes dos sinais de trânsito
classes = {
    0: "Limite de Velocidade (20km/h)",
    1: "Limite de Velocidade (30km/h)",
    2: "Limite de Velocidade (50km/h)",
    3: "Limite de Velocidade (60km/h)",
    4: "Limite de Velocidade (70km/h)",
    5: "Limite de Velocidade (80km/h)",
    6: "Fim do Limite de Velocidade (80km/h)",
    7: "Limite de Velocidade (100km/h)",
    8: "Limite de Velocidade (120km/h)",
    9: "Proibido Ultrapassar",
    10: "Proibido Ultrapassar Veículos acima de 3.5t",
    11: "Interseção com Prioridade",
    12: "Estrada Principal",
    13: "Dê a Preferência",
    14: "Pare",
    15: "Trânsito Proibido",
    16: "Caminhões Proibidos",
    17: "Entrada Proibida",
    18: "Perigo",
    19: "Curva Perigosa à Esquerda",
    20: "Curva Perigosa à Direita",
    21: "Dupla Curva",
    22: "Desnível na Pista",
    23: "Pista Escorregadia",
    24: "Estreitamento de Pista",
    25: "Obras",
    26: "Semáforo à Frente",
    27: "Pedestres",
    28: "Crianças",
    29: "Ciclistas",
    30: "Perigo de Neve/Gelo",
    31: "Animais Selvagens",
    32: "Fim de Todas as Restrições",
    33: "Vire à Direita",
    34: "Vire à Esquerda",
    35: "Siga em Frente",
    36: "Em Frente ou à Direita",
    37: "Em Frente ou à Esquerda",
    38: "Mantenha-se à Direita",
    39: "Mantenha-se à Esquerda",
    40: "Rotatória Obrigatória",
    41: "Fim da Proibição de Ultrapassar",
    42: "Fim da Proibição de Ultrapassar Caminhões"
}


def mostrar_menu():
    """
    Exibe um menu interativo para o usuário selecionar a configuração desejada.

    As opções incluem diferentes diretórios de dados e locais para carregar o modelo.

    Retorna:
        tuple: Uma tupla contendo:
            - model_filename (str): Caminho para o modelo treinado.
            - data_directory (str): Caminho para o diretório de dados.
            - config_name (str): Nome da configuração selecionada.
    """
    print("Selecione a configuração:")
    print("1. desktop_vini")
    print("2. laptop_vini")

    while True:
        escolha = input("Digite sua escolha (1 ou 2): ")
        if escolha == '1':
            model_filename = r"C:\Users\Pichau\Desktop\ti327v-projeto4-equipe4\program\saved_model\my_model.keras"
            data_directory = r"C:\Users\Pichau\Desktop\ti327v-projeto4-equipe4\program\gtsrb"
            return model_filename, data_directory, 'desktop_vini'
        elif escolha == '2':
            model_filename = r"C:\Users\vinic\OneDrive\Área de Trabalho\ti327v-projeto4-equipe4\program\saved_model\my_model.keras"
            data_directory = r"C:\Users\vinic\OneDrive\Área de Trabalho\ti327v-projeto4-equipe4\program\gtsrb"
            return model_filename, data_directory, 'laptop_vini'
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")


def carregar_imagem(image_path):
    """
    Carrega e preprocessa a imagem para a previsão.

    Parâmetros:
        image_path (str): Caminho para a imagem a ser processada.

    Retorna:
        np.ndarray: Array NumPy representando a imagem processada.
    """
    try:
        img = Image.open(image_path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img.convert('RGB')  # Garante que a imagem tenha 3 canais
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normaliza os valores dos pixels
        img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão de batch
        return img_array
    except Exception as e:
        print(f"Erro ao carregar a imagem {image_path}: {e}")
        return None


def prever_imagem(model, img_array):
    """
    Realiza a previsão da classe da imagem usando o modelo fornecido.

    Parâmetros:
        model (tf.keras.Model): Modelo treinado para realizar a previsão.
        img_array (np.ndarray): Array NumPy representando a imagem processada.

    Retorna:
        tuple: Uma tupla contendo:
            - predicted_class (int): Classe prevista para a imagem.
            - confidence (float): Confiança da previsão.
    """
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    return predicted_class, confidence


def exibir_resultado(image_path, predicted_class, confidence):
    """
    Exibe a imagem com a classe prevista e a confiança.

    Parâmetros:
        image_path (str): Caminho para a imagem original.
        predicted_class (int): Classe prevista para a imagem.
        confidence (float): Confiança da previsão.
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Previsto: {classes[predicted_class]} ({confidence * 100:.2f}%)")
    plt.show()


if __name__ == '__main__':
    # Mostrar o menu e obter as configurações selecionadas
    model_filename, data_directory, config_name = mostrar_menu()
    print(f"\nConfiguração selecionada: {config_name}")
    print(f"Caminho do modelo: {model_filename}")
    print(f"Caminho dos dados: {data_directory}\n")

    # Verifica se o modelo existe
    if not os.path.exists(model_filename):
        sys.exit(f"Modelo não encontrado em: {model_filename}")

    # Carrega o modelo treinado
    model = tf.keras.models.load_model(model_filename)
    print("Modelo carregado com sucesso.\n")

    # Pergunta ao usuário se deseja testar todas as imagens ou imagens de uma classe específica
    print("Deseja testar todas as imagens em todos os diretórios ou todas as imagens em um diretório específico?")
    print("1. Testar todas as imagens em todos os diretórios")
    print("2. Testar todas as imagens em um diretório específico")

    # Configura o diretório de resultados
    result_dir = os.path.join(os.getcwd(), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    while True:
        escolha_teste = input("Digite sua escolha (1 ou 2): ")
        if escolha_teste == '1':
            # Testar todas as imagens em todos os diretórios
            for class_num in range(N):
                class_dir = os.path.join(data_directory, str(class_num))
                if not os.path.isdir(class_dir):
                    print(f"Diretório não encontrado para a classe {class_num}: {class_dir}")
                    continue
                print(f"\nProcessando imagens da classe {class_num} - {classes[class_num]}:")

                # Define o arquivo CSV para a classe atual
                output_file = os.path.join(result_dir, f"{class_num}-result.csv")
                with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Filename", "Actual_Class", "Predicted_Class", "Class_Name", "Confidence (%)"])

                    for filename in os.listdir(class_dir):
                        filepath = os.path.join(class_dir, filename)
                        if not os.path.isfile(filepath):
                            continue
                        img_array = carregar_imagem(filepath)
                        if img_array is not None:
                            predicted_class, confidence = prever_imagem(model, img_array)
                            # exibir_resultado(filepath, predicted_class, confidence)  # Opcional
                            print(
                                f"{filename}: Classe real {class_num} - {classes[class_num]} | Classe predita {predicted_class} - {classes[predicted_class]} (Confiança: {confidence * 100:.2f}%)")
                            # Escreve os resultados no arquivo CSV
                            writer.writerow([filename, class_num, predicted_class, classes[predicted_class],
                                             f"{confidence * 100:.2f}"])
                print(f"Resultados da classe {class_num} salvos em {output_file}")
            break
        elif escolha_teste == '2':
            # Solicita ao usuário o número da classe (0 a 42)
            while True:
                class_input = input(f"Digite o número da classe (0 a {N - 1}): ")
                if class_input.isdigit() and 0 <= int(class_input) < N:
                    class_num = int(class_input)
                    break
                else:
                    print(f"Entrada inválida. Por favor, digite um número entre 0 e {N - 1}.")

            class_dir = os.path.join(data_directory, str(class_num))
            if not os.path.isdir(class_dir):
                sys.exit(f"Diretório não encontrado para a classe {class_num}: {class_dir}")

            print(f"\nProcessando imagens da classe {class_num} - {classes[class_num]}:")

            # Define o arquivo CSV para a classe selecionada
            output_file = os.path.join(result_dir, f"{class_num}-result.csv")
            with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Filename", "Actual_Class", "Predicted_Class", "Class_Name", "Confidence (%)"])

                for filename in os.listdir(class_dir):
                    filepath = os.path.join(class_dir, filename)
                    if not os.path.isfile(filepath):
                        continue
                    img_array = carregar_imagem(filepath)
                    if img_array is not None:
                        predicted_class, confidence = prever_imagem(model, img_array)
                        # exibir_resultado(filepath, predicted_class, confidence)  # Opcional
                        print(
                            f"{filename}: Classe real {class_num} - {classes[class_num]} | Classe predita {predicted_class} - {classes[predicted_class]} (Confiança: {confidence * 100:.2f}%)")
                        # Escreve os resultados no arquivo CSV
                        writer.writerow(
                            [filename, class_num, predicted_class, classes[predicted_class], f"{confidence * 100:.2f}"])
            print(f"Resultados da classe {class_num} salvos em {output_file}")
            break
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

    print("\nProcessamento concluído. Os resultados foram salvos na pasta 'result'.")
