import os

# Configurações para reduzir os logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import csv  # Para manipulação de CSV
from sklearn.metrics import confusion_matrix, classification_report

# Definições de hiperparâmetros
IMAGE_WIDTH = 30  # Largura das imagens após redimensionamento
IMAGE_HEIGHT = 30  # Altura das imagens após redimensionamento
N = 43  # Número de categorias/classes de sinais de trânsito

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


def detect_base_dir(target_folder_name="9-cnn"):
    """
    Detecta automaticamente o caminho base da aplicação percorrendo o diretório pai até encontrar a pasta alvo.

    Parâmetros:
        target_folder_name (str): O nome da pasta alvo que representa o caminho base da aplicação.

    Retorna:
        Path: O caminho para a pasta alvo.

    Se a pasta alvo não for encontrada, o script será encerrado com uma mensagem de erro.
    """
    current_dir = Path(__file__).resolve().parent
    while True:
        if current_dir.name == target_folder_name:
            return current_dir
        if current_dir.parent == current_dir:
            # Chegou ao diretório raiz sem encontrar a pasta alvo
            break
        current_dir = current_dir.parent

    print(f"Pasta base '{target_folder_name}' não encontrada. Verifique a estrutura de diretórios.")
    sys.exit(1)


def config_paths(base_dir):
    """
    Configura os caminhos para data_directory e model_filename relativamente ao caminho base.

    Parâmetros:
        base_dir (Path): O caminho base da aplicação.

    Retorna:
        tuple: Uma tupla contendo:
            - data_directory (Path): Caminho para o diretório de dados.
            - model_filename (Path): Caminho para salvar o modelo treinado.
            - config_name (str): Nome da configuração (baseado no ambiente, se necessário).
    """
    # Ajuste os caminhos conforme a estrutura do seu projeto
    data_directory = base_dir / "program" / "gtsrb"
    model_filename = base_dir / "program" / "saved_model" / "my_model.keras"
    config_name = 'default_config'

    return data_directory, model_filename, config_name


def load_image(image_path):
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


def predict_image(model, img_array):
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


def display_result(image_path, predicted_class, confidence):
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


def plot_confusion_matrix_func(all_true, all_predicted, results_directory):
    """
    Calcula e plota a matriz de confusão com todas as previsões e rótulos verdadeiros.

    Parâmetros:
        all_true (list): Lista de rótulos verdadeiros.
        all_predicted (list): Lista de classes previstas.
        results_directory (Path): Diretório onde o gráfico será salvo.
    """
    cm = confusion_matrix(all_true, all_predicted, labels=range(N))
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[classes[i] for i in range(N)],
                yticklabels=[classes[i] for i in range(N)])
    plt.title('Matriz de Confusão nos Dados de Teste')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Caminho relativo
    confusion_matrix_path = results_directory / "test" / "confusion_matrix_test.png"
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {confusion_matrix_path}")


def save_classification_report_func(all_true, all_predicted, results_directory):
    """
    Salva o relatório de classificação com todas as previsões e rótulos verdadeiros.

    Parâmetros:
        all_true (list): Lista de rótulos verdadeiros.
        all_predicted (list): Lista de classes previstas.
        results_directory (Path): Diretório onde o relatório será salvo.
    """
    target_names = [classes[i] for i in range(N)]
    report = classification_report(
        all_true,
        all_predicted,
        labels=range(N),  # Especifica todas as 43 classes
        target_names=target_names,
        zero_division=0  # Opcional: evita erros de divisão por zero
    )

    # Caminho relativo
    report_path = results_directory / "test" / "test_classification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Relatório de classificação salvo em: {report_path}")


def main():
    # Detectar o caminho base da aplicação
    base_dir = detect_base_dir()
    print(f"Caminho base detectado: {base_dir}")

    # Configurar os caminhos relativos
    data_directory, model_filename, config_name = config_paths(base_dir)
    print(f"\nConfiguração selecionada: {config_name}")
    print(f"Diretório de dados: {data_directory}")
    print(f"Caminho do modelo: {model_filename}\n")

    # Configurar o caminho para a pasta 'results'
    results_directory = base_dir / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    # Detectar o caminho base da aplicação
    base_dir = detect_base_dir()
    print(f"Caminho base detectado: {base_dir}")

    # Configurar os caminhos relativos
    data_directory, model_filename, config_name = config_paths(base_dir)
    print(f"\nConfiguração selecionada: {config_name}")
    print(f"Diretório de dados: {data_directory}")
    print(f"Caminho do modelo: {model_filename}\n")

    # Configurar o caminho para a pasta 'results'
    results_directory = base_dir / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    # Verifica se o modelo existe
    if not model_filename.exists():
        sys.exit(f"Modelo não encontrado em: {model_filename}")

    # Carrega o modelo treinado
    model = tf.keras.models.load_model(model_filename)
    print("Modelo carregado com sucesso.\n")

    # Pergunta ao usuário se deseja testar todas as imagens ou imagens de uma classe específica
    print("Deseja testar todas as imagens em todos os diretórios ou todas as imagens em um diretório específico?")
    print("1. Testar todas as imagens em todos os diretórios")
    print("2. Testar todas as imagens em um diretório específico")

    # Listas para acumular todas as previsões e rótulos verdadeiros
    all_predicted = []
    all_true = []

    while True:
        escolha_teste = input("Digite sua escolha (1 ou 2): ")
        if escolha_teste == '1':
            # Testar todas as imagens em todos os diretórios
            for class_num in range(N):
                class_dir = data_directory / str(class_num)
                if not class_dir.is_dir():
                    print(f"Diretório não encontrado para a classe {class_num}: {class_dir}")
                    continue
                print(f"\nProcessando imagens da classe {class_num} - {classes[class_num]}:")

                # Define o arquivo CSV para a classe atual
                output_file = results_directory / "test" / f"{class_num}-result.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Filename", "Actual_Class", "Predicted_Class", "Class_Name", "Confidence (%)"])

                    for filename in os.listdir(class_dir):
                        filepath = class_dir / filename
                        if not filepath.is_file():
                            continue
                        img_array = load_image(filepath)
                        if img_array is not None:
                            predicted_class, confidence = predict_image(model, img_array)
                            # display_result(filepath, predicted_class, confidence)  # Opcional
                            print(
                                f"{filename}: Classe real {class_num} - {classes[class_num]} | Classe predita {predicted_class} - {classes[predicted_class]} (Confiança: {confidence * 100:.2f}%)")
                            # Escreve os resultados no arquivo CSV
                            writer.writerow([filename, class_num, predicted_class, classes[predicted_class],
                                             f"{confidence * 100:.2f}"])
                            # Acumula para a matriz de confusão
                            all_predicted.append(predicted_class)
                            all_true.append(class_num)
            # Após testar todas as classes, gerar a matriz de confusão e o relatório
            print("\nGerando matriz de confusão e relatório de classificação...")
            plot_confusion_matrix_func(all_true, all_predicted, results_directory)
            save_classification_report_func(all_true, all_predicted, results_directory)
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

            class_dir = data_directory / str(class_num)
            if not class_dir.is_dir():
                sys.exit(f"Diretório não encontrado para a classe {class_num}: {class_dir}")

            print(f"\nProcessando imagens da classe {class_num} - {classes[class_num]}:")

            # Define o arquivo CSV para a classe selecionada
            output_file = results_directory / "test" / f"{class_num}-result.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Filename", "Actual_Class", "Predicted_Class", "Class_Name", "Confidence (%)"])

                # Listas para acumular previsões e rótulos verdadeiros da classe específica
                class_predicted = []
                class_true = []

                for filename in os.listdir(class_dir):
                    filepath = class_dir / filename
                    if not filepath.is_file():
                        continue
                    img_array = load_image(filepath)
                    if img_array is not None:
                        predicted_class, confidence = predict_image(model, img_array)
                        # display_result(filepath, predicted_class, confidence)  # Opcional
                        print(
                            f"{filename}: Classe real {class_num} - {classes[class_num]} | Classe predita {predicted_class} - {classes[predicted_class]} (Confiança: {confidence * 100:.2f}%)")
                        # Escreve os resultados no arquivo CSV
                        writer.writerow(
                            [filename, class_num, predicted_class, classes[predicted_class], f"{confidence * 100:.2f}"])
                        # Acumula para a matriz de confusão
                        class_predicted.append(predicted_class)
                        class_true.append(class_num)

            # Adiciona as previsões e verdadeiros à lista geral
            all_predicted.extend(class_predicted)
            all_true.extend(class_true)

            # Após testar a classe selecionada, gerar a matriz de confusão e o relatório
            print("\nGerando matriz de confusão e relatório de classificação...")
            plot_confusion_matrix_func(all_true, all_predicted, results_directory)
            save_classification_report_func(all_true, all_predicted, results_directory)
            break
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

    print("\nProcessamento concluído. Os resultados foram salvos na pasta 'results'.")


if __name__ == '__main__':
    main()
