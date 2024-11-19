import os

# Configurações para reduzir os logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Definições de hiperparâmetros
EPOCHS = 15              # Número de épocas para o treinamento
IMAGE_WIDTH = 30         # Largura das imagens após redimensionamento
IMAGE_HEIGHT = 30        # Altura das imagens após redimensionamento
N = 43                   # Número de categorias/classes de sinais de trânsito
TEST_SIZE = 0.40         # Proporção de dados reservados para teste

# Dicionário de mapeamento das classes para os nomes dos sinais de trânsito
classes_dict = {
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

"""
Este script treina uma rede neural convolucional para classificar sinais de trânsito usando o conjunto de dados GTSRB (German Traffic Sign Recognition Benchmark).
O script detecta automaticamente o caminho base da aplicação e configura os diretórios de dados e modelo de acordo.
O modelo treinado é salvo no formato nativo Keras para uso posterior.
Adicionalmente, gera gráficos de perda e acurácia durante o treinamento, bem como uma matriz de confusão após os testes, salvando-os na pasta 'results'.
"""


def load_data(directory):
    """
    Carrega as imagens e os rótulos das classes a partir do diretório especificado.

    Cada subdiretório dentro de 'directory' deve ser nomeado de 0 a N-1, correspondendo a cada classe,
    e conter as imagens da classe respectiva.

    Parâmetros:
        directory (Path): O caminho para o diretório contendo as imagens organizadas em subpastas por classe.

    Retorna:
        tuple: Uma tupla contendo duas listas:
            - images (list): Lista de arrays NumPy representando as imagens carregadas.
            - labels (list): Lista de inteiros representando as classes correspondentes às imagens.
    """
    images = []
    labels = []
    total_images = 0

    # Itera por cada categoria/classe
    for label in range(N):
        label_dir = directory / str(label)
        if not label_dir.is_dir():
            print(f"Diretório não encontrado para a categoria {label}: {label_dir}")
            continue  # Pula se o diretório não existir

        num_images_in_label = 0

        # Itera por cada arquivo de imagem na categoria
        for filename in os.listdir(label_dir):
            filepath = label_dir / filename

            # Verifica se é um arquivo
            if not filepath.is_file():
                continue

            try:
                img = Image.open(filepath)  # Carrega a imagem usando PIL
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Redimensiona a imagem para o tamanho padrão
                img = img.convert('RGB')  # Converte a imagem para RGB (caso esteja em outro modo)
                img_array = np.array(img)  # Converte a imagem em um array NumPy

                # Adiciona a imagem e o rótulo às listas
                images.append(img_array)
                labels.append(label)

                num_images_in_label += 1
                total_images += 1
            except Exception as e:
                print(f"Erro ao processar a imagem {filepath}: {e}")
                continue

        print(f"Categoria {label}: {num_images_in_label} imagens carregadas.")

    print(f"Total de imagens carregadas: {total_images}")
    return images, labels


def get_model():
    """
    Cria e compila uma rede neural convolucional para classificação de imagens de sinais de trânsito.

    A arquitetura do modelo inclui:
    - Camadas convolucionais para extração de características.
    - Camadas de pooling para redução de dimensionalidade.
    - Camada totalmente conectada com dropout para evitar overfitting.
    - Camada de saída com ativação softmax para classificação nas N classes.

    Retorna:
        model (tf.keras.Model): O modelo compilado pronto para treinamento.
    """
    model = tf.keras.models.Sequential([
        # Camada de entrada com a forma das imagens
        tf.keras.layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),

        # Primeira camada convolucional seguida de pooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Segunda camada convolucional seguida de pooling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Terceira camada convolucional
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten para converter os mapas de características em um vetor
        tf.keras.layers.Flatten(),

        # Camada totalmente conectada com dropout
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Camada de saída com ativação softmax para classificação
        tf.keras.layers.Dense(N, activation='softmax')
    ])

    # Compila o modelo com otimizador, função de perda e métricas
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


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
    # Aqui você pode adicionar lógica adicional para diferentes ambientes se necessário
    # Por exemplo, se precisar de configurações diferentes dentro do mesmo projeto

    # Para simplificar, assumimos uma única configuração baseada no caminho base
    data_directory = base_dir / "program" / "gtsrb"
    model_filename = base_dir / "program" / "saved_model" / "my_model.keras"
    config_name = 'default_config'

    return data_directory, model_filename, config_name


def plot_training_history(history, results_directory):
    """
    Plota e salva os gráficos de perda e acurácia do treinamento e validação.

    Parâmetros:
        history: Objeto History retornado por model.fit().
        results_directory (Path): Diretório onde os gráficos serão salvos.
    """
    # Plot da perda (loss)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
    loss_plot_path = results_directory / "training" / 'training_validation_loss.png'
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Gráfico de perda salvo em: {loss_plot_path}")

    # Plot da acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = results_directory / "training" / 'training_validation_accuracy.png'
    accuracy_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"Gráfico de acurácia salvo em: {accuracy_plot_path}")


def plot_confusion_matrix(model, x_test, y_test, results_directory):
    """
    Calcula e plota a matriz de confusão nos dados de teste.

    Parâmetros:
        model: Modelo treinado.
        x_test: Dados de teste (imagens).
        y_test: Rótulos de teste (one-hot encoded).
        results_directory (Path): Diretório onde o gráfico será salvo.
    """
    # Realiza as previsões nos dados de teste
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[classes_dict[i] for i in range(N)],
                yticklabels=[classes_dict[i] for i in range(N)])
    plt.title('Matriz de Confusão nos Dados de Teste')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    confusion_matrix_path = results_directory / "training" / 'test_confusion_matrix.png'
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Matriz de confusão salva em: {confusion_matrix_path}")


def save_classification_report(model, x_test, y_test, results_directory, classes_dict):
    """
    Salva o relatório de classificação nos dados de teste.

    Parâmetros:
        model: Modelo treinado.
        x_test: Dados de teste (imagens).
        y_test: Rótulos de teste (one-hot encoded).
        results_directory (Path): Diretório onde o relatório será salvo.
        classes_dict (dict): Dicionário mapeando os índices das classes para seus nomes.
    """
    from sklearn.metrics import classification_report

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    target_names = [classes_dict[i] for i in range(N)]
    report = classification_report(y_true, y_pred, target_names=target_names)

    report_path = results_directory / "training" / 'classification_report_training.txt'
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
    print(f"Caminho para salvar o modelo: {model_filename}\n")

    # Configurar o caminho para a pasta 'results'
    results_directory = base_dir / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    # Carrega os dados a partir do diretório especificado
    images, labels = load_data(data_directory)

    # Verifica se alguma imagem foi carregada
    if len(images) == 0 or len(labels) == 0:
        sys.exit("Nenhuma imagem foi carregada. Verifique o diretório de dados e o formato das imagens.")

    # Converte as listas em arrays NumPy para processamento eficiente
    images = np.array(images)
    labels = np.array(labels)

    # Normaliza os valores dos pixels para o intervalo [0, 1]
    images = images / 255.0

    # Converte os rótulos em codificação one-hot
    labels = tf.keras.utils.to_categorical(labels, N)

    # Divide os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=42
    )

    # Obtém o modelo compilado
    model = get_model()

    # Treina o modelo nos dados de treinamento e captura o histórico
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Avalia o desempenho do modelo nos dados de teste
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nDesempenho no conjunto de teste - Perda: {test_loss}, Acurácia: {test_accuracy}")

    # Plota e salva os gráficos de treinamento
    plot_training_history(history, results_directory)

    # Plota e salva a matriz de confusão
    plot_confusion_matrix(model, x_test, y_test, results_directory)

    # Salva o relatório de classificação
    # classes_dict já está definido globalmente
    save_classification_report(model, x_test, y_test, results_directory, classes_dict)

    # Garante que o diretório para salvar o modelo existe
    model_directory = model_filename.parent
    model_directory.mkdir(parents=True, exist_ok=True)

    # Salva o modelo treinado no formato nativo Keras
    model.save(model_filename)
    print(f"Modelo salvo em {model_filename}")


if __name__ == '__main__':
    main()
