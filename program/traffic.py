import os

# Configurações para reduzir os logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Definições de hiperparâmetros
EPOCHS = 10  # Número de épocas para o treinamento
IMAGE_WIDTH = 30  # Largura das imagens após redimensionamento
IMAGE_HEIGHT = 30  # Altura das imagens após redimensionamento
N = 43  # Número de categorias/classes de sinais de trânsito
TEST_SIZE = 0.40  # Proporção de dados reservados para teste

"""
Este script treina uma rede neural convolucional para classificar sinais de trânsito usando o conjunto de dados GTSRB (German Traffic Sign Recognition Benchmark).
O usuário pode selecionar diferentes configurações de diretórios para os dados e o modelo através de um menu interativo.
O modelo treinado é salvo no formato nativo Keras para uso posterior.
"""


def load_data(directory):
    """
    Carrega as imagens e os rótulos das classes a partir do diretório especificado.

    Cada subdiretório dentro de 'directory' deve ser nomeado de 0 a N-1, correspondendo a cada classe,
    e conter as imagens da classe respectiva.

    Parâmetros:
        directory (str): O caminho para o diretório contendo as imagens organizadas em subpastas por classe.

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
        label_dir = os.path.join(directory, str(label))
        if not os.path.isdir(label_dir):
            print(f"Diretório não encontrado para a categoria {label}: {label_dir}")
            continue  # Pula se o diretório não existir

        num_images_in_label = 0

        # Itera por cada arquivo de imagem na categoria
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)

            # Verifica se é um arquivo
            if not os.path.isfile(filepath):
                continue

            try:
                # Carrega a imagem usando PIL
                img = Image.open(filepath)

                # Redimensiona a imagem para o tamanho padrão
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

                # Converte a imagem para RGB (caso esteja em outro modo)
                img = img.convert('RGB')

                # Converte a imagem em um array NumPy
                img_array = np.array(img)

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


def mostrar_menu():
    """
    Exibe um menu interativo para o usuário selecionar a configuração desejada.

    As opções incluem diferentes diretórios de dados e locais para salvar o modelo.

    Retorna:
        tuple: Uma tupla contendo:
            - data_directory (str): Caminho para o diretório de dados selecionado.
            - model_filename (str): Caminho para salvar o modelo treinado.
            - config_name (str): Nome da configuração selecionada.
    """
    print("Selecione a configuração:")
    print("1. desktop_vini")
    print("2. laptop_vini")

    while True:
        escolha = input("Digite sua escolha (1 ou 2): ")
        if escolha == '1':
            data_directory = r"C:\Users\Pichau\Desktop\ti327v-projeto4-equipe4\program\gtsrb"
            model_filename = r"C:\Users\Pichau\Desktop\ti327v-projeto4-equipe4\program\saved_model\my_model.keras"
            return data_directory, model_filename, 'desktop_vini'
        elif escolha == '2':
            data_directory = r"C:\Users\vinic\OneDrive\Área de Trabalho\ti327v-projeto4-equipe4\program\gtsrb"
            model_filename = r"C:\Users\vinic\OneDrive\Área de Trabalho\ti327v-projeto4-equipe4\program\saved_model\my_model.keras"
            return data_directory, model_filename, 'laptop_vini'
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")


if __name__ == '__main__':
    # Mostrar o menu e obter as configurações selecionadas
    data_directory, model_filename, config_name = mostrar_menu()
    print(f"\nConfiguração selecionada: {config_name}")
    print(f"Diretório de dados: {data_directory}")
    print(f"Caminho para salvar o modelo: {model_filename}\n")

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
        images, labels, test_size=TEST_SIZE
    )

    # Obtém o modelo compilado
    model = get_model()

    # Treina o modelo nos dados de treinamento
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Avalia o desempenho do modelo nos dados de teste
    model.evaluate(x_test, y_test, verbose=2)

    # Garante que o diretório para salvar o modelo existe
    model_directory = os.path.dirname(model_filename)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Salva o modelo treinado no formato nativo Keras
    model.save(model_filename)
    print(f"Modelo salvo em {model_filename}")
