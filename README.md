# Projeto 4 de TI327 - Tópicos em Inteligência Artificial

## 🧑‍🎓 Integrantes
- **Anna Clara Ferraz** - 23306
- **Vinícius Dos Santos Andrade** - 22333

---

## 📝 Descrição

Este projeto consiste no desenvolvimento de uma **rede neural convolucional (CNN)** para a classificação de sinais de trânsito utilizando o conjunto de dados **GTSRB (German Traffic Sign Recognition Benchmark)**. O objetivo é treinar um modelo capaz de reconhecer e classificar imagens de sinais de trânsito em **43 categorias distintas**, contribuindo para avanços na área de **visão computacional aplicada ao trânsito e segurança viária**.

---

## 🚀 Tecnologias Utilizadas

- **Python 3**
- **TensorFlow 2**
- **NumPy**
- **Pillow (PIL)**
- **scikit-learn**

---

## 📂 Estrutura do Projeto

- `traffic.py`: Script principal para treinamento da rede neural.
- `test_model.py`: Script para testar o modelo treinado em novas imagens.
- `gtsrb/`: Diretório contendo o conjunto de dados organizado em subpastas por classe.
- `saved_model/`: Diretório onde o modelo treinado é salvo.
- `requirements.txt`: Arquivo contendo todas as dependências necessárias.
- `README.md`: Documentação do projeto.

---

## 🔧 Instalação e Execução

### 📋 Pré-requisitos
Antes de iniciar, certifique-se de que você tem o **Python 3** instalado em seu sistema. Além disso, é recomendado utilizar um ambiente virtual para gerenciar as dependências do projeto.

### 🛠️ Configurando o Ambiente

1. **Clone o Repositório ou Baixe os Arquivos do Projeto**:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Crie um Ambiente Virtual** (Opcional, mas Recomendado):
   ```bash
   python -m venv .venv
   ```

3. **Ative o Ambiente Virtual**:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Instale as Dependências** Usando o `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🗄️ Preparação do Conjunto de Dados

1. **Download do GTSRB Dataset**:
   - Baixe o conjunto de dados GTSRB e extraia-o no diretório `gtsrb/`.

2. **Organização do Conjunto de Dados**:
   - Certifique-se de que os dados estão organizados em subpastas por classe.

---

## 🚀 Treinamento do Modelo

- Execute o script `traffic.py` para iniciar o treinamento da rede neural.
  ```bash
  python traffic.py
  ```
- O modelo será salvo no diretório `saved_model/` após o término do treinamento.

---

## 📈 Avaliação do Modelo

Após o treinamento, o modelo será avaliado no conjunto de teste, e as métricas de desempenho serão exibidas no terminal.

---

## 🧪 Testando o Modelo Treinado

Para testar o modelo em novas imagens, utilize o script `test_model.py`.

1. **Prepare uma Imagem de Teste**:
   - Certifique-se de que a imagem está no formato suportado (por exemplo, `.ppm`, `.jpg`, `.png`).
   - Redimensione a imagem para as dimensões adequadas se necessário (30x30 pixels).

2. **Execute o Script de Teste**:
   ```bash
   python test_model.py
   ```
   - O script exibirá a imagem com a classe prevista e a confiança da previsão, além de imprimir as informações no terminal.

---

## 📊 Resultados Obtidos

- **Acurácia de Treinamento**: [Insira a acurácia alcançada durante o treinamento, por exemplo, 98%].

- **Desempenho do Modelo**:
  - O modelo demonstrou alta precisão na classificação de sinais de trânsito, evidenciando a eficácia de redes neurais convolucionais para tarefas de visão computacional.

---

## 🗂️ Estrutura dos Scripts

### `traffic.py`
- **Descrição**: Script responsável pelo treinamento do modelo.
- **Funcionalidades**:
  - Carregamento e pré-processamento dos dados.
  - Construção e compilação da rede neural.
  - Treinamento e avaliação do modelo.
  - Salvamento do modelo treinado.

### `test_model.py`
- **Descrição**: Script utilizado para testar o modelo treinado em novas imagens.
- **Funcionalidades**:
  - Carregamento do modelo salvo.
  - Pré-processamento da imagem de teste.
  - Realização da previsão e exibição dos resultados.

---

## 🤝 Contribuições

Contribuições para o aprimoramento deste projeto são bem-vindas. Sinta-se à vontade para abrir **issues** ou enviar **pull requests**.

---

## 📚 Referências

- [GTSRB Dataset: German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)

---

## 📋 Licença

Este projeto utiliza a licença **MIT**.

---

## 📞 Contato

- **Anna Clara Ferraz** - [annaclara2006ferraz@gmail.com](mailto:annaclara2006ferraz@gmail.com)
- **Vinícius Dos Santos Andrade** - [vinicius_andrade2010@hotmail.com](mailto:vinicius_andrade2010@hotmail.com)

---
