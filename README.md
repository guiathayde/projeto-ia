## **1. Importação de Bibliotecas**
Bibliotecas utilizadas para manipulação de dados, visualização, NLP e modelagem:
- **Pandas/NumPy**: Manipulação de dados.
- **Matplotlib/Seaborn/WordCloud**: Visualização de dados.
- **TensorFlow/Keras**: Construção de redes neurais.
- **Scikit-learn**: Modelos clássicos (Regressão Logística), vetorização (TF-IDF, CountVectorizer) e avaliação.
- **NLTK**: Pré-processamento de texto (tokenização, stopwords, stemming).
- **Gensim**: Word2Vec (não utilizado diretamente no código final).
- **SymSpellpy**: Correção ortográfica.
- **Emoji**: Tratamento de emojis.

## **2. Carregamento e Pré-Processamento dos Dados**
### **2.1. Leitura dos Dados**
```python
df_tweets = pd.read_csv("en.csv")
df_tweets.drop_duplicates(inplace=True)
```
- Carrega o dataset de tweets e remove duplicatas.

### **2.2. Dicionários de Substituição**
Dicionários para normalizar expressões informais em português (`dict_replace_words`) e inglês (`dict_replace_words_en`):
- Exemplo: "vc" → "você", "u" → "you".
- Regex substitui repetições de letras (ex: "kkkk" → "risada").

### **2.3. Configuração de NLP**
- **Stopwords**: Listas de palavras irrelevantes em português e inglês.
- **Stemming**: Redução de palavras à raiz usando `RSLPStemmer` para português.

---

## **3. Correção Ortográfica**
### **3.1. Carregamento do Corpus**
- Um corpus de frequência de palavras (`FREQUENCIA_en.csv`) é usado para treinar o corretor ortográfico SymSpell.
- **SymSpell**: Corrige palavras com base na distância de edição e frequência no corpus.

### **3.2. Exceções**
- Listas de nomes, marcas e palavras em inglês são usadas para evitar correções incorretas.

---

## **4. Funções de Pré-Processamento**
### **4.1. `preproc_tokenizer(text)`**
- Tokeniza o texto, converte para minúsculas, remove stopwords e pontuação.

### **4.2. `TrataDfTexto(df_texto, col_texto, dict_replace)`**
- Aplica etapas de pré-processamento:
  1. Remove emojis, links e pontuação.
  2. Substitui expressões usando os dicionários.
  3. Reduz repetições de caracteres (ex: "siiiim" → "sim").
  4. Corrige palavras com SymSpell.
  5. Aplica stemming nas palavras.

---

## **5. Visualização de Dados**
### **5.1. WordCloud**
Gera nuvens de palavras para visualizar termos mais frequentes antes e após o stemming.

### **5.2. Gráfico de Frequência de Palavras**
- Plotagem interativa (usando Plotly) mostra a distribuição de palavras por frequência no corpus.

---

## **6. Modelagem**
### **6.1. Vetorização**
- **TF-IDF**: Transforma texto em vetores numéricos ponderados.
- **CountVectorizer**: Considera bigramas e ignora termos com menos de 7 ocorrências.

### **6.2. Modelos**
#### **6.2.1. Rede Neural (TensorFlow)**
```python
model = Sequential([
    Dense(16, activation='relu'), Dropout(0.5),
    Dense(32, activation='relu'), Dropout(0.5),
    Dense(8, activation='relu'), Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
- Arquitetura com camadas densas e dropout para evitar overfitting.
- Compilada com otimizador Adam e função de perda `BinaryCrossentropy`.

#### **6.2.2. Regressão Logística (Scikit-learn)**
```python
lr_model = LogisticRegression(random_state=35, solver='lbfgs')
```
- Treinada com dados vetorizados (TF-IDF ou CountVectorizer).

### **6.3. Avaliação**
- Métricas: Acurácia, matriz de confusão.
- Curvas de perda e precisão durante o treinamento da rede neural.

---

## **7. Aplicação do Modelo**
### **7.1. Pré-Processamento do Teste**
- O dataset `test_pt.csv` é processado com as mesmas funções de tratamento.

### **7.2. Predição**
- As previsões são geradas usando a rede neural e a regressão logística.
- Resultados são exportados para CSV (`regressao_resultado.csv`, `neural_resultado.csv`).

---
