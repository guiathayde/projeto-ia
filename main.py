import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
import re
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import wordpunct_tokenize

# lista de stopwords
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import string
import emoji

df_tweets = pd.read_csv("en.csv")
df_tweets.drop_duplicates(inplace=True)

"""# Tratamento dos Dados

## Dicionário para tratar "expressões de internet"

Nesta etapa juntei o máximo de expressões possíveis que consegui lembrar para unificá-las antes de passar pelo corretor automático.
"""

#Dicionário com todas palavras ou expressões regulares para substituição
dict_replace_words = {"user": "", "rt":"", "vc":"você", "cê": "você", "ce": "você", "pra":"para", "mt": "muito", "mto": "muito",
                      "q": "que", "qm": "quem", "qd": "quando", "qdo": "quando", "qt": "quanto", "qto": "quanto", "pq": "porque",
                      "vtnc": "vai tomar no cu", "poha":"porra", "vsf": "vai se fuder", "vsfd": "vai se fuder", "fdp": "filho da puta",
                      "to": "estou", "ta": "está", "aq": "aqui", "aki": "aqui", "tb": "também", "tbm": "também", "s": "sim", "n": "não",
                      "trampo": "trabalho", "pdp": "sim", "pode pa": "sim", "pode pá": "sim", "man": "homem", "mn": "homem", "mano": "homem",
                      "mina": "mulher", "eh": "é", "ngm": "ninguém", "ng": "ninguém", "algm":"alguém", "qr": "quer", "obg": "obrigado",
                      "bnt": "bonito", "oq": "o que", "grt": "garoto", "msm": "mesmo", "pae": "pai", "sdd": "saudade", "sdds": "saudade",
                      "bj": "beijo", "bjo": "beijo", "bjs": "beijo", "cmg": "comigo", "smp": "sempre", "sp": "sao_paulo", "são paulo": "sao_paulo",
                      "rj": "rio_de_janeiro", "rio de janeiro": "rio_de_janeiro", "boceta": "buceta", "pica": "pênis", "pika": "pênis",
                      "rs": "risada", "tnc": "vai tomar no cu", "kct": "cacete", "ae": "aí", "td": "tudo", "tão": "estão", "tao": "estão",
                      "net":"internet", "né":"sim", "ne":"sim", "fd": "foda", "crl": "caralho", "cel": "celular", "agr": "agora",
                      "bb": "bebê", "hr": "hora", "tava": "estava", "d": "de", "nd": "nada", "fml": "família", "slc": "está louco",
                      "slk": "está louco", "amg": "amigo", "plmdds": "pelo amor de deus", "ntj": "não tem jeito", "pdc": "sim",
                      "pprt": "papo reto", "pqp": "puta que pariu", "daora": "bom", "dahora": "bom", "bora": "vamos", "vms":"vamos",
                      "mpb": "música popular brasileira", "roque": "rock", "qqr": "qualquer", "qlqr": "qualquer", "qlq": "qualquer",
                      "mlk": "menino", "sfd": "vai se fuder",
                      re.compile(r'k+'): 'risada'
}

dict_replace_words_en = {
                      "u": "you", "ur": "your", "urs": "yours", "r": "are", "y": "why", "w/": "with", "w/o": "without",
                      "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you", "np": "no problem",
                      "idk": "I don't know", "idc": "I don't care", "imo": "in my opinion", "imho": "in my humble opinion",
                      "btw": "by the way", "brb": "be right back", "bbl": "be back later", "g2g": "got to go", "gtg": "got to go",
                      "tbh": "to be honest", "afk": "away from keyboard", "smh": "shaking my head", "fml": "fuck my life",
                      "ikr": "I know, right?", "lmk": "let me know", "dm": "direct message", "gg": "good game", "hf": "have fun",
                      "gl": "good luck", "wp": "well played", "jk": "just kidding", "lol": "laughing out loud",
                      "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing", "omg": "oh my god",
                      "wtf": "what the fuck", "wth": "what the hell", "stfu": "shut the fuck up", "afaik": "as far as I know",
                      "icymi": "in case you missed it", "fyi": "for your information", "irl": "in real life", "imo": "in my opinion",
                      "bff": "best friends forever", "hmu": "hit me up", "nvm": "never mind", "tho": "though",
                      "bro": "brother", "sis": "sister", "bc": "because", "asap": "as soon as possible",
                      "tgif": "thank god it's Friday", "nsfw": "not safe for work", "tldr": "too long; didn't read",
                      "wyd": "what are you doing?", "wya": "where are you at?", "ttyl": "talk to you later",
                      "ikr": "I know, right?", "fr": "for real", "goat": "greatest of all time", "cap": "lie",
                      "no cap": "no lie", "sus": "suspicious", "yeet": "throw",
                      re.compile(r'h+a+h+a+|l+o+l+|l+m+a+o+'): 'laugh'
}

# Inicialize o stemmer RSLP
ptstemmer = RSLPStemmer()

#Setar como portugues e ingles
stopwordspt = set(stopwords.words("portuguese"))
stopwordsen = set(stopwords.words("english"))

"""## Definição do Corpus para Correção das Palavras

Para compor o Corpus optei pelo _Corpus Brasileiro_ disponível em https://www.linguateca.pt/ACDC/, contendo aproximadamente 1bilhão de palavras, sendo o Corpus público mais completo que encontrei.

O Corpus Brasileiro é uma coletânea de aproximadamente um bilhão de palavras de português brasileiro, resultado de projeto coordenado por Tony Berber Sardinha, (GELC, LAEL, Cepril, PUCSP), com financiamento da Fapesp.
"""

#https://www.linguateca.pt/ACDC/
#http://www.nilc.icmc.usp.br/nilc/tools/corpora.htm

#https://www.wordfrequency.info/samples.asp

#pt
#corpus = pd.read_csv('FREQUENCIA_pt.csv', encoding='Latin-1',sep=';')
#en
corpus = pd.read_csv('FREQUENCIA_en.csv')

corpus = corpus[['Palavra',  'Frequencia']]

# 361900 palavras
corpus.to_csv('FREQUENCIA_TRATADA.csv', header=False, index=False, sep=";")

"""## Correção de Palavras
Como era uma competição que estava desenvolvendo entre meu tempo livre entre trabalho e faculdade, precisava que o processamento fosse o mais rápido possível mantendo a qualidade do tratamento, testei algumas bibliotecas para correção de palavras e a que teve o melhor desepenho foi a symspellpy, onde obtive maiores detalhes sobre seu funcionamento através dos artigos [A quick overview of the implementation of a fast spelling correction algorithm](https://medium.com/@agusnavce/a-quick-overview-of-the-implementation-of-a-fast-spelling-correction-algorithm-39a483a81ddc) e [Spell check and correction[NLP, Python]](https://medium.com/@yashj302/spell-check-and-correction-nlp-python-f6a000e3709d)
"""

#Corretor Ortográfico por cada palavra - execução rápida

from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=10)

# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary('FREQUENCIA_TRATADA.csv', term_index=0, count_index=1, separator=';')

"""### Exceções

Na célula abaixo considerei nomes comuns no Brasil, as maiores marcas globais, maiores marcas do Brasil e palavras em inglês para não serem alteradas para outras palavras erroneamente pelo SymSpell
"""

from spellchecker import SpellChecker
#Corretor ortográfico para verificar se a palavra existe no dicinoário
portugueseSpellChecker = SpellChecker(language='pt')
englishSpellChecker = SpellChecker(language='en')

#DataFrame de nomes do Brasil https://brasil.io/dataset/genero-nomes/files/
df_nomes = pd.read_csv('nomes.csv.gz', compression='gzip', sep=',')
nomes = list(df_nomes['group_name'].unique())
nomes = [palavra.lower() for palavra in nomes]

#DataFrame com as principais marcas do Mundo https://www.kaggle.com/datasets/gauravarora1091/top-100-global-brands-by-brandirectory2022
df_brand = pd.read_csv('brandirectory-ranking-data-global-2022.csv', encoding= 'Latin-1')
brand_names = list(df_brand['Brand'])
brand_names = [palavra.lower() for palavra in brand_names]

#DataFrame com as principais marcas do Brasil https://brandirectory.com/rankings/brazil/table
df_brand_br = pd.read_csv('brandirectory-ranking-data-brazil-2024.csv')
brand_names_br = list(df_brand_br['Name'])
brand_names_br = [palavra.lower() for palavra in brand_names_br]

#Dicionário de Inglês
import nltk
from nltk.corpus import wordnet
# Baixar o corpus WordNet (caso ainda não tenha sido baixado)
nltk.download('wordnet')
# Criar o dicionário em inglês
dicionario_ingles = list(set(wordnet.words()))

"""### Funções para Tratamento

Nesta etapa é realizada a remoção das stop words, links, substituição de emojis por palavras, correção de palavras e tokenização

"""

def preproc_tokenizer(text):

    # quebra o documento em  tokens
    words = wordpunct_tokenize(text)

    # converte palavras para minusculo
    words = [word.lower() for word in words]

    # remove as stopwords
    words = [word for word in words if word not in stopwordspt]

    # remove pontuacao
    words = [word for word in words if word not in string.punctuation]

    #if reduc == 'lemmatizer':
    #    words = [wnl.lemmatize(word) for word in words]
    #else:
    #    words = [stm.stem(word) for word in words]

    return(words)
def substituir_links_por_link(texto):
    # Expressão regular para encontrar URLs
    padrao_url = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    # Substituir URLs por "link"
    texto_substituido = re.sub(padrao_url, '', texto)

    return texto_substituido

def remover_pontuacao(texto):
    # Definir padrão de expressão regular para correspondência de pontuações
    padrao = r'[^\w\s]'

    # Substituir pontuações por uma string vazia
    texto_sem_pontuacao = re.sub(padrao, ' ', texto)

    return texto_sem_pontuacao

def remover_emojis(texto):
    #texto = emoji.demojize(texto, language='pt')
    texto = emoji.demojize(texto, language='en')
    return texto

def CorrigePalavra(input_term, lista_correcao):


    if input_term in lista_correcao:
        # lookup suggestions for multi-word input strings (supports compound
        # splitting & merging)

        # max edit distance per lookup (per single word, not per whole input string)
        #suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
        suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=5, ignore_token=r"\w+(?:_\w+){2,4}")
        #
        #print(suggestions)
        # display suggestion term, edit distance, and term frequency
        #textoCorrigido = suggestions[0].term
        try:
            palavraCorrigida = suggestions[0].term
        except:
            palavraCorrigida = input_term
        return palavraCorrigida
    else:
        return input_term


def RemovePalavra (lista, palavra):
    if palavra in lista:
        lista.remove(palavra)
    return lista
# Função para substituir palavras inteiras usando expressões regulares
def substituir_palavras(texto, substituicoes):
    # Substituir palavras usando expressões regulares no dicionário
    for chave, valor in substituicoes.items():
        if isinstance(chave, str):
            # Usar expressão regular para encontrar palavras inteiras
            regex = r'\b' + re.escape(chave) + r'\b'
            texto = re.sub(regex, valor, texto)
        elif isinstance(chave, re.Pattern):
            # Se a chave já é uma expressão regular, usar diretamente
            texto = chave.sub(valor, texto)
    return texto
# Função para substituir sequências repetidas de letras em palavras com 5 dígitos ou mais
def reduzir_repeticoes(texto):
    # Expressão regular para encontrar palavras com 5 dígitos ou mais
    regex = r'\b\w{5,}\b'
    # Função de substituição para reduzir sequências repetidas de letras
    def substituicao(match):
        palavra = match.group()
        return re.sub(r'(\w)\1{2,}', r'\1', palavra)  # Substituir sequências repetidas por uma única ocorrência
    # Aplicar a substituição usando a função de substituição definida acima
    return re.sub(regex, substituicao, texto)
def ListaCorrecao(coluna_texto):
    #coluna_texto = df['text']

    lista_palavras_unicas = list(set(coluna_texto.str.split().sum()))
    lista_palavras_unicas = list(portugueseSpellChecker.unknown(lista_palavras_unicas))
    #Cria uma lista com as palavras não encontradas no dicionário da língua portuguesa
    #para serem corrigidas, desconsiderando nomes, marcas populares mundialmente e no Brasil e palavras em inglês
    lista_palavras_unicas = [palavra for palavra in lista_palavras_unicas if palavra not in nomes]
    lista_palavras_unicas = [palavra for palavra in lista_palavras_unicas if palavra not in brand_names]
    lista_palavras_unicas = [palavra for palavra in lista_palavras_unicas if palavra not in brand_names_br]
    lista_palavras_unicas = [palavra for palavra in lista_palavras_unicas if palavra not in dicionario_ingles]

    return lista_palavras_unicas

def TrataDfTexto(df_texto, col_texto, dict_replace):
    #Remove emojis, links, pontuação
    df_texto['text_editado'] = df_texto[col_texto].str.lower()
    df_texto['text_editado'] = df_texto['text_editado'].apply(remover_emojis)
    df_texto['text_editado'] = df_texto['text_editado'].apply(substituir_links_por_link)
    df_texto['text_editado'] = df_texto['text_editado'].apply(remover_pontuacao)

    #utilizando regex substitui repetições de letras como "siiiim" para "sim"
    df_texto['text_editado'] = df_texto['text_editado'].apply(reduzir_repeticoes)

    #substitui palavras baseado num dicinoário com palavras ou expressões regulares
    df_texto['text_editado'] = df_texto['text_editado'].apply(lambda x: substituir_palavras(x, dict_replace))

    #lista com todas palavras que devem ser corrigidas
    lista_correcao = ListaCorrecao(df_tweets['text_editado'])

    #tokeniza o texto como uma lista
    df_texto['text_tokenizado'] = [preproc_tokenizer(text) for text in df_texto['text_editado']]
    df_texto['text_tokenizado_corrigido'] = df_texto['text_tokenizado'].apply(lambda lista: [CorrigePalavra(palavra, lista_correcao) for palavra in lista])
    df_texto['text_tokenizado_corrigido'] = df_texto['text_tokenizado_corrigido'].apply(lambda lista: [ palavra for palavra in lista if len(palavra) > 1])
    df_texto['text_stem'] = df_texto['text_tokenizado_corrigido'].apply(lambda lista: [ptstemmer.stem(word) for word in lista])

    return df_texto

# Aplicar a função ao DataFrame df_tweets
df_tweets = TrataDfTexto(df_tweets, 'text', dict_replace_words)

"""# Visualização dos Dados

## Word Cloud
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def gen_wordcloud(texts, title, size=50):
    # conta a frequencia de cada termo
    frequencies = Counter(token for doc in texts for token in set(doc))

  # gera a nuvem de palavras
    wc = WordCloud()
    wc.generate_from_frequencies(dict(frequencies.most_common(size)))

    # plota a nuvem
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()

gen_wordcloud(df_tweets['text_stem'],"Stemming")

counter_palavras = Counter(token for doc in df_tweets['text_stem'] for token in set(doc))
counter_palavras.most_common(10)

gen_wordcloud(df_tweets['text_tokenizado_corrigido'],"Stemming")

counter_palavras = Counter(token for doc in df_tweets['text_tokenizado_corrigido'] for token in set(doc))
counter_palavras.most_common(10)

print("counter_palavras")
print(len(counter_palavras))

counter_palavras_stem = Counter(token for doc in df_tweets['text_stem'] for token in set(doc))
counter_palavras_stem.most_common(10)

print("counter_palavras_stem")
print(len(counter_palavras_stem))

"""## Quantidade de Palavras x Aparições na Base"""

import plotly.graph_objects as go
from collections import Counter

# Seu Counter counter_palavras
counter_palavras_stem = Counter(token for doc in df_tweets['text_stem'] for token in set(doc))

# Definir as categorias
categorias = [
    {"nome": "0-9", "faixa": range(0, 10)},
    {"nome": "10-19", "faixa": range(0, 20)},
    {"nome": "20-29", "faixa": range(0, 30)},
    {"nome": "30-39", "faixa": range(0, 40)},
    {"nome": "40-49", "faixa": range(0, 50)},
    {"nome": "50-99", "faixa": range(50, 100)},
    {"nome": "100-500", "faixa": range(100, 501)},
    {"nome": "500-1000", "faixa": range(500, 1001)},
    {"nome": "1000-2000", "faixa": range(1000, 2001)},
    {"nome": "2000 ou mais", "faixa": range(2000, max(counter_palavras_stem.values()) + 1)}
]

# Inicializar o contador de palavras por categoria
contagem_categorias = {categoria["nome"]: 0 for categoria in categorias}

# Contar quantas palavras estão em cada categoria
for contagem in counter_palavras_stem.values():
    for categoria in categorias:
        if contagem in categoria["faixa"]:
            contagem_categorias[categoria["nome"]] += 1
            break

# Criar uma lista com os valores de contagem formatados para exibição sobre as barras
valores_formatados = [f"{valor}" for valor in contagem_categorias.values()]

# Criar o gráfico de barras com Plotly
fig = go.Figure()

# Adicionar os dados ao gráfico
fig.add_trace(go.Bar(
    x=[categoria["nome"] for categoria in categorias],  # Categorias
    y=list(contagem_categorias.values()),  # Contagem de palavras em cada categoria
    text=valores_formatados,  # Texto para exibir sobre as barras
    textposition='auto',  # Posição do texto (sobre a barra)
    marker_color='rgb(55, 83, 109)'
))

# Adicionar título e rótulos aos eixos
fig.update_layout(
    title='Contagem de Stem por Categoria',
    xaxis=dict(title='Categorias de Contagem'),
    yaxis=dict(title='Quantidade de Stem')
)

# Exibir o gráfico
fig.show()

len_counter = len(counter_palavras_stem)

palavras_comuns = [word for word, _ in counter_palavras.most_common(2000)]

df_tweets['text_stem_2000'] = df_tweets['text_stem'].apply(lambda lista: [palavra for palavra in lista if palavra in palavras_comuns])

df_tweets['text_stem_2000'] = df_tweets['text_stem_2000'].apply(lambda lista: [palavra for palavra in lista if len(palavra) > 1])

"""# Treino e Validação do Modelo

Nesta etapa foram testados diversos vectorizadores e modelos nas mais diversas configurações.

Ao usar redes neurais mais complexas foi observado overfitting nos dados de treino. O mesmo ocorreu ao utilizar word2Vec como vectorizador.

O melhor desempenho do modelo ocorreu ao utilizar o Count Vectorizer com bigramas e o limiar mínimo de 7 aparições para a palavra, com Regressão Logística ou uma Rede Neural mais simples com dropout em cada camada buscando reduzir o overfitting.

"""

# Separando os dados em features (X) e target (y)
text_stem= df_tweets['text_stem']
y = df_tweets['label']
X = [" ".join(tweet_words) for tweet_words in text_stem]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, df_tweets['label'].values, test_size=0.1, random_state=35, stratify=df_tweets['label'].values)

# Vetorizando os textos com TF-IDF (Term Frequency - Inverse Documment Frequency)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Count Vectorizer considerando bigramas
count_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=7)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

def plot_loss_precision_epochs(history):

    # Acesse as métricas de perda e precisão do histórico de treinamento
    losses = history.history['loss']
    accuracies = history.history['accuracy']

    # Acesse as métricas de perda e precisão do histórico de validação, se disponível
    val_losses = history.history['val_loss']
    val_accuracies = history.history['val_accuracy']

    # Plotar as curvas de perda
    plt.plot(losses, 'b', label='Perda de Treinamento')
    plt.plot(val_losses, 'r', label='Perda de Validação')
    plt.title('Curva de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

    # Plotar as curvas de precisão
    plt.plot(accuracies, 'b', label='Precisão de Treinamento')
    plt.plot(val_accuracies, 'r', label='Precisão de Validação')
    plt.title('Curva de Precisão')
    plt.xlabel('Épocas')
    plt.ylabel('Precisão')
    plt.legend()
    plt.show()

"""## Rede Neural"""

# Set random seed before creating the model
tf.random.set_seed(35)

model = Sequential()

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo com TF-IDF
history = model.fit(X_train_count.toarray(), y_train, batch_size=64, epochs=3, validation_data=(X_test_count.toarray(), y_test))

# Avaliando o modelo
loss, accuracy = model.evaluate(X_test_count.toarray(), y_test)
print(f'Acurácia do modelo com Count Vectorizer: {accuracy*100:.2f}%')

plot_loss_precision_epochs(history)

"""## Regressão Logística"""

lr_model = LogisticRegression(random_state=35, solver= 'liblinear')
lr_model.fit(X_train_tfidf, y_train)

y_pred = lr_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
confu_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.5f}')
print('Confusion Matrix:')
print(confu_matrix)

lr_model = LogisticRegression(random_state=35, solver= 'lbfgs')
lr_model.fit(X_train_count, y_train)

# Make predictions on the testing set
y_pred = lr_model.predict(X_test_count)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confu_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.5f}')
print('Confusion Matrix:')
print(confu_matrix)

"""# Aplicação do Modelo"""

df_test = pd.read_csv("test_pt.csv")
#display(df_test)

"""## Tratamento dos dados submetidos para a Competição"""

# Aplicar a função ao DataFrame df_tweets
df_test = TrataDfTexto(df_test, 'text', dict_replace_words)

text_stem_test = df_test['text_stem'].tolist()
flattened_text_test = [" ".join(tweet_words) for tweet_words in text_stem_test]

#Usando o Vectorizer treinado
X_final_tfidf = tfidf_vectorizer.transform(flattened_text_test)
X_final_count = count_vectorizer.transform(flattened_text_test)

"""## Predição"""

#Regressão Logística
regressao_predictions_test = lr_model.predict(X_final_count.toarray())

regressao_predicao = pd.DataFrame(regressao_predictions_test)
regressao_predicao.columns = ['predicted']

regressao_predicao['label'] = [1 if y >=0.5 else 0 for y in regressao_predicao['predicted']]

regressao_resultado = regressao_predicao.drop(['predicted'], axis= 1)

regressao_resultado['label'].sum()

regressao_resultado.index.name = 'id'

regressao_resultado.to_csv("regressao_resultado.csv")

#Rede Neural
rede_neural_predictions_test = model.predict(X_final_count.toarray())

neural_predicao = pd.DataFrame(rede_neural_predictions_test)
neural_predicao.columns = ['predicted']

neural_predicao['label'] = [1 if y >=0.5 else 0 for y in neural_predicao['predicted']]

neural_resultado = neural_predicao.drop(['predicted'], axis= 1)

neural_resultado['label'].sum()

neural_resultado.index.name = 'id'

neural_resultado.to_csv("neural_resultado.csv")
