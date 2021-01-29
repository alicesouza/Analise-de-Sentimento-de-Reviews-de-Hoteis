import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import tokenize
import seaborn as sns
from string import punctuation
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
import random

'''from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder'''
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

def pre_processamento():
    
    with open(os.path.join("Corpus_reduzido", "corpus.p"), "rb") as file:
        reviews = pickle.load(file)       

    with open(os.path.join("Corpus_reduzido", "corpus_polaridade.p"), "rb") as file:
        pol = pickle.load(file)
         
    dados = pd.DataFrame(data=reviews, index=range(0,len(reviews)), columns=['corpus'])
    dados['polaridade'] = pol
    print(dados)
    print(type(dados))
    print(dados.polaridade.value_counts())

    token_espaco = tokenize.WhitespaceTokenizer()
    token_pontuacao = tokenize.WordPunctTokenizer()

    frase_processada = list()    

    for opiniao in dados.corpus:
        nova_frase = list()
        palavras_texto = token_espaco.tokenize(opiniao)
        for palavra in palavras_texto:
            nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
        
    dados["tratamento_1"] = frase_processada

    stopwords_ = nltk.corpus.stopwords.words("portuguese")

    pontuacao = list()
    for ponto in punctuation:
        pontuacao.append(ponto)

    pontuacao_stopwords = pontuacao + stopwords_

    sem_acentos = [unidecode.unidecode(texto) for texto in dados["tratamento_1"]]
    #pontuação, stopwords, sem acento
    stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]
    dados["tratamento_2"] = sem_acentos
    stemmer = nltk.RSLPStemmer()

    frase_processada = list()
    for opiniao in dados["tratamento_2"]:
        nova_frase = list()
        opiniao = opiniao.lower() 
        palavras_texto = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_texto:
            if palavra not in stopwords_sem_acento:
                nova_frase.append(stemmer.stem(palavra)) #stemmer
                #nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
        
    dados["tratamento_2"] = frase_processada

    return dados


regressao_logistica = LogisticRegression(solver = "lbfgs")

def Regressao_Logistica():
    dados = pre_processamento()
    tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
    vetor_tfidf = tfidf.fit_transform(dados["tratamento_2"])
    treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                                  dados["polaridade"],
                                                                  test_size = 0.3,
                                                                  random_state = 42)
    regressao_logistica.fit(treino, classe_treino)
    acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
    print(acuracia_tfidf_ngrams)



def SupportVectorMachine():
    dados = pre_processamento()
    tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
    vetor_tfidf = tfidf.fit_transform(dados["tratamento_2"])
    treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                                  dados["polaridade"],
                                                                  test_size = 0.3,
                                                                  random_state = 42)

    print("cheguei até aqui")

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    print("oi1")
    SVM.fit(treino, classe_treino)
    print("oi2")
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(teste)
    print("oi3")
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, classe_teste)*100)
    print("oi4")


def NaiveBayes():
    dados = pre_processamento()
    tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
    vetor_tfidf = tfidf.fit_transform(dados["tratamento_2"])
    treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                                  dados["polaridade"],
                                                                  test_size = 0.3,
                                                                  random_state = 42)

    print("cheguei até aqui")

    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    print("oi1")
    Naive.fit(treino, classe_treino)
    print("oi2")
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(teste)
    print("oi3")
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, classe_teste)*100)
    print("oi4")
          
Regressao_Logistica()
NaiveBayes()
SupportVectorMachine()

