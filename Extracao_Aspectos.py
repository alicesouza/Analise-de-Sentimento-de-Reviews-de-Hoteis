import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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

import spacy
import fnmatch
from collections import Counter
import subprocess
import re
import gensim
from enelvo import normaliser
from ftfy import fix_encoding

def most_commom(lst):
    data = Counter(lst)
    return(data.most_common())

def corpus_treino_teste(tipo, save=False):
    try:
        with(open(os.path.join("Corpus_reduzido", tipo+"_corpus.p"), "rb")) as file:
            corpus = pickle.load(file)
        with(open(os.path.join("Corpus_reduzido", tipo+"_polaridade.p"), "rb")) as file:
            polaridade = pickle.load(file)
              
    except:

        with open(os.path.join("Corpus_reduzido", "corpus.p"), "rb") as file:
            reviews = pickle.load(file)       

        with open(os.path.join("Corpus_reduzido", "corpus_polaridade.p"), "rb") as file:
            pol = pickle.load(file)

        dados = pd.DataFrame(data=reviews, index=range(0,len(reviews)), columns=['corpus'])
        dados['polaridade'] = pol
        #print(dados)
        #print(type(dados))
        #print(dados.polaridade.value_counts())

        treino, teste, classe_treino, classe_teste = train_test_split(dados.corpus,
                                                                     dados.polaridade,
                                                                     test_size = 0.3,
                                                                     random_state = 42)

        treino = treino.values.tolist()
        teste = teste.values.tolist()
        classe_treino = classe_treino.values.tolist()
        classe_teste = classe_teste.values.tolist()

        classe_treino = [(str(classe)).replace("0", "-1") for classe in classe_treino]
        classe_teste = [(str(classe)).replace("0", "-1") for classe in classe_teste]
        
            
        if save:
            with open(os.path.join("Corpus_reduzido", "train_corpus.p"), "wb") as file:
                pickle.dump(treino, file)

            with open(os.path.join("Corpus_reduzido", "test_corpus.p"), "wb") as file:
                pickle.dump(teste, file)

            with open(os.path.join("Corpus_reduzido", "train_polaridade.p"), "wb") as file:
                pickle.dump(classe_treino, file)

            with open(os.path.join("Corpus_reduzido", "test_polaridade.p"), "wb") as file:
                pickle.dump(classe_teste, file)

        if tipo == "train":
            corpus = treino
            polaridade = classe_treino
        else:
            corpus = teste
            polaridade = classe_teste
            
    return corpus, polaridade

def corpus_normalizado(tipo, save="False"):
    
    try:
        with(open(os.path.join("Corpus_reduzido", tipo+"_corpus_normalized.p"), "rb")) as file:
            all_reviews_normalizado = pickle.load(file)
        with open(os.path.join("Corpus_reduzido", "corpus_polaridade.p"), "rb") as file:
            polaridade = pickle.load(file)
              
    except:
        all_reviews, polaridade  = corpus_treino_teste(tipo)
        all_reviews_normalizado = []
        for i, review in enumerate(all_reviews):
            print(i)
            norm = normaliser.Normaliser()
            review = norm.normalise(review)
            all_reviews_normalizado.append(review)
                
            
        if save:
            with open(os.path.join("Corpus_reduzido", tipo+"_corpus_normalized.p"), "wb") as file:
                pickle.dump(all_reviews_normalizado, file)

            
    return all_reviews_normalizado, polaridade
    

def pre_processing_text(text):

    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u"]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text

def pre_processamento(text):
    text = text.lower()

    input_chars = ["\n", ".", "!", "?", " / ", " - ", "|", '``', "''"]
    output_chars = [" . ", " . ", " . ", " . ", "/", "-", "", "", ""]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text

def pre_processamento2(corpus, remover_stopwords = "False", remover_acentos = "False", remover_pontuacao = "False", blabla = "False"): 

    token_pontuacao = tokenize.WordPunctTokenizer()
    token_espaco = tokenize.WhitespaceTokenizer()
    
    reviews = corpus

    frase_processada = list()

    for opiniao in reviews:
        nova_frase = list()
        opiniao = opiniao.lower()
        palavras_texto = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_texto:    
            nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))

    reviews = frase_processada

    StopWords_ = nltk.corpus.stopwords.words("portuguese")
    stopwords_sem_acento = [unidecode.unidecode(texto) for texto in StopWords_]
    stopwords_ = (StopWords_ + stopwords_sem_acento)
    
    if remover_acentos: 

        sem_acento = [unidecode.unidecode(texto) for texto in reviews]
        reviews = sem_acento

    if remover_stopwords:
        frase_processada = list()

        for opiniao in reviews:
            nova_frase = list()
            palavras_texto = token_espaco.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in stopwords_:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))

        reviews = frase_processada     
        

    if remover_pontuacao:
    
        pontuacao = list()
        for ponto in punctuation:
            pontuacao.append(ponto)
            
        frase_processada = list()
        for opiniao in reviews:
            nova_frase = list()
            palavras_texto = token_espaco.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in pontuacao:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))
                
        reviews = frase_processada
    return reviews

    if blabla:
        stemmer = nltk.RSLPStemmer()
        frase_processada = list()
        
        for opiniao in reviews:
            nova_frase = list()
            palavras_texto = token_espaco.tokenize(opiniao)
            for palavra in palavras_texto:
                nova_frase.append(stemmer.stem(palavra))
            frase_processada.append(' '.join(nova_frase))
            
        reviews = frase_processada

    return reviews





def TreeTagger(texto):
    
    file =  open(os.path.join("C:\TreeTagger", "texto.txt"), "w", encoding="utf8" )
    file.writelines(texto)    
    file.close()

    process = subprocess.Popen([r'\TreeTagger\executar.bat'],
                         shell = True,
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         universal_newlines=True)
    stdout, stderr = process.communicate()
    #print(stdout)
    result = stdout.split('\n')
    pos_tag = []
    for x in result:
        word = fix_encoding(x)
        pos_tag.append(word.split('\t'))
        
    return pos_tag


def aspecto_substantivo_TreeTagger(save = "False"):

    try:
        with(open("Nounprocessed_Reviews_TreeTagger.p", "rb")) as f:
            all_reviews = pickle.load(f)
    except:
        all_reviews, polaridade  = corpus_treino_teste("train")
        reviews = []
        for review in all_reviews:
            reviews.append(pre_processamento(review))
            all_reviews = reviews
        with open("Nounprocessed_Reviews_TreeTagger.p", "wb") as f:
            pickle.dump(all_reviews, f)

    
    mais_comum = []
    subst = []
    aspects = []
    for i, text in enumerate(all_reviews[:2000]):
        print(i)
        pos_tag = TreeTagger(text)
        pos_tag.remove([''])
        for token in pos_tag:            
            if (token[1] == 'NCMS') or (token[1] == 'NCMP') or (token[1] == 'NCFS') or (token[1] == 'NCFP') or (token[1] == 'NCCP') or (token[1] == 'NCCS') or (token[1] == 'NCCI'):
                palavra = pre_processing_text(token[0])
                #token = unidecode.unidecode(token.norm_) 
                #subst.append(str(token.lemma_))
                subst.append(str(palavra))

    mais_comum = most_commom(subst)
    #freq = (len(mais_comum))*0.02
    #mais_comum = (mais_comum[:(int(freq))])
    print("\n",mais_comum[:200])
    
    for tupla in mais_comum:
        aspects.append(tupla[0])
    
    print(aspects[:200])

    if save:
        with open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "wb") as f:
            pickle.dump(aspects, f)

    return aspects


def aspecto_substantivo_NLTK(frequency_cut=0.03, save=False):
    reviews = []
    with open("tagger.pkl", "rb") as f:
        tagger = pickle.load(f)

    try:
        with(open("Nounprocessed_Reviews_NLTK.p", "rb")) as f:
            all_reviews = pickle.load(f)
    except:
        all_reviews, polaridade  = corpus_treino_teste("train")
        for review in all_reviews:
            reviews.append(pre_processamento(review))
            all_reviews = reviews
        with open("Nounprocessed_Reviews_NLTK.p", "wb") as f:
            pickle.dump(all_reviews, f)
            
    portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
    
    noun_words = {}
    aspects =[]
    subst = []
    mais_comum = []
    for i, review in enumerate(all_reviews):
        #print(i)
        sentences = portuguese_sent_tokenizer.tokenize(review)
        tag_review = [tagger.tag(nltk.word_tokenize(sentence)) for sentence in sentences]
        for tag_sentence in tag_review:
            for tag in tag_sentence:
                if tag[1] == "NOUN":
                    word = pre_processing_text(tag[0])
                    subst.append(word)

    mais_comum = most_commom(subst)
    #freq = (len(mais_comum))*0.02
    #mais_comum = (mais_comum[:(int(freq))])
    #print("\n",mais_comum[:200])
    
    for tupla in mais_comum:
        aspects.append(tupla[0])
    
    print(aspects[:250])
    
    if save:
        with open(os.path.join("Aspectos","noun_aspects_NLTK.p"), "wb") as f:
            pickle.dump(aspects, f)
    return aspects

'''def aspecto_substantivo_TreeTagger(save = "False"):
    file = open("Corpus_TreeTagger.txt", "r", encoding="utf8")
    corpus = file.read()
    all_reviews = corpus.split('id_')
    file.close()
    all_reviews.remove('')
    mais_comum = []
    subst = []
    aspects = []
    for i, text in enumerate(all_reviews):
        #print(i)
        text = text.split('\n')
        pos_tag = []        
        for line in text:
            pos_tag.append(line.split('\t'))

        #print(pos_tag)
        pos_tag.remove([''])
        for token in pos_tag:
            if (token[1] == 'NCMS') or (token[1] == 'NCFS') or (token[1] == 'NCFP') or (token[1] == 'NCCP') or (token[1] == 'NCCS') or (token[1] == 'NCCI'):
                              
                palavra = pre_processing_text(token[0])
                subst.append(str(palavra))

    mais_comum = most_commom(subst)
    #print("\n",mais_comum[:200])
    
    for tupla in mais_comum:
        aspects.append(tupla[0])
    
    print(aspects[:200])

    if save:
        with open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "wb") as f:
            pickle.dump(aspects, f)

    return aspects'''
    
def aspecto_substantivo_spacy(save = "False"):

    try:
        with(open("Nounprocessed_Reviews_Spacy.p", "rb")) as f:
            all_reviews = pickle.load(f)
    except:
        all_reviews, polaridade  = corpus_treino_teste("train")
        reviews = []
        for review in all_reviews:
            reviews.append(pre_processamento(review))
            all_reviews = reviews

        with open("Nounprocessed_Reviews_Spacy.p", "wb") as f:
            pickle.dump(all_reviews, f)

    
    mais_comum = []
    subst = []
    aspects = []
    for i,text in enumerate(all_reviews):
        print(i)
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(text)
        for token in doc:
            if token.pos_ == 'NOUN':
                token = pre_processing_text(token.norm_)
                #token = unidecode.unidecode(token.norm_) 
                #subst.append(str(token.lemma_))
                subst.append(str(token))

    mais_comum = most_commom(subst)
    #freq = (len(mais_comum))*0.02
    #mais_comum = (mais_comum[:(int(freq))])
    print("\n",mais_comum[:200])
    
    for tupla in mais_comum:
        aspects.append(tupla[0])
    
    print(aspects[:200])

    if save:
        with open(os.path.join("Aspectos","noun_aspects_Spacy.p"), "wb") as f:
            pickle.dump(aspects, f)

    return aspects

def create_aspects_lexicon_ontologies():
    """Create a list of the aspects indicated in the groups file"""
    aspects = []
    with open("Hontology.xml", "r", encoding="utf8") as file:
        text = file.readlines()
        for line in text:
            if "pt" in line:
                word  = line.split('>')[1].split('<')[0]
                if word != "\n":
                    aspects.append(pre_processing_text(word))
                
    print(aspects)
    
    with open(os.path.join("Aspectos","ontology_aspects.p"), "wb") as f:
        pickle.dump(aspects, f)
    return aspects

def create_aspects_lexicon_embeddings(seeds, seeds_type, number_synonym=3,save=False):
    aspects_list = []
    model = gensim.models.Word2Vec.load("word2vec.model")
    for word in seeds:
        aspects_list.append(word)
        if word in model.wv.vocab:
            out = model.wv.most_similar(positive=word, topn=number_synonym)  
            aspects_list.append(out[0][0].lower())
            aspects_list.append(out[1][0].lower())
            aspects_list.append(out[2][0].lower())
    aspects_list = list(set(aspects_list))
    if save:
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects.p"), "wb") as file:
            pickle.dump(aspects_list, file)
    return aspects_list

#def implicito_explicido_aspectos():
    

#aspecto_substantivo_NLTK(save = 'True')
#aspecto_substantivo_TreeTagger('True')
#aspecto_substantivo_spacy(save = 'True')
#create_aspects_lexicon_ontologies()

#all_reviews, polaridade = corpus_normalizado("train", save="True")

with open(os.path.join("Aspectos","noun_aspects_TreeTagger.p"), "rb") as f:
    seeds = pickle.load(f)

print(seeds[:10])
    
aspects_list = create_aspects_lexicon_embeddings(seeds[:10], 'noun_TreeTagger_agora',save=True)
print(aspects_list)



'''all_reviews, polaridade  = corpus_treino_teste("train")
TreeTagger(all_reviews[0])'''

