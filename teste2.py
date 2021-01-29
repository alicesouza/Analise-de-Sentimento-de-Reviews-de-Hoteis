# -*- coding: utf-8 -*-
#mandar para marcio https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/tripadvisor/
"""
Created on Tue Nov 12 14:34:17 2019

@author: Alice
"""
import fnmatch
import pickle
import sys
import gensim
import heapq

import os
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
from enelvo import normaliser
import unidecode as N
import re
import numpy as np
import spacy
from collections import Counter

norm = normaliser.Normaliser()

def most_commom(lst):
    data = Counter(lst)
    return(data.most_common())

def pre_processamento(text):
    for i,elemento in enumerate(text):
        text[i] = norm.normalise(text[i])
        text[i] = text[i].lower()
        text[i] = text[i].replace('number ,' , '',1)
        norm.capitalize_inis = True
        #tirar acentuação
        #text[i] = N.unidecode(text[i])
        text[i] = text[i].strip() #removi os espaços no começo e no fim da string

        all_reviews.append(text[i])


def create_aspects_lexicon_ontologies():
    """Create a list of the aspects indicated in the groups file"""
    aspects = []
    with open("Hontology.xml", "r", encoding="utf8") as file:
        text = file.readlines()
        for line in text:
            if "pt" in line:
                word  = line.split('>')[1].split('<')[0]
                aspects.append(word)
            

    # Remove repetition on aspects list
    aspects = list(set(aspects))
    
    with open(os.path.join("Aspectos","ontology_aspects.p"), "wb") as f:
        pickle.dump(aspects, f)
    return [aspects]

def aspecto_substantivo(all_reviews, tipo):
    mais_comum = []
    subst = []
    aspect = []
    for i,text in enumerate(all_reviews):
        print('entrou')
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(text)
        for token in doc:
            if token.pos_ == 'NOUN':
                subst.append(str(token.lemma_))

    mais_comum = most_commom(subst)

    with open(os.path.join("Aspectos",tipo+"_aspects_quantidade.p"), "wb") as file:
            pickle.dump(mais_comum, file)
            
    for tupla in mais_comum:
        aspect.append(tupla[0])
        
    with open(os.path.join("Aspectos",tipo+"_aspects.p"), "wb") as file:
            pickle.dump(aspect, file)

    return aspect


def create_aspects_lexicon_embeddings(seeds, seeds_type, number_synonym=5,save=False):
    aspects_list = []
    model = gensim.models.Word2Vec.load("word2vec.model")
    for word in seeds:
        aspects_list.append(word)
        if word in model.wv.vocab:
            out = model.wv.most_similar(positive=word, topn=number_synonym)  
            aspects_list.append(out[0][0].lower())
            aspects_list.append(out[1][0].lower())
            aspects_list.append(out[2][0].lower())
            aspects_list.append(out[3][0].lower())
            aspects_list.append(out[4][0].lower())
    aspects_list = list(set(aspects_list))
    if save:
        with open(os.path.join("Aspectos",seeds_type+"_embedding_aspects.p"), "wb") as file:
            pickle.dump(aspects_list, file)
    return aspects_list

def create_aspects_lexicon_ontologies():
    """Create a list of the aspects indicated in the groups file"""
    aspects = []
    with open("Hontology.xml", "r", encoding="utf8") as file:
        text = file.readlines()
        for line in text:
            if "pt" in line:
                word  = line.split('>')[1].split('<')[0]
                aspects.append(word)
            

    # Remove repetition on aspects list
    aspects = list(set(aspects))
    
    with open(os.path.join("Aspectos","ontology_aspects.p"), "wb") as f:
        pickle.dump(aspects, f)
    return [aspects]



def lexicos_sentimento():
    sent_words = []
    sent_words_polarity = {}
    f = open("lexico_v3.0.txt", encoding="utf8")
    text = f.readlines()
    for line in text:
        line = line.split(',')
        word = line[0]
        #word = N.unidecode(word)
        polarity = line[2]
        sent_words.append(word)
        sent_words_polarity[word] = polarity
    
    with open(os.path.join("Palavras_Sentimento","sent_words_OpLexico.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("Palavras_Sentimento","sent_words_polarity_OpLexico.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)

    return [sent_words, sent_words_polarity]

def bag_of_words(reviews):
    text = ""
    for review in reviews[:10]:
        text += review
        #text.extend(review)

    print(type(text))    
    print(text)
    dataset = nltk.sent_tokenize(text) 
    for i in range(len(dataset)):
        #print(dataset[i])
        dataset[i] = dataset[i].lower()
        #print(dataset[i])            
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        #print(dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        #print(dataset[i])

    # Creating the Bag of Words model 
    word2count = {} 
    for data in dataset: 
        words = nltk.word_tokenize(data) 
        for word in words: 
            if word not in word2count.keys(): 
                word2count[word] = 1
            else: 
                word2count[word] += 1

    print((word2count))

    freq_words = heapq.nlargest(100, word2count, key=word2count.get)

    X = [] 
    for data in dataset: 
        vector = [] 
        for word in freq_words: 
            if word in nltk.word_tokenize(data): 
                vector.append(1) 
            else: 
                vector.append(0) 
        X.append(vector) 
    X = np.asarray(X) 




def polaridade_comentarios(reviews):
    del reviews[0]
    sent_words_polarity = {}
    sent_words = []
    stop_words = stopwords.words('portuguese')    
    stop_words.extend(['number','ser','estar'])
    result_review = []
    negacao = ['jamais','nada','nem','nenhum','ninguém','nunca','não','tampouco']
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']
    palavra = []
    positive = 0
    negative = 0
    try:
        with open(os.path.join("Palavras_Sentimento","sent_words_OpLexico.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("Palavras_Sentimento","sent_words_polarity_OpLexico.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        lexicos_sentimento()
        polaridade_comentarios(reviews)
    
    for i, review in enumerate (reviews):
        #review = review.split()
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(review)       
        
        positive = 0
        negative = 0
        palavra = ['','','']
        polaridade = 1
        sentimento_texto = 0
        for x, text in enumerate (doc):            
            palavra.append((str(text)))
            word_aux = (str(text))
            word = (str(text.lemma_))
            
            if text.pos_ == 'ADJ':
                word = (str(text))
                
            if word in sent_words and word not in stop_words:
              
                '''polaridade = float(sent_words_polarity[word])
                
                if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                        polaridade = polaridade/3                        
                    else:
                        polaridade = polaridade*3

                elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                        polaridade = polaridade*3
                    else:
                        polaridade = polaridade/3
                        
                elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:
                    polaridade = -1*polaridade

                print(polaridade)   
                sentimento_texto = sentimento_texto + polaridade
                
        print(review)
        print(sentimento_texto)'''
                troca = False
                
                if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao:

                    troca = True                           
                        
                if sent_words_polarity[word] == '1':
                    if troca:
                        print('troca')
                        negative += 1
                    else:    
                        positive += 1

                    #print(word)
                    #print(sent_words_polarity[word])
                        
                elif sent_words_polarity[word] == '-1':
                    if troca:
                        print('troca - neg')
                        positive += 1
                    else:
                        negative += 1

                    #print(word_aux)
                    #print(word)
                    #print(sent_words_polarity[word])

        if positive >= negative:
            result_review.append('1')
        else:
            result_review.append('-1')

        print(review)
        print(result_review[i])
        print(positive-negative)  
            
    
    

folder = os.listdir('C:\\Users\\Alice\\Desktop\\UFG\\Projeto\\Aprendendo\\hoteis')

try:
    with(open("Processed_Reviews_com_acento.p", "rb")) as file:
        all_reviews = pickle.load(file)
except:
    print("Processed_Reviews_com_acento.p couldn't be found. All reviews will be loaded from txt files, this will take a few minutes")
    all_reviews = []

    for arquivo in folder:
        file = open('hoteis/'+arquivo, "r",encoding="utf8")
        texto = file.read()
        review = texto.split('id_')
        if review[0] == '':
            del review[0]
        #textos.extend(sent)contatena uma string na outra
        pre_processamento(review)
    
    with open("Processed_Reviews_com_acento.p", "wb") as file:
        pickle.dump(all_reviews, file)


#aspecto_substantivo(all_reviews, 'substantivo')
        
'''substantivos = []    
file = open(os.path.join("Aspectos","substantivo_aspects.p"), "rb")
substantivos = pickle.load(file)

nouns = substantivos[int(len(substantivos)*.00) : int(len(substantivos) * .50)]
create_aspects_lexicon_embeddings(nouns, 'substantivo', 5, True)
#create_sentiment_words_lexicon(True)

file = open(os.path.join("Aspectos","ontology_aspects.p"), "rb")
ontologia = pickle.load(file)
create_aspects_lexicon_embeddings(ontologia, 'ontology', 5, True)
create_aspects_lexicon_ontologies()'''
polaridade_comentarios(all_reviews)
#bag_of_words(all_reviews)
file.close()

'''def pre_processing_text(text, use_normalizer=False):
    
    if use_normalizer:
        norm = normaliser.Normaliser()
        text = norm.normalise(text)

    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u"]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text


def create_sentiment_words_lexicon(save=False):
    """Create a sentiment words list using LIWC lexicon"""
    sent_words = []
    sent_words_polarity = {}
    with open("liwc.txt", encoding="utf8") as f:
        text = f.readlines()
        for line in text:
            word = pre_processing_text(line.split()[0])
            if "126" in line:
                # Positive sentiment word
                sent_words.append(word)
                sent_words_polarity[word] = "+"
            elif "127" in line:
                sent_words.append(word)
                sent_words_polarity[word] = "-"
    # Remove duplicated words
    sent_words = list(set(sent_words))
    if save:
        with open(os.path.join("Palavras_Sentimento","sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("Palavras_Sentimento","sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)

    return [sent_words, sent_words_polarity]


'''


