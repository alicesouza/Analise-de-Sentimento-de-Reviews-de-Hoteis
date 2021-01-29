import os
import pickle
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
import spacy
import random
from enelvo import normaliser
from lexicos import lexicos_sentimento_LIWC, lexicos_sentimento_SentiLex, lexicos_sentimento_OpLexicon, concatenar, corpus_treino_teste

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

def corpus_normalizado(tipo, save="False"):
    
    try:
        with(open(os.path.join("Corpus_reduzido", tipo+"_corpus_normalized.p"), "rb")) as file:
            all_reviews_normalizado = pickle.load(file)
        with open(os.path.join("Corpus_reduzido", tipo+"_polaridade.p"), "rb") as file:
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


def analise(polaridade_comentario, result_review):
    VP = 0
    FP = 0
    VN = 0
    FN = 0
    
    for i, pol in enumerate(polaridade_comentario):
        if pol == '1':            
            if pol == result_review[i]:
                VP +=1
            else:
                FN +=1
        elif pol == '-1':            
            if pol == result_review[i]:
                VN +=1
            else:
                FP +=1

    Pp = VP/(VP+FP)
    Pn = VN/(VN+FN)
    Cp = VP/(VP+FN)
    Cn = VN/(VN+FP)
    Fp = 2*((Pp*Cp)/(Pp+Cp))
    Fn = 2*((Pn*Cn)/(Pn+Cn))
    print("PRECISÃO")
    print("P-Positivo = ", Pp)
    print("P-Negativo = ", Pn)

    print("COBERTURA")
    print("C-Positivo = ", Cp)
    print("C-Negativo = ", Cn)

    print("F-Measure")
    print("F1-Positivo = ", Fp)
    print("F1-Negativo = ", Fn)
    print("F1-Média = ", (Fp+Fn)/2)

    print("ACURÁCIA")
    print("Acurácia = ", (VP+VN)/(VP+FN+FP+VN))

def polaridade_comentarios_palavras(sent_words, sent_words_polarity):
    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
        

    result_review = []
    polaridade_comentario = []
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco'] 
    reviews, polarity_reviews = corpus_treino_teste('train')

    
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 5            
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))   
    
            if word in sent_words and word not in stop_words:
                troca = False
                
                if palavra[x-1] in negacao or palavra[x-2] in negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                    troca = True                        
                    
                if sent_words_polarity[word] == '1':
                    if troca:
                        #print('troca')
                        polaridade += -1
                    else:    
                        polaridade += 1
                    '''print(word)
                    print(sent_words_polarity[word])'''
                        
                elif sent_words_polarity[word] == '-1':
                    if troca:
                        #print('troca')
                        polaridade += 1
                    else:
                        polaridade += -1
                    '''print(word)
                    print(sent_words_polarity[word])'''

        if polaridade >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')

        #print(review)
        """print("resultado = ", result_review[i])
        print("total = ", polaridade)"""

        
    analise(polaridade_comentario, result_review)            
            

def polaridade_comentarios_ADJ(sent_words, sent_words_polarity):
    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
        

    result_review = []
    polaridade_comentario = []
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco'] 
    reviews, polarity_reviews = corpus_treino_teste('train')
    
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 5           
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'ADJ'):
                if word in sent_words and word not in stop_words:
                    troca = False
                    
                    if palavra[x-1] in negacao or palavra[x-2] in negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        troca = True                        
                        
                    if sent_words_polarity[word] == '1':
                        if troca:
                            #print('troca')
                            polaridade += -1
                        else:    
                            polaridade += 1
                        '''print(word)
                        print(sent_words_polarity[word])'''
                            
                    elif sent_words_polarity[word] == '-1':
                        if troca:
                            #print('troca')
                            polaridade += 1
                        else:
                            polaridade += -1
                        '''print(word)
                        print(sent_words_polarity[word])'''

        if polaridade >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')

        #print(review)
        """print("resultado = ", result_review[i])
        print("total = ", polaridade)"""
    print("hello")
    analise(polaridade_comentario, result_review)    


def polaridade_comentarios_prefADJ(sent_words, sent_words_polarity):

    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco']

    result_review = []
    polaridade_comentario = []
    reviews, polarity_reviews = corpus_treino_teste('train')    
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
        ADJ = 'False'
        for x, token in enumerate(text):
            x = x + 5        
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'ADJ'):
                    ADJ = 'True'
                    if word in sent_words and word not in stop_words:
                        troca = False
                        
                        if palavra[x-1] in negacao or palavra[x-2] in negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            troca = True                        
                            
                        if sent_words_polarity[word] == '1':
                            if troca:
                                polaridade += -1
                            else:    
                                polaridade += 1
                                                            
                        elif sent_words_polarity[word] == '-1':
                            if troca:
                                polaridade += 1
                            else:
                                polaridade += -1

        if ADJ == 'False':
            for x, token in enumerate(text):
                x = x + 5                 
                word = pre_processing_text(token[0])
                palavra.append(str(word))
                if word in sent_words and word not in stop_words:
                    troca = False
                    
                    if palavra[x-1] in negacao or palavra[x-2] in negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        troca = True                        
                        
                    if sent_words_polarity[word] == '1':
                        if troca:
                            polaridade += -1
                        else:    
                            polaridade += 1
                            
                    elif sent_words_polarity[word] == '-1':
                        if troca:
                            polaridade += 1
                        else:
                            polaridade += -1

        if polaridade >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')

        #print(review)
        #print(result_review[i])
        #print(polaridade)

        
    analise(polaridade_comentario, result_review)

def polaridade_comentarios_negacao_reducao_itensificacao_palavras(sent_words, sent_words_polarity):
    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
        

    result_review = []
    polaridade_comentario = []
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco']
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']
    reviews, polarity_reviews = corpus_treino_teste('train')
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 5          
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if word in sent_words and word not in stop_words:
                polaridade = float(sent_words_polarity[word])
                
                if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao or palavra[x-4] in intensificacao or palavra[x-5] in intensificacao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        polaridade = polaridade/3                        
                    else:
                        polaridade = polaridade*3

                elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao or palavra[x-4] in reducao or palavra[x-5] in reducao:
                    if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        polaridade = polaridade*3
                    else:
                        polaridade = polaridade/3
                        
                elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                    polaridade = -1*polaridade
   
                sentimento_texto = sentimento_texto + polaridade
            
        if sentimento_texto >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')
                    
    analise(polaridade_comentario, result_review)

def polaridade_comentarios_negacao_reducao_itensificacao_ADJ(sent_words, sent_words_polarity):
    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
        

    result_review = []
    polaridade_comentario = []
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco']
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']
    reviews, polarity_reviews = corpus_treino_teste('train')
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 5           
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'ADJ'):
                if word in sent_words and word not in stop_words:
                    polaridade = float(sent_words_polarity[word])
                    
                    if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao or palavra[x-4] in intensificacao or palavra[x-5] in intensificacao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            polaridade = polaridade/3                        
                        else:
                            polaridade = polaridade*3

                    elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao or palavra[x-4] in reducao or palavra[x-5] in reducao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            polaridade = polaridade*3
                        else:
                            polaridade = polaridade/3
                            
                    elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        polaridade = -1*polaridade
       
                    sentimento_texto = sentimento_texto + polaridade
                
        if sentimento_texto >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')
                    
    analise(polaridade_comentario, result_review)


def polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(sent_words, sent_words_polarity):

    stop_words_ = stopwords.words('portuguese')
    stop_words = []
    for stop in stop_words_:
        stop_words.append(pre_processing_text(stop))
        
    #mal
    negacao = ['jamais','nada','nem','nenhum','ninguem','nunca','nao','tampouco'] 
    intensificacao = ['mais','muito','demais','completamente','absolutamente','totalmente','definitivamente','extremamente','frequentemente','bastante']
    reducao = ['pouco','quase','menos','apenas']

    result_review = []
    polaridade_comentario = []
    reviews, polarity_reviews = corpus_treino_teste('train')  
    with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['','','','','']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
        ADJ = 'False'
        for x, token in enumerate(text):
            x = x + 5            
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'ADJ'):
                ADJ = 'True'
                if word in sent_words and word not in stop_words:
                    polaridade = float(sent_words_polarity[word])
            
                    if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao or palavra[x-4] in intensificacao or palavra[x-5] in intensificacao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao:
                            polaridade = polaridade/3                        
                        else:
                            polaridade = polaridade*3

                    elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao or palavra[x-4] in reducao or palavra[x-5] in reducao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            polaridade = polaridade*3
                        else:
                            polaridade = polaridade/3
                            
                    elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        polaridade = -1*polaridade
       
                    sentimento_texto = sentimento_texto + polaridade
                            

        if ADJ == 'False':
            for x, token in enumerate(text):
                x = x + 5                  
                word = pre_processing_text(token[0])
                palavra.append(str(word))
                if word in sent_words and word not in stop_words:
                    polaridade = float(sent_words_polarity[word])
            
                    if palavra[x-1] in intensificacao or palavra[x-2] in intensificacao or palavra[x-3] in intensificacao or palavra[x-4] in intensificacao or palavra[x-5] in intensificacao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            polaridade = polaridade/3                        
                        else:
                            polaridade = polaridade*3

                    elif palavra[x-1] in reducao or palavra[x-2] in reducao or palavra[x-3] in reducao or palavra[x-4] in reducao or palavra[x-5] in reducao:
                        if palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                            polaridade = polaridade*3
                        else:
                            polaridade = polaridade/3
                            
                    elif palavra[x-1] in negacao or palavra[x-2] in  negacao or palavra[x-3] in negacao or palavra[x-4] in negacao or palavra[x-5] in negacao:
                        polaridade = -1*polaridade
       
                    sentimento_texto = sentimento_texto + polaridade
                    
        if sentimento_texto >= 0:
            result_review.append('1')
        else:
            result_review.append('-1')
        

        #print(review)
        #print(result_review[i])
        #print(polaridade)

        
    analise(polaridade_comentario, result_review)
    
print("Analise 6 Janelas")

OpLexicon_words, OpLexicon_words_polarity = lexicos_sentimento_OpLexicon()
SentiLex_words, SentiLex_words_polarity = lexicos_sentimento_SentiLex()
LIWC_words, LIWC_words_polarity = lexicos_sentimento_LIWC()
LOS_words, LOS_words_polarity = concatenar('LIWC', 'OpLexicon', 'SentiLex')
LSO_words, LSO_words_polarity = concatenar('LIWC', 'SentiLex', 'OpLexicon')
OLS_words, OLS_words_polarity = concatenar('OpLexicon', 'LIWC', 'SentiLex')
OSL_words, OSL_words_polarity = concatenar('OpLexicon', 'SentiLex', 'LIWC')
SLO_words, SLO_words_polarity = concatenar('SentiLex', 'LIWC', 'OpLexicon')
SOL_words, SOL_words_polarity = concatenar('SentiLex', 'OpLexicon', 'LIWC')


print("*****SOMENTE PALAVRAS*****")
print("OpLexicon")
polaridade_comentarios_palavras(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_palavras(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_palavras(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_palavras(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_palavras(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_palavras(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_palavras(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_palavras(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_palavras(SOL_words, SOL_words_polarity)
   

print("*****SOMENTE ADJETIVO*****")
print("OpLexicon")
polaridade_comentarios_ADJ(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_ADJ(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_ADJ(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_ADJ(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_ADJ(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_ADJ(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_ADJ(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_ADJ(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_ADJ(SOL_words, SOL_words_polarity)

print("*****PREFERÊNCIA A ADJETIVO*****")
print("OpLexicon")
polaridade_comentarios_prefADJ(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_prefADJ(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_prefADJ(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_prefADJ(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_prefADJ(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_prefADJ(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_prefADJ(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_prefADJ(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_prefADJ(SOL_words, SOL_words_polarity)

print("NEGAÇÃO, REDUÇÃO E NEGAÇÃO")

print("*****SOMENTE PALAVRAS*****")
print("OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_palavras(SOL_words, SOL_words_polarity)

print("*****SOMENTE ADJETIVOS*****")
print("OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_ADJ(SOL_words, SOL_words_polarity)

print("*****PREFERÊNCIA ADJETIVOS*****")
print("OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(OpLexicon_words, OpLexicon_words_polarity)
print("SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(SentiLex_words, SentiLex_words_polarity)
print("LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(LIWC_words, LIWC_words_polarity)
print("LIWC, OpLexicon, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(LOS_words, LOS_words_polarity)
print("LIWC, SentiLex, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(LSO_words, LSO_words_polarity)
print("OpLexicon, LIWC, SentiLex")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(OLS_words, OLS_words_polarity)
print("OpLexicon, SentiLex, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(OSL_words, OSL_words_polarity)
print("SentiLex, LIWC, OpLexicon")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(SLO_words, SLO_words_polarity)
print("SentiLex, OpLexicon, LIWC")
polaridade_comentarios_negacao_reducao_itensificacao_prefADJ(SOL_words, SOL_words_polarity)

