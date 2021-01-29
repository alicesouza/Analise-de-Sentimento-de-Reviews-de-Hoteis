import os
import pickle
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
import spacy
import random
from enelvo import normaliser

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



def lexicos_sentimento_LIWC(save=True):
    try:
        with open(os.path.join("Palavras_Sentimento","LIWC_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("Palavras_Sentimento","LIWC_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        sent_words = []
        sent_words_polarity = {}
        with open("liwc.txt", encoding="utf8") as f:
            text = f.readlines()
            for line in text:
                word = line.split()[0]
                word = pre_processing_text(word)
                if "126" in line:
                    # Positive sentiment word
                    sent_words.append(word)
                    sent_words_polarity[word] = "1"
                elif "127" in line:
                    sent_words.append(word)
                    sent_words_polarity[word] = "-1"
        # Remove duplicated words
        sent_words = list(set(sent_words))
    if save:
        with open(os.path.join("Palavras_Sentimento","LIWC_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("Palavras_Sentimento","LIWC_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)

    return sent_words, sent_words_polarity
    
def lexicos_sentimento_SentiLex(save=True):
    try:
        with open(os.path.join("Palavras_Sentimento","SentiLex_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("Palavras_Sentimento","SentiLex_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        print("\nProcessing Lexico")
        sent_words = []
        sent_words_polarity = {}
        f = open("SentiLex-flex-PT02.txt", encoding="utf8")
        text = f.readlines()
        for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            try:
                polarity = line[1].split('N0=')[1].split(';')[0]
            except:
                polarity = line[1].split('N1=')[1].split(';')[0]
            sent_words.append(word)
            sent_words_polarity[word] = polarity
    if save:    
        with open(os.path.join("Palavras_Sentimento","SentiLex_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("Palavras_Sentimento","SentiLex_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)
    
    return sent_words, sent_words_polarity

def lexicos_sentimento_OpLexicon():
    try:
        with open(os.path.join("Palavras_Sentimento","OpLexicon_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("Palavras_Sentimento","OpLexicon_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        print("\nProcessing Lexico")
        sent_words = []
        sent_words_polarity = {}
        f = open("lexico_v3.0.txt", encoding="utf8")
        text = f.readlines()
        for line in text:
            line = line.split(',')
            word = line[0]
            word = pre_processing_text(word)
            #word = N.unidecode(word) #tira acentuação
            polarity = line[2]
            sent_words.append(word)
            sent_words_polarity[word] = polarity
        
    with open(os.path.join("Palavras_Sentimento","OpLexicon_sent_words.p"), "wb") as f:
        pickle.dump(sent_words, f)
    with open(os.path.join("Palavras_Sentimento","OpLexicon_sent_words_polarity.p"), "wb") as f:
        pickle.dump(sent_words_polarity, f)
    
    return sent_words, sent_words_polarity

def concatenar(lexico_1, lexico_2, lexico_3, save=False):

    try:
        with open(os.path.join("Testes",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "rb") as f:
            sent_words = pickle.load(f)
        with open(os.path.join("Testes",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "rb") as f:
            sent_words_polarity = pickle.load(f)
    except:
        lexicos_sentimento_OpLexicon()
        lexicos_sentimento_SentiLex()
        lexicos_sentimento_LIWC()

        f = open(os.path.join("Palavras_Sentimento",lexico_1+"_sent_words.p"), "rb")
        sent_words = pickle.load(f)
        f = open(os.path.join("Palavras_Sentimento",lexico_1+"_sent_words_polarity.p"), "rb")
        sent_words_polarity = pickle.load(f)
        
        f = open(os.path.join("Palavras_Sentimento",lexico_2+"_sent_words.p"), "rb")
        sent_words_lexico_2 = pickle.load(f)
        f = open(os.path.join("Palavras_Sentimento",lexico_2+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_2 = pickle.load(f)

        f = open(os.path.join("Palavras_Sentimento",lexico_3+"_sent_words.p"), "rb")
        sent_words_lexico_3 = pickle.load(f)
        f = open(os.path.join("Palavras_Sentimento",lexico_3+"_sent_words_polarity.p"), "rb")
        sent_words_polarity_lexico_3 = pickle.load(f)

        for word in sent_words_lexico_2:
            if word not in sent_words:
                sent_words.append(word)
                sent_words_polarity[word] = sent_words_polarity_lexico_2[word]

        for word in sent_words_lexico_3:
            if word not in sent_words:
                sent_words.append(word)
                sent_words_polarity[word] = sent_words_polarity_lexico_3[word]
                
        with open(os.path.join("Testes",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words.p"), "wb") as f:
            pickle.dump(sent_words, f)
        with open(os.path.join("Testes",lexico_1+ '_'+ lexico_2+ '_'+ lexico_3+"_sent_words_polarity.p"), "wb") as f:
            pickle.dump(sent_words_polarity, f)   
    

    return sent_words, sent_words_polarity

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
    with(open(os.path.join("Corpus_Analisados", "TreeTagger_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 1            
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))   

            if word in sent_words and word not in stop_words:
                troca = False
                
                if palavra[x-1] in negacao:
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
    with(open(os.path.join("Corpus_Analisados", "TreeTagger_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
      
        for x, token in enumerate(text):
            x = x + 1            
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'AQ0') or (token[1] == 'AQA') or (token[1] == 'AQC') or (token[1] == 'AQS') or (token[1] == 'AO0') or (token[1] == 'AOA') or (token[1] == 'AOC') or (token[1] == 'AOS'):
                if word in sent_words and word not in stop_words:
                    troca = False
                    
                    if palavra[x-1] in negacao:
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
    with(open(os.path.join("Corpus_Analisados", "TreeTagger_analisados.p"), "rb")) as file:
        all_reviews = pickle.load(file)
    mais_comum = []
    subst = []
    aspects = []
     
    for i, text in enumerate(all_reviews):
        #print(i)
        '''print(reviews[i])
        print("polaridade review = ",polarity_reviews[i])'''
        palavra = ['']
        polaridade = 0
        sentimento_texto = 0
        polaridade_comentario.append(polarity_reviews[i])
        ADJ = "False"
        for x, token in enumerate(text):
            x = x + 1           
            
            word = pre_processing_text(token[0])
            palavra.append(str(word))
            
            if (token[1] == 'AQ0') or (token[1] == 'AQA') or (token[1] == 'AQC') or (token[1] == 'AQS') or (token[1] == 'AO0') or (token[1] == 'AOA') or (token[1] == 'AOC') or (token[1] == 'AOS'):                
                ADJ = 'True'
                if word in sent_words and word not in stop_words:
                    troca = False
                    
                    if palavra[x-1] in negacao:
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
                x = x + 1                   
                word = pre_processing_text(token[0])
                palavra.append(str(word))
                if word in sent_words and word not in stop_words:
                    troca = False
                    
                    if palavra[x-1] in negacao :
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

        
    acertos = 0
    analise(polaridade_comentario, result_review)

    
print("Analise 2 Janelas")

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

