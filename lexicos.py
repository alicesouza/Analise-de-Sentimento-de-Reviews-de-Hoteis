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



def lexicos_sentimento_LIWC(save=False):
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
                word = pre_processing_text(word)
                #word = line.split()[0]
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
