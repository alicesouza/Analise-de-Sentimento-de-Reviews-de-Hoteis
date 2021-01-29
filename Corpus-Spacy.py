import os
import pickle
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
import spacy

from lexicos import corpus_treino_teste

def Pos_Tag_Spacy(save = "False"):

    try:
        with(open(os.path.join("Corpus_Analisados", "Spacy_analisados.p"), "rb")) as file:
            analisados = pickle.load(file)
            
        reviews, polarity_reviews = corpus_treino_teste('train')
              
    except:
        reviews, polarity_reviews = corpus_treino_teste('train')
        analisados = []
        for i, review in enumerate(reviews):
            print(i)
            nlp = spacy.load("pt_core_news_sm")
            doc = nlp(review)
            pos_tags = []
            
            for x, token in enumerate (doc):
                pos_tag = []
                       
                #pos_tag.append(ident)
                pos_tag.append(str(token))
                pos_tag.append(str(token.pos_))
                pos_tag.append(str(token.lemma_))
                
                pos_tags.append(pos_tag)

            analisados.append(pos_tags)

        print(analisados)

        if save:
            with open(os.path.join("Corpus_Analisados","Spacy_analisados.p"), "wb") as f:
                pickle.dump(analisados, f)

    return analisados, polarity_reviews


Pos_Tag_Spacy(save = "True")
