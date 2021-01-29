import os
import pickle
import nltk
from nltk.corpus import stopwords

'''file = open("Corpus_TreeTagger.txt", "r", encoding="utf8")
corpus = file.read()
all_reviews = corpus.split('id_')
file.close()

analisados = []
for i, review in enumerate(all_reviews):
    review = review.split('\n')
    word = []
    pos_tags = []
    
    for line in review:
        word.append(line.split('\t'))
        
    word.remove([''])
    for token in enumerate(word):
        pos_tag = []

        pos_tag.append(str(token[1][0]))
        pos_tag.append(str(token[1][1]))
        pos_tag.append(str(token[1][2]))

        pos_tags.append(pos_tag)

    analisados.append(pos_tags)
    #print(analisados)


with open(os.path.join("Corpus_Analisados","TreeTagger_analisados.p"), "wb") as f:
    pickle.dump(analisados, f)'''

with(open(os.path.join("Corpus_Analisados", "TreeTagger_analisados.p"), "rb")) as file:
    all_reviews = pickle.load(file)

print(all_reviews[0])

del(all_reviews[0])

print(all_reviews[0])

with open(os.path.join("Corpus_Analisados","TreeTagger_analisados.p"), "wb") as f:
    pickle.dump(all_reviews, f)
