import numpy as np
import pandas as pd
import re
from gensim.utils import tokenize
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read patent high level mapping and claims
claims = pd.read_csv("../data/claims.csv", encoding = 'utf8')

# process claim text
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()

def sentence_process(sent):
    words = str(sent).split()

    # 1. alphanumeric only
    alphanum_only = [re.sub("[^a-zA-Z0-9]", "", w) for w in words]

    # 2. remove stop words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in alphanum_only if not w in stops]

    # 3. lemmatization
    words_lemma = [lemma.lemmatize(w) for w in meaningful_words]

    return words_lemma

# merge claims for the same patent
patent_corpus = {}
for i, v in claims.iterrows():
    pid = v["PATENT_ID"]
    text = sentence_process(v["CLAIM_TEXT"])
    if(pid not in patent_corpus):
        patent_corpus[pid] = text
    else:
        patent_corpus[pid].extend(text)
del claims

# Define Document Embedding class
class TaggedPatentDocument:
    def __init__(self, patent):
        self.patent = patent
    def __iter__(self):
        for pid, content in self.patent.items():
            yield TaggedDocument(content, [pid])

patent_docs = TaggedPatentDocument(patent_corpus)

# Training the Doc2vec Model
cores = multiprocessing.cpu_count()
model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=cores)
model.build_vocab(patent_docs)

for epoch in range(10):
    model.train(patent_docs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    #%%time model.train(patent_docs, total_examples=model.corpus_count, epochs=model.iter)

model.save('D2V')
