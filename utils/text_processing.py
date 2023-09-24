import numpy as np
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()

STOP = set(stopwords.words("english"))

def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, "r")
    model = {}
    for line in f:
        split_line = line.split()
        if len(split_line) > 100:
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print("Done.", len(model), " word vectors loaded!")
    return model


def text_to_vector(s, model, k=100):
    s = s.lower()
    l_s = word_tokenize(s)
    for i in range(len(l_s)):
        l_s[i]= lemmatizer.lemmatize(l_s[i])
    l_s_found = [a for a in l_s if (a in model and a and a not in STOP)]
    if l_s_found:
        out = 0
        for a in l_s_found:
            out += model[a]

        out = out / len(l_s_found)

        out = out / (np.linalg.norm(out) + 1e-9)

        return out
    else:
        return np.zeros((k,))