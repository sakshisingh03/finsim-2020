import nltk
import os
from gensim.models import Word2Vec
import regex as re
from tqdm import tqdm
import pdftotext
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
pdf_paths= '../data/English_prospectuses/'
dir_list= os.listdir(pdf_paths)

sentences= []
for i, pdf_name in enumerate(tqdm(dir_list)):
    if(pdf_name[-3:]!='pdf'):
        continue
    path= pdf_paths + pdf_name
    with open(path, "rb") as f:
        pdf = pdftotext.PDF(f)
    text = "\n\n".join(pdf)
    for sentence in sent_detector.tokenize(text.lower().strip()):
        temp_text= re.sub('[^a-zA-Z]',' ',sentence)
        temp_text= re.sub('\s+',' ',temp_text)
        sentences.append(word_tokenize(temp_text))


dim = 300
model = Word2Vec(sentences=sentences, sg=1, min_count=1, size=dim)

model.wv.save_word2vec_format(f"../models/custom_w2v_{dim}d.txt")

