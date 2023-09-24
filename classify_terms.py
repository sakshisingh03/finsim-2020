#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:45:44 2020

@author: sakshi
"""

import json
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_term(term):
    temp= term.lower()
    temp= lemmatizer.lemmatize(temp)
    return temp
  
lemmatizer = WordNetLemmatizer() 

train_test= 'test'

data_path= 'data/terms/' + train_test + '.json'
tagset= 'data/tagset/finsim.json'

data = json.load(open(data_path, "r"))
tagset = json.load(open(tagset, "r"))

taglist= []

for tag in tagset:
    taglist.append(preprocess_term(tag['label']))
    
zero_class= []
one_class= []
    
for term in data:
    tokenized_term= tokenizer.tokenize(term['term'])
    processed_terms= list(map(preprocess_term,tokenized_term))
    common= list(set(taglist) & set(processed_terms))
    # moe than one class can also be there in terms. We will treat them as no class in terms
    if len(common)==1:
        one_class.append(term)
    else:
        zero_class.append(term)
        
json.dump(zero_class, open("data/terms/" + train_test + "_class_not_in_terms.json", "w"), indent=4)
json.dump(one_class, open("data/terms/" + train_test + "_class_in_terms.json", "w"), indent=4)