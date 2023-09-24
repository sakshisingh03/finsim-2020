import json
import numpy
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from utils.scorer import compute_accuracy, compute_average_rank
from utils.text_processing import load_glove_model, text_to_vector

def get_X_Y(embeddings, term_list, dim):
    X, Y = [], []
    for term in term_list:
        phrase_vector= text_to_vector(term["term"], embeddings, dim)
    
        if type(X) == list:
            X = numpy.array(phrase_vector)
        else:
            X = numpy.vstack([X, phrase_vector])
        Y.append(term["label"])
    return X, Y


def train(X, Y, m):
    if m== 'l':
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
    elif m=='b':
        model= BernoulliNB()
    model.fit(X, Y)
    return model

def main(embeddings_path, terms_path, test_path= None, k=5, m= 'b', dim= 100):
    term_list = json.load(open(terms_path))
    embeddings = load_glove_model(embeddings_path)
    random.shuffle(term_list)
    
    if not test_path:

        # Train test split
        train_list = term_list[:int(0.8*len(term_list))]
        test_list = term_list[int(0.8*len(term_list)):]
        X_train, Y_train = get_X_Y(embeddings, train_list, dim= dim)
        model = train(X_train, Y_train, 'b')
    
        # Test
        predictions = []
        X_test, Y_test = get_X_Y(embeddings, test_list, dim= dim)
        probas = model.predict_proba(X_test).tolist()
    
        [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
    
        # Score
        accuracy = compute_accuracy(Y_test, predictions)
        average_rank = compute_average_rank(Y_test, predictions)
        print("Accuracy: ", accuracy)
        print("Average Rank: ", average_rank)
        
    else:
        train_list = term_list
        test_list = json.load(open(test_path))
        X_train, Y_train = get_X_Y(embeddings, train_list, dim= dim)
        model = train(X_train, Y_train, m)
        
        predictions = []
        X_test, Y_test = get_X_Y(embeddings, test_list, dim= dim)
        probas = model.predict_proba(X_test).tolist()
    
        [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]
    
        # Dump test predictions
        data = []
        for idx, example in enumerate(predictions):
            data.append({"term": test_list[idx]["term"], "label": test_list[idx]["label"], "predicted_labels": example[:k]})
        json.dump(data, open("data/outputs/prediction_class_not_in_terms.json", "w"), indent=4)


dim= 100
embeddings_path= f'models/custom_w2v_{dim}d.txt'
terms_path= 'data/terms/train_class_not_in_terms.json'
test_path= 'data/terms/test_class_not_in_terms.json'
main(embeddings_path, terms_path, k=5, m='b', dim= dim)
