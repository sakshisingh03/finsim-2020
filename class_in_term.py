import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import load_glove_model, text_to_vector
from utils.scorer import compute_accuracy, compute_average_rank

def get_vectorized_data(data,tagset,model,dim=100):
    for t_entity in tagset:
        t_entity["vector"] = text_to_vector(t_entity["label"], model, dim)
    
    for entity in data:
        entity["vector"] = text_to_vector(entity["term"], model, dim)
    return data, tagset

def get_prediction(data,tagset,t='l2',k=5):
    for entity in data:
        if t=='l1':
            ontology_x_distance = [
                (o_entity, np.linalg.norm(entity["vector"] - o_entity["vector"],ord= 1))
                for o_entity in tagset
            ]
        elif t=='l2':
            ontology_x_distance = [
                (o_entity, np.linalg.norm(entity["vector"] - o_entity["vector"],ord= 2))
                for o_entity in tagset
            ]
        else:
            ontology_x_distance = [
                (o_entity, 1-cosine_similarity([entity["vector"]],[o_entity["vector"]]))
                for o_entity in tagset
            ]
        ontology_x_distance.sort(key=lambda x: x[1])
        ontology_x_distance = ontology_x_distance[:k]
        entity["predicted_labels"] = [
            o_entity["label"] for o_entity, _ in ontology_x_distance
        ]
        del entity["vector"]
    return data

def get_y_list(data_temp,col):
    y= []
    for term in data_temp:
        y.append(term[col])
    return y


k= 5
dim= 300
train_test= 'train'
data_path= 'data/terms/'+ train_test +'_class_in_terms.json'
tagset= 'data/tagset/finsim.json'
embeddings_path= f'models/custom_w2v_{dim}d.txt'

model = load_glove_model(embeddings_path)
data= json.load(open(data_path, "r"))
tagset = json.load(open(tagset, "r"))

data, tagset= get_vectorized_data(data,tagset,model,dim)
predicted_data= get_prediction(data,tagset,t='cosine')


if train_test=='train':
    accuracy = compute_accuracy(get_y_list(predicted_data,'label'), get_y_list(predicted_data,'predicted_labels'))
    average_rank = compute_average_rank(get_y_list(predicted_data,'label'), get_y_list(predicted_data,'predicted_labels'))
    print("Accuracy: ", accuracy)
    print("Average Rank: ", average_rank)
    
else:        
    data_final= []
    for idx, example in enumerate(predicted_data):
        data_final.append({"term": example["term"], "label": example["label"], "predicted_labels": example["predicted_labels"]})
    
    json.dump(data_final, open("data/outputs/prediction_class_in_terms.json", "w"), indent=4)
