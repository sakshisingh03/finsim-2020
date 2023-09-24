import numpy as np
from sklearn import metrics


def get_rank(gold, list_predictions, max_k=3):
    list_predictions = list_predictions[:max_k]
    try:
        rank = list_predictions.index(gold) + 1
    except ValueError:
        rank = max_k + 1
    return rank


def compute_average_rank(Y_test, predictions):
    ranks = []
    for idx, y in enumerate(Y_test):
        ranks.append(get_rank(Y_test[idx], predictions[idx]))
    return np.mean(ranks)


def compute_accuracy(Y_test, predictions):
    predictions = [prediction[0] for prediction in predictions]
    return metrics.accuracy_score(Y_test, predictions)
