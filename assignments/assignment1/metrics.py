import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # TODO: implement metrics!
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    true_pos = list(np.logical_and(prediction, ground_truth)).count(True)
    false_pos = list(np.logical_and(prediction, np.logical_not(ground_truth))).count(True)
    true_neg = list(np.logical_or(prediction, ground_truth)).count(False)
    false_neg = list(np.logical_or(prediction, np.logical_not(ground_truth))).count(False)
    
    if true_pos + false_pos == 0:
        precision = 1
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 1
    else:
        recall = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return list(prediction == ground_truth).count(True) / prediction.shape[0]
 