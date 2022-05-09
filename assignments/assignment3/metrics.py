def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!

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

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return list(prediction == ground_truth).count(True) / prediction.shape[0]
