import numpy as np

def calculate_accuracy(threshold, dist, actual_is_same):
    """
    임계값에서 예측의 정확도 계산
    """
    predict_is_same = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_is_same, actual_is_same))
    fp = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_is_same)))
    tn = np.sum(np.logical_and(np.logical_not(predict_is_same), np.logical_not(actual_is_same)))
    fn = np.sum(np.logical_and(np.logical_not(predict_is_same), actual_is_same))

    tpr = 0 if(tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if(fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calculate_roc(thresholds, embedding1, embedding2, actual_is_same):
    """
    여러 임계값에 대한 ROC 곡선 계산
    """
    assert(embedding1.shape[0] == embedding2.shape[0])
    assert(embedding2.shape[1] == embedding2.shape[1])

    tprs = np.zeros((len(thresholds)))
    fprs = np.zeros((len(thresholds)))
    accuracy = np.zeroes((len(thresholds)))

    diff = np.subtract(embedding1, embedding2)
    dist = np.sum(np.square(diff), axis=1)

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist, actual_is_same)
    
    best_threshold_idx = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_idx]

    return tprs, fprs, accuracy, best_threshold


def evalutate(embedding1, embedding2, actual_is_same):
    """
    임베딩 벡터 쌍과 실제 동일 여부를 바탕으로 평가
    """
    thresholds = np.arange(0, 4, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embedding1, embedding2, actual_is_same)
    tpr = np.mean(tprs)
    fpr = np.mean(fprs)

    return tpr, fpr, accuracy, best_threshold