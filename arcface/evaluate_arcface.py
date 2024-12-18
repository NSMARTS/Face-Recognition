import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.utils import l2_norm, load_yaml
from networks.recognition.models import ArcFaceModel


def get_embedding(model, x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)
    embeds = model.predict(x)
    embeds = l2_norm(embeds)
    return embeds


def calculate_accuracy(threshold, dist, actual_is_same):
    predict_is_same = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_is_same, actual_is_same))
    fp = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_is_same)))
    tn = np.sum(np.logical_and(np.logical_not(predict_is_same), np.logical_not(actual_is_same)))
    fn = np.sum(np.logical_and(np.logical_not(predict_is_same), actual_is_same))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    # precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    tprs = np.zeros((len(thresholds)))
    fprs = np.zeros((len(thresholds)))
    accuracy = np.zeros((len(thresholds)))
    # precision = np.zeros((len(thresholds)))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), axis=1)

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist, actual_is_same)
    best_threshold_idx = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_idx]

    return tprs, fprs, accuracy, best_threshold


def evaluate(embeddings1, embeddings2, actual_is_same):
    thresholds = np.arange(0, 4, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embeddings1, embeddings2, actual_is_same)
    tpr = np.mean(tprs)
    fpr = np.mean(fprs)
    accuracy = np.max(accuracy)
    return tpr, fpr, accuracy, best_threshold


if __name__ == '__main__':
    import os
    import cv2
    from sklearn.metrics import auc
    
    # # parser = argparse.ArgumentParser(description='테스트')
    # # # parser.add_argument("--mode", type=str)
    # # parser.add_argument("--weights", type=str)
    # # args = parser.parse_args()

    # arcface = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    # ckpt_path = tf.train.latest_checkpoint(
    #     '/checkpoints/arc_res50_kface_finetune-lr0.001-bs8-trainable2\e_50_b_1070.ckpt.data-00001-of-00002'
    #     )
    # arcface.load_weights(ckpt_path)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", type=str)
    # args = parser.parse_args()

    print('=============================')
    # print(args)

    arcface = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint('checkpoints/arc_res50_kface_finetune-lr0.005-bs32-trainable2/')
    print('ckpt_path : ', ckpt_path)
    arcface.load_weights(ckpt_path)

    data_path = "data/kface_val_npy"

    ref = np.load(os.path.join(data_path, "references.npy")).astype(np.float32) / 255.
    queries = np.load(os.path.join(data_path, "queries.npy")).astype(np.float32) / 255.
    is_same = np.load(os.path.join(data_path, "is_same.npy"))

    print(ref.shape, queries.shape)

    embeds1 = get_embedding(arcface, ref)
    embeds2 = get_embedding(arcface, queries)
    print(embeds1.shape, embeds2.shape)

    # tpr, fpr, accuracy, best_threshold = evaluate(embeds1, embeds2, is_same)
    thresholds = np.arange(0, 3, 0.01)
    tprs, fprs, accuracy, best_threshold = calculate_roc(thresholds, embeds1, embeds2, is_same)
    accuracy = np.max(accuracy)
    print(f"Maximum Accuracy: {float(accuracy):.4f}, Best Threshold: {best_threshold}")

    plt.figure(figsize=(5, 5))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fprs, tprs, label=f"ROC Curve (AUC = {auc(fprs, tprs):.4f}")
    plt.legend()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.show()

