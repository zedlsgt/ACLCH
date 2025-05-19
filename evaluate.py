import numpy as np


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def fx_calc_map_label(qB, rB, query_L, retrieval_L, k=50, rank=0):
    num_query = query_L.shape[0]
    map_score = 0
    if k is None:
        k = retrieval_L.shape[0]

    for iter in range(num_query):
        gnd = (np.dot(query_L[iter], retrieval_L.T) > 0).astype(float)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = np.arange(1, total + 1).astype(float)
        tindex = np.nonzero(gnd)[0][:total].astype(float) + 1.0
        map_score += np.mean(count / tindex)

    map_score = map_score / num_query
    return map_score


def calc_topk_precision(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)

    for iter in range(num_query):
        gnd = np.dot(query_label[iter], retrieval_label.T) > 0
        tsum = np.sum(gnd)
        if tsum == 0:
            continue

        hamm = CalcHammingDist(qB[iter, :], rB)

        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = np.argsort(hamm)[:total]
            gnd_ = gnd[ind]
            p[i] += np.sum(gnd_) / total

    p = np.array(p) / num_query
    return p
