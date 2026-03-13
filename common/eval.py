import numpy as np


def build_user_item_dict(df, user_col="user_id", item_col="movie_id"):
    user_items = {}

    for user, item in zip(df[user_col], df[item_col]):
        user_items.setdefault(user, set()).add(item)

    return user_items


def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    hits = len(set(recommended) & relevant)

    return hits / k


def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    hits = len(set(recommended) & relevant)

    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0.0

    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_at_k(recommended, relevant, k):
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1 / (i + 1)

    return 0.0


def hitrate_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    return int(len(set(recommended) & relevant) > 0)


def evaluate_ranking(recommendations, ground_truth, k=10):

    precisions = []
    recalls = []
    maps = []
    ndcgs = []
    mrrs = []
    hitrates = []

    for user, relevant in ground_truth.items():

        if user not in recommendations:
            continue

        recommended = recommendations[user]

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        maps.append(average_precision_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        mrrs.append(mrr_at_k(recommended, relevant, k))
        hitrates.append(hitrate_at_k(recommended, relevant, k))

    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "map": np.mean(maps),
        "ndcg": np.mean(ndcgs),
        "mrr": np.mean(mrrs),
        "hitrate": np.mean(hitrates),
    }