"""
Metrics.
"""

import numpy as np
import torch
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, recall_at_k
from tqdm.auto import tqdm


def compute_metrics(ground_truth, preds, k=10):

    if not hasattr(ground_truth, 'rating'):
        ground_truth = ground_truth.assign(rating=1)

    # when we have 1 true positive, HitRate == Recall and MRR == MAP
    metrics = {
        f'ndcg@{k}': ndcg_at_k(ground_truth, preds, col_user='user_id', col_item='item_id',
                               col_prediction='prediction', col_rating='rating', k=k),
        f'hit_rate@{k}': recall_at_k(ground_truth, preds, col_user='user_id', col_item='item_id',
                                     col_prediction='prediction', col_rating='rating', k=k),
        f'mrr@{k}': map_at_k(ground_truth, preds, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating='rating', k=k)
    }

    return metrics


def compute_sampled_metrics(seqrec_module, predict_dataset, test, item_counts,
                            popularity_sampling=True, num_negatives=100, k=10,
                            device='cuda'):

    test = test.set_index('user_id')['item_id'].to_dict()
    all_items = item_counts.index.values
    item_weights = item_counts.values
    # probabilities = item_weights/item_weights.sum()

    seqrec_module = seqrec_module.eval().to(device)

    ndcg, hit_rate, mrr = 0.0, 0.0, 0.0
    user_count = 0

    for user in tqdm(predict_dataset):

        if user['user_id'] not in test:
            continue

        positive = test[user['user_id']]
        indices = ~np.isin(all_items, user['full_history'])
        negatives = all_items[indices]
        if popularity_sampling:
            probabilities = item_weights[indices]
            probabilities = probabilities/probabilities.sum()
        else:
            probabilities = None
        negatives = np.random.choice(negatives, size=num_negatives,
                                     replace=False, p=probabilities)
        items = np.concatenate([np.array([positive]), negatives])

        # code from BERT4Rec original repo https://github.com/FeiSun/BERT4Rec/blob/master/run.py#L195
        # items = [test[user['user_id']]]
        # while len(items) < num_negatives + 1:
        #     sampled_ids = np.random.choice(all_items, num_negatives + 1, replace=False, p=probabilities)
        #     sampled_ids = [x for x in sampled_ids if x not in user['full_history'] and x not in items]
        #     items.extend(sampled_ids[:])
        # items = items[:num_negatives + 1]

        batch = {'input_ids': torch.tensor(user['input_ids']).unsqueeze(0).to(device),
                 'attention_mask': torch.tensor([1] * len(user['input_ids'])).unsqueeze(0).to(device)}
        pred = seqrec_module.prediction_output(batch)
        pred = pred[0, -1, items]

        rank = (-pred).argsort().argsort()[0].item() + 1
        if rank <= k:
            ndcg += 1 / np.log2(rank + 1)
            hit_rate += 1
            mrr += 1 / rank
        user_count += 1

    ndcg = ndcg / user_count
    hit_rate = hit_rate / user_count
    mrr = mrr / user_count

    return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}
