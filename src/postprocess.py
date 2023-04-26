"""
Postprocessing.
"""

import numpy as np
import pandas as pd


def preds2recs(preds, item_mapping=None):

    user_ids = np.hstack([pred['user_ids'] for pred in preds])
    scores = np.vstack([pred['scores'] for pred in preds])
    preds = np.vstack([pred['preds'] for pred in preds])

    user_ids = np.repeat(user_ids[:, None], repeats=scores.shape[1], axis=1)

    recs = pd.DataFrame({'user_id': user_ids.flatten(),
                         'item_id': preds.flatten(),
                         'prediction': scores.flatten()})

    if item_mapping is not None:
        recs.item_id = recs.item_id.map(item_mapping)

    return recs
