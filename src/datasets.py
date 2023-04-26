"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LMDataset(Dataset):

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=True,
                 user_col='user_id', item_col='item_id', time_col='time_idx'):

        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):

        return len(self.data)

    def sample_negatives(self, item_sequence):

        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=False)

        return negatives


class CausalLMDataset(LMDataset):

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=True,
                 user_col='user_id', item_col='item_id', time_col='time_idx',
                 label_masking_probability=0):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.label_masking_probability = label_masking_probability

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length + 1:
            item_sequence = item_sequence[-self.max_length - 1:]

        input_ids = np.array(item_sequence[:-1])
        labels = np.array(item_sequence[1:])

        # for testing how masking labels influence performance
        if self.label_masking_probability > 0:
            mask = np.random.rand(len(labels)) < self.label_masking_probability
            labels[mask] = -100

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class CausalLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='time_idx'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length-1:-1]
            item_sequence = item_sequence[:-1]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            input_ids = item_sequence[-self.max_length:]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}


class MaskedLMDataset(LMDataset):

    def __init__(self, df, max_length=128,
                 num_negatives=None, full_negative_sampling=True,
                 mlm_probability=0.2,
                 masking_value=1, ignore_value=-100,
                 force_last_item_masking_prob=0,
                 user_col='user_id', item_col='item_id',
                 time_col='time_idx'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length:
            item_sequence = item_sequence[-self.max_length:]

        input_ids = np.array(item_sequence)
        mask = np.random.rand(len(item_sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value

        labels = np.array(item_sequence)
        labels[input_ids != self.masking_value] = self.ignore_value

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class MaskedLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='time_idx'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length:-1]
            item_sequence = item_sequence[:-1]
        else:
            input_ids = item_sequence[-self.max_length + 1:]

        input_ids += [self.masking_value]

        if self.validation_mode:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}


class PaddingCollateFn:

    def __init__(self, padding_value=0, labels_padding_value=-100):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key == 'labels':
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)  

        return collated_batch
