"""
Run experiment.
"""

import time
import os

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch.utils.data import DataLoader

from datasets import (CausalLMDataset, CausalLMPredictionDataset,
                      MaskedLMDataset, MaskedLMPredictionDataset,
                      PaddingCollateFn)
from metrics import compute_metrics, compute_sampled_metrics
from models import RNN, BERT4Rec, SASRec
from modules import SeqRec, SeqRecWithSampling
from postprocess import preds2recs
from preprocess import add_time_idx


@hydra.main(version_base=None, config_path="configs", config_name="SASRec")
def main(config):

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    train, validation, validation_full, test, item_count = prepare_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation_full, config)
    model = create_model(config, item_count=item_count)
    start_time = time.time()
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    print('training_time', training_time)

    recs_validation, validation_dataset = predict(trainer, seqrec_module, train, config)
    evaluate(recs_validation, validation, train, seqrec_module,
             validation_dataset, task, config, prefix='val')

    recs_test, test_dataset = predict(trainer, seqrec_module, validation_full, config)
    evaluate(recs_test, test, train, seqrec_module,
             test_dataset, task, config, prefix='test')

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)
        task.close()


def prepare_data(config):

    data = pd.read_csv(config.data_path, sep=' ', header=None, names=['user_id', 'item_id'])
    data = add_time_idx(data, sort=False)

    # index 1 is used for masking value
    if config.model == 'BERT4Rec':
        data.item_id += 1

    train = data[data.time_idx_reversed >= 2]
    validation = data[data.time_idx_reversed == 1]
    validation_full = data[data.time_idx_reversed >= 1]
    test = data[data.time_idx_reversed == 0]

    item_count = data.item_id.max()

    return train, validation, validation_full, test, item_count


def create_dataloaders(train, validation, config):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]

    if config.model in ['SASRec', 'GPT4Rec', 'RNN']:
        train_dataset = CausalLMDataset(train, **config['dataset'])
        eval_dataset = CausalLMPredictionDataset(
            validation, max_length=config.dataset.max_length, validation_mode=True)
    elif config.model == 'BERT4Rec':
        train_dataset = MaskedLMDataset(train, **config['dataset'])
        eval_dataset = MaskedLMPredictionDataset(
            validation, max_length=config.dataset.max_length, validation_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, item_count):

    if hasattr(config.dataset, 'num_negatives') and config.dataset.num_negatives:
        add_head = False
    else:
        add_head = True

    if config.model == 'SASRec':
        model = SASRec(item_num=item_count, add_head=add_head, **config.model_params)
    if config.model == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, add_head=add_head,
                         bert_config=config.model_params)
    elif config.model == 'GPT4Rec':
        model = GPT4Rec(vocab_size=item_count + 1, add_head=add_head,
                        gpt_config=config.model_params)
    elif config.model == 'RNN':
        model = RNN(vocab_size=item_count + 1, add_head=add_head,
                    rnn_config=config.model_params)

    return model


def training(model, train_loader, eval_loader, config):

    if hasattr(config.dataset, 'num_negatives') and config.dataset.num_negatives:
        seqrec_module = SeqRecWithSampling(model, **config['seqrec_module'])
    else:
        seqrec_module = SeqRec(model, **config['seqrec_module'])

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, gpus=1, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)
    
    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module


def predict(trainer, seqrec_module, data, config):

    if config.model in ['SASRec', 'GPT4Rec', 'RNN']:
        predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset.max_length)
    elif config.model  == 'BERT4Rec':
        predict_dataset = MaskedLMPredictionDataset(data, max_length=config.dataset.max_length)

    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.top_k_metrics)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)

    return recs, predict_dataset


def evaluate(recs, test, train, seqrec_module, dataset, task, config, prefix='test'):

    all_metrics = {}
    for k in config.top_k_metrics:
        metrics = compute_metrics(test, recs, k=k)
        metrics = {prefix + '_' + key: value for key, value in metrics.items()}
        print(metrics)
        all_metrics.update(metrics)

    if config.sampled_metrics:
        item_counts = train.item_id.value_counts()

        uniform_metrics = compute_sampled_metrics(seqrec_module, dataset, test, item_counts,
                                                popularity_sampling=False, num_negatives=100, k=10)
        uniform_metrics = {prefix + '_' + key + '_uniform': value
                        for key, value in uniform_metrics.items()}
        print(uniform_metrics)

        popularity_metrics = compute_sampled_metrics(seqrec_module, dataset, test, item_counts,
                                                    num_negatives=100, k=10)
        popularity_metrics = {prefix + '_' + key + '_popularity': value
                            for key, value in popularity_metrics.items()}
        print(popularity_metrics)

    if task:

        clearml_logger = task.get_logger()

        for key, value in all_metrics.items():
            clearml_logger.report_single_value(key, value)
        if config.sampled_metrics:
            for key, value in uniform_metrics.items():
                clearml_logger.report_single_value(key, value)
            for key, value in popularity_metrics.items():
                clearml_logger.report_single_value(key, value)

        if config.sampled_metrics:
            all_metrics.update(uniform_metrics)
            all_metrics.update(popularity_metrics)
        all_metrics = pd.Series(all_metrics).to_frame().reset_index()
        all_metrics.columns = ['metric_name', 'metric_value']

        clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                    table_plot=all_metrics)
        task.upload_artifact(f'{prefix}_metrics', all_metrics)


if __name__ == "__main__":

    main()
