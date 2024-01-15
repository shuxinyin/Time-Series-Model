#!/usr/bin/env python
# coding: utf-8

import pickle
from typing import Dict, List, Tuple, Any, Generator
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
from utils import weight_init, DictDataSet, EarlyStopping, QueueAggregator

# ### Data-related
data_path = '../data/grocery_sales/data.pickle'

with open(data_path, 'rb') as fp:
    data = pickle.load(fp)
print(list(data.keys()))

for set_name in data['data_sets']:
    print(f'======={set_name}=======')
    for arr_name, arr in data['data_sets'][set_name].items():
        print(f"{arr_name} (shape,dtype)")
        print(arr.shape, arr.dtype)

# ### Modeling configuration
configuration = {'optimization':
    {
        'batch_size': {'training': 256, 'inference': 4096},
        'learning_rate': 0.001,
        'max_grad_norm': 1.0,
    }
    ,
    'model':
        {
            'dropout': 0.05,
            'state_size': 64,
            'output_quantiles': [0.1, 0.5, 0.9],
            'lstm_layers': 2,
            'attention_heads': 4
        },
    # these arguments are related to possible extensions of the model class
    'task_type': 'regression',
    'target_window_start': None
}

feature_map = data['feature_map']
cardinalities_map = data['categorical_cardinalities']

structure = {
    'num_historical_numeric': len(feature_map['historical_ts_numeric']),
    'num_historical_categorical': len(feature_map['historical_ts_categorical']),
    'num_static_numeric': len(feature_map['static_feats_numeric']),
    'num_static_categorical': len(feature_map['static_feats_categorical']),
    'num_future_numeric': len(feature_map['future_ts_numeric']),
    'num_future_categorical': len(feature_map['future_ts_categorical']),
    'historical_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in
                                             feature_map['historical_ts_categorical']],
    'static_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in
                                         feature_map['static_feats_categorical']],
    'future_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['future_ts_categorical']],
}

configuration['data_props'] = structure
model = TemporalFusionTransformer(config=OmegaConf.create(configuration))
model.apply(weight_init)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
model.to(device)

opt = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                 lr=configuration['optimization']['learning_rate'])


# ### Data Preparation
def recycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_set_and_loaders(data_dict: Dict[str, np.ndarray],
                        shuffled_loader_config: Dict,
                        serial_loader_config: Dict,
                        ignore_keys: List[str] = None,
                        ) -> Tuple[DictDataSet, Generator[Any, Any, None], DataLoader[Any]]:
    dataset = DictDataSet({k: v for k, v in data_dict.items() if (ignore_keys and k not in ignore_keys)})
    loader = torch.utils.data.DataLoader(dataset, **shuffled_loader_config)
    serial_loader = torch.utils.data.DataLoader(dataset, **serial_loader_config)

    return dataset, iter(recycle(loader)), serial_loader


shuffled_loader_config = {'batch_size': configuration['optimization']['batch_size']['training'],
                          'drop_last': True,
                          'shuffle': True}

serial_loader_config = {'batch_size': configuration['optimization']['batch_size']['inference'],
                        'drop_last': False,
                        'shuffle': False}

# the following fields do not contain actual data, but are only identifiers of each observation
meta_keys = ['time_index', 'combination_id']

train_set, train_loader, train_serial_loader = get_set_and_loaders(data['data_sets']['train'],
                                                                   shuffled_loader_config,
                                                                   serial_loader_config,
                                                                   ignore_keys=meta_keys)
validation_set, validation_loader, validation_serial_loader = get_set_and_loaders(data['data_sets']['validation'],
                                                                                  shuffled_loader_config,
                                                                                  serial_loader_config,
                                                                                  ignore_keys=meta_keys)
test_set, test_loader, test_serial_loader = get_set_and_loaders(data['data_sets']['test'],
                                                                shuffled_loader_config,
                                                                serial_loader_config,
                                                                ignore_keys=meta_keys)

# ### Training Settings
max_epochs = 10000
epoch_iters = 200
eval_iters = 500
log_interval = 20
# the running-window  saved for the training performance
ma_queue_size = 50
patience = 5

# initialize early stopping mechanism
es = EarlyStopping(patience=patience)
# initialize the loss aggregator for running window performance estimation
loss_aggregator = QueueAggregator(max_size=ma_queue_size)

# initialize counters
batch_idx = 0
epoch_idx = 0

quantiles_tensor = torch.tensor(configuration['model']['output_quantiles']).to(device)


def process_batch(batch: Dict[str, torch.tensor],
                  model: nn.Module,
                  quantiles_tensor: torch.tensor,
                  device: torch.device):
    if is_cuda:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch['target']

    predicted_quantiles = batch_outputs['predicted_quantiles']
    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_quantiles,
                                                               targets=labels,
                                                               desired_quantiles=quantiles_tensor)
    return q_loss, q_risk


while epoch_idx < max_epochs:
    print(f"Starting Epoch Index {epoch_idx}")

    # evaluation round
    model.eval()
    with torch.no_grad():
        # for each subset
        for subset_name, subset_loader in zip(['train', 'validation', 'test'],
                                              [train_loader, validation_loader, test_loader]):
            print(f"Evaluating {subset_name} set")

            q_loss_vals, q_risk_vals = [], []  # used for aggregating performance along the evaluation round
            for _ in range(eval_iters):
                # get batch
                batch = next(subset_loader)
                # process batch
                batch_loss, batch_q_risk = process_batch(batch=batch, model=model, quantiles_tensor=quantiles_tensor,
                                                         device=device)
                # accumulate performance
                q_loss_vals.append(batch_loss)
                q_risk_vals.append(batch_q_risk)

            # aggregate and average
            eval_loss = torch.stack(q_loss_vals).mean(axis=0)
            eval_q_risk = torch.stack(q_risk_vals, axis=0).mean(axis=0)

            # keep for feeding the early stopping mechanism
            if subset_name == 'validation':
                validation_loss = eval_loss

            # log performance
            print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx}" + \
                  f"- Eval {subset_name} - " + \
                  f"q_loss = {eval_loss:.5f} , " + \
                  " , ".join([f"q_risk_{q:.1} = {risk:.5f}" for q, risk in zip(quantiles_tensor, eval_q_risk)]))

    # switch to training mode
    model.train()

    # update early stopping mechanism and stop if triggered
    if es.step(validation_loss):
        print('Performing early stopping...!')
        break

    # initiating a training round
    for _ in range(epoch_iters):
        # get training batch
        batch = next(train_loader)

        opt.zero_grad()
        # process batch
        loss, _ = process_batch(batch=batch,
                                model=model,
                                quantiles_tensor=quantiles_tensor,
                                device=device)
        # compute gradients
        loss.backward()
        # gradient clipping
        if configuration['optimization']['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), configuration['optimization']['max_grad_norm'])
        # update weights
        opt.step()

        # accumulate performance
        loss_aggregator.append(loss.item())

        # log performance
        if batch_idx % log_interval == 0:
            print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx} - Train Loss = {np.mean(loss_aggregator.get())}")

        # completed batch
        batch_idx += 1

    # completed epoch
    epoch_idx += 1
