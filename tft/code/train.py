#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: xyshu
# @Time: 2023-08-06 11:07
# @File: train.py
# @Description:

import sys
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import seaborn as sns
import matplotlib.pyplot as plt

from quantile_loss import QuantileLoss
from model import TemporalFusionTransformer
from dataloader import TFT_Dataset, get_data

# Global variables

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
ENCODER_STEPS = 175
DECODER_STEPS = ENCODER_STEPS + 5
HIDDEN_LAYER_SIZE = 80
EMBEDDING_DIMENSION = 8
NUM_LSTM_LAYERS = 1
NUM_ATTENTION_HEADS = 2
QUANTILES = [0.1, 0.5, 0.9]

# Dataset variables
input_columns = ["traffic", "Delta", 'DaysFromStart', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month', 'Entity',
                 'Class', 'Entity']
target_column = "traffic"
entity_column = "Entity"
time_column = "date"
col_to_idx = {col: idx for idx, col in enumerate(input_columns)}

train, test = get_data()
print(train.columns)
print(train.head())

training_data = TFT_Dataset(train, train[entity_column].unique(), entity_column, time_column, target_column,
                            input_columns, ENCODER_STEPS, DECODER_STEPS)
testing_data = TFT_Dataset(test, train[entity_column].unique(), entity_column, time_column, target_column,
                           input_columns, ENCODER_STEPS, DECODER_STEPS)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

params = {
    "quantiles": QUANTILES,
    "batch_size": BATCH_SIZE,
    "dropout": DROPOUT,
    "device": DEVICE,
    "hidden_layer_size": HIDDEN_LAYER_SIZE,
    "num_lstm_layers": NUM_LSTM_LAYERS,
    "embedding_dim": EMBEDDING_DIMENSION,
    "encoder_steps": ENCODER_STEPS,
    "num_attention_heads": NUM_ATTENTION_HEADS,
    "col_to_idx": col_to_idx,
    "static_covariates": ['Class', 'Entity'],  # 静态变量
    "time_dependent_categorical": ['DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month'],  #
    "time_dependent_continuous": ['traffic', 'DaysFromStart', "Delta"],
    "category_counts": {"DayOfWeek": 7, "DayOfMonth": 31, "WeekOfYear": 53, "Month": 12, "Class": 3, "Entity": 16},
    "known_time_dependent": ['DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month', 'DaysFromStart'],
    "observed_time_dependent": ["traffic", "Delta"]
}
model = TemporalFusionTransformer(params)
model.to(DEVICE)

criterion = QuantileLoss(QUANTILES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print_every_k = 100
losses = []

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    print(f"===== Epoch {epoch + 1} =========")
    epoch_loss = 0.0
    running_loss = 0.0

    for i, batch in enumerate(train_dataloader):
        labels = batch['outputs'][:, :, 0].flatten().float().to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs, attention_weights = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()

        if (i + 1) % print_every_k == 0:
            print(f"Mini-batch {i + 1} average loss: {round(running_loss / print_every_k, 5)}")
            running_loss = 0.0

    t1 = time.time()
    print(f"\nEpoch trained for {round(t1 - t0, 2)} seconds")
    print("\nEpoch loss:", round(epoch_loss / (i + 1), 5), "\n")
    losses.append(epoch_loss / (i + 1))


def plot_test(test_dataloader):
    out_df = pd.DataFrame(columns=['p10', 'p50', 'p90', 'identifier'])

    start_id = -1
    for i, batch in enumerate(test_dataloader):
        outputs, attention_weights = model(batch)
        bs = batch["outputs"].shape[0]

        process_map = {
            f"p{int(q * 100)}": outputs.reshape(bs, 5, 3)[:, :, i].cpu().detach().numpy()[::5, :].reshape(-1)[:bs] for
            i, q
            in enumerate(QUANTILES)}

        tmp = pd.DataFrame(data=process_map, index=pd.to_datetime(batch['time'][:, ENCODER_STEPS - 1, 0]))
        tmp["labels"] = batch["outputs"].reshape(-1)[::5]
        tmp["identifier"] = batch['identifier'][:, 0, 0]

        out_df = pd.concat([out_df, tmp])

        e = int(batch['identifier'][0, 0, 0].numpy())
        if batch['identifier'][0, 0, 0].numpy() != start_id:
            print("=" * 20)
            print(f"Plotting interpreation plots for a batch of entity {e}:")

            id_mask = batch['identifier'][:, 0, 0] == e

            # Plotting multi-head attention
            plt.figure(figsize=(15, 10))
            sns.lineplot(x=pd.to_datetime(batch["time"][0, :, 0].numpy()),
                         y=batch["inputs"][0, :, 0].numpy(), color="blue")
            ax2 = plt.twinx()
            sns.lineplot(x=pd.to_datetime(batch["time"][0, :, 0].numpy()),
                         y=attention_weights['multihead_attention'][0].cpu().detach().numpy()[:, 175:].mean(axis=1),
                         ax=ax2, color="orange")
            plt.show()

            past_inputs = ["day_of_week", "day_of_month", "week_of_year", "month", 'log_vol', 'days_from_start',
                           "open_to_close"]
            future_inputs = ["day_of_week", "day_of_month", "week_of_year", "month", "days_from_start"]

            # Plotting past weights
            plt.figure(figsize=(15, 4))
            sns.barplot(x=past_inputs,
                        y=attention_weights['past_weights'][id_mask, :, :].mean(dim=(0, 1)).cpu().detach().numpy(),
                        palette="crest")
            plt.show()

            # Plotting future weights
            plt.figure(figsize=(15, 4))
            sns.barplot(x=future_inputs,
                        y=attention_weights['future_weights'][id_mask, :, :].mean(dim=(0, 1)).cpu().detach().numpy(),
                        palette="crest")
            plt.show()
            start_id = e
            print()
