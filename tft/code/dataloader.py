#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: xyshu
# @Time: 2023-08-06 09:47
# @File: dataset.py
# @Description: implementation of TFT's Dataloader

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class WebTrafficGenerator:
    def __init__(self, start_date='2021-01-01', end_date='2024-12-31', trend_base=0.5,
                 weekly_seasonality=None, yearly_seasonality=None, noise_multiplier=10):
        self.dates = dates = pd.date_range(start=start_date, end=end_date, freq='D')
        self.trend_base = trend_base
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.noise_multiplier = noise_multiplier
        self.web_traffic = []

    def generate_data(self):

        day = 24 * 60 * 60
        week = day * 7
        year = 365.2425 * day

        if self.yearly_seasonality:
            yearly = ((1 + np.sin(
                self.dates.view('int64') // 1e9 * (self.yearly_seasonality * np.pi / year))) * 100).astype(int)
        else:
            yearly = 0

        if self.weekly_seasonality:
            weekly = ((1 + np.sin(
                self.dates.view('int64') // 1e9 * (self.weekly_seasonality * np.pi / week))) * 10).astype(int)
        else:
            weekly = 0

        trend = np.array(range(len(self.dates))) * self.trend_base
        noise = ((np.random.random(len(self.dates)) - 0.5) * self.noise_multiplier).astype(int)

        return trend + yearly + weekly + noise


def fit_preprocessing(train, real_columns, categorical_columns):
    real_scalers = StandardScaler().fit(train[real_columns].values)

    categorical_scalers = {}
    num_classes = []
    for col in categorical_columns:
        srs = train[col].apply(str)
        categorical_scalers[col] = LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique())

    return real_scalers, categorical_scalers


def transform_inputs(df, real_scalers, categorical_scalers, real_columns, categorical_columns):
    out = df.copy()
    out[real_columns] = real_scalers.transform(df[real_columns].values)

    for col in categorical_columns:
        string_df = df[col].apply(str)
        out[col] = categorical_scalers[col].transform(string_df)

    return out


def get_data():
    entities = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    groups = {"High": ["A", "B", "C", "D", "E"], "Medium": ["F", "G", "H", "I", "J", "K"],
              "Low": ["L", "M", "N", "O", "P"]}

    raw = pd.DataFrame(
        columns=['date', 'traffic', 'Entity', 'Class', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month', 'Year'])

    for e in entities:
        trend_base = np.round(np.random.random(), 2)

        if e in groups["High"]:
            trend_base *= 1.5
            group = "High"
        elif e in groups["Low"]:
            trend_base *= 0.7
            group = "Low"
        else:
            group = "Medium"

        traffic_generator = WebTrafficGenerator(start_date='2020-01-01',
                                                end_date='2024-12-31',
                                                trend_base=trend_base,
                                                weekly_seasonality=0.7 + np.round(np.random.uniform(0, 2.0), 2),
                                                yearly_seasonality=6.0 + np.round(np.random.uniform(0, 2.0), 2),
                                                noise_multiplier=80)

        traffic = traffic_generator.generate_data()
        tmp = pd.DataFrame(data={"date": traffic_generator.dates, "traffic": traffic})
        tmp["Entity"] = [e] * tmp.shape[0]
        tmp["Class"] = [group] * tmp.shape[0]
        tmp["DayOfWeek"] = tmp.date.dt.dayofweek
        tmp["DayOfMonth"] = tmp.date.dt.day
        tmp["WeekOfYear"] = tmp.date.dt.isocalendar().week
        tmp["Month"] = tmp.date.dt.month
        tmp["Year"] = tmp.date.dt.year
        tmp["DaysFromStart"] = np.arange(tmp.shape[0])
        tmp["Delta"] = tmp["traffic"].diff().fillna(0)

        raw = pd.concat([raw, tmp])

    raw.reset_index(inplace=True)
    # total = raw.groupby("date")["traffic"].sum()
    # plt.plot(total.index, total)
    # plt.xticks(rotation=90);
    # plt.show()

    train = raw[raw['date'] < '2023-01-01']
    valid = raw.loc[(raw['date'] >= '2023-01-01') & (raw['date'] < '2024-01-01')]
    test = raw.loc[(raw['date'] > '2024-01-01')]

    real_columns = ['traffic', "Delta", 'DaysFromStart']
    categorical_columns = ['Entity', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month', 'Class']

    real_scalers, categorical_scalers = fit_preprocessing(train, real_columns, categorical_columns)

    train = transform_inputs(train, real_scalers, categorical_scalers, real_columns, categorical_columns)
    # valid = transform_inputs(valid, real_scalers, categorical_scalers, real_columns, categorical_columns)
    test = transform_inputs(test, real_scalers, categorical_scalers, real_columns, categorical_columns)

    return train, test


def draw_traffic():
    traffic_generator = WebTrafficGenerator(start_date='2021-01-01',
                                            end_date='2024-12-31',
                                            trend_base=0.5,
                                            weekly_seasonality=0.7,
                                            yearly_seasonality=2.3,
                                            noise_multiplier=80)
    traffic = traffic_generator.generate_data()

    plt.plot(traffic_generator.dates, traffic)
    plt.xticks(rotation=90);
    plt.show()


class TFT_Dataset(Dataset):
    def __init__(self, data, train_column_unique, entity_column, time_column, target_column,
                 input_columns, encoder_steps, decoder_steps):
        """
          data (pd.DataFrame): dataframe containing raw data
          entity_column (str): name of column containing entity data
          time_column (str): name of column containing date data
          target_column (str): name of column we need to predict
          input_columns (list): list of string names of columns used as input
          encoder_steps (int): number of known past time steps used for forecast. Equivalent to size of LSTM encoder
          decoder_steps (int): number of input time steps used for each forecast date. Equivalent to the width N of the decoder
        """

        self.encoder_steps = encoder_steps

        inputs = []
        outputs = []
        entity = []
        time = []
        # print(train[entity_column].unique())
        for e in train_column_unique:
            entity_group = data[data[entity_column] == e]

            data_time_steps = len(entity_group)

        if data_time_steps >= decoder_steps:
            x = entity_group[input_columns].values.astype(np.float32)
            inputs.append(
                np.stack([x[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))

            y = entity_group[[target_column]].values.astype(np.float32)
            outputs.append(
                np.stack([y[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))

            e = entity_group[[entity_column]].values.astype(np.float32)
            entity.append(
                np.stack([e[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))

            t = entity_group[[time_column]].values.astype(np.int64)
            time.append(
                np.stack([t[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))

        self.inputs = np.concatenate(inputs, axis=0)
        self.outputs = np.concatenate(outputs, axis=0)[:, encoder_steps:, :]
        self.entity = np.concatenate(entity, axis=0)
        self.time = np.concatenate(time, axis=0)
        self.active_inputs = np.ones_like(outputs)

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs[:, self.encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[:, self.encoder_steps:, :]),
            'time': self.time,
            'identifier': self.entity
        }

    def __getitem__(self, index):
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
            'active_entries': np.ones_like(self.outputs[index]),
            'time': self.time[index],
            'identifier': self.entity[index]
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]


if __name__ == "__main__":
    BATCH_SIZE = 32
    ENCODER_STEPS = 175
    DECODER_STEPS = ENCODER_STEPS + 5

    draw_traffic()
    train, test = get_data()

    # Dataset variables
    input_columns = ["traffic", "Delta", 'DaysFromStart', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Month', 'Entity',
                     'Class', 'Entity']
    target_column = "traffic"
    entity_column = "Entity"
    time_column = "date"
    col_to_idx = {col: idx for idx, col in enumerate(input_columns)}
    training_data = TFT_Dataset(train, train[entity_column].unique(), entity_column, time_column, target_column,
                                input_columns, ENCODER_STEPS, DECODER_STEPS)
    testing_data = TFT_Dataset(test, train[entity_column].unique(), entity_column, time_column, target_column,
                               input_columns, ENCODER_STEPS, DECODER_STEPS)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
