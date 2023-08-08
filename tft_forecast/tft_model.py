# coding=utf-8
"""
@Author: xyshu
@file:tft_model.py
@date: 2023/8/8 20:55
@description:
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss

from xehq_data_demo import data

max_prediction_length = 1
max_encoder_length = 3
print(data["time_idx"].max())
training_cutoff = data["time_idx"].max() - max_prediction_length

# # static input categories
# [ecif_no, occupation]

# # static input real_val
# [age, corp_score, income]

# # known input categories
# [discount_use]

# # known input real_val
# [discount_rate]

# # observed input categories
# [interest_free, num_installments, transaction_notes]

# # observed input real_val
# [transaction_amount, last_repayment_date, last_repayment_money]

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="transaction_amount",
    group_ids=["ecif_no"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["ecif_no", "occupation"],
    static_reals=["age", "corp_score", "income"],
    time_varying_known_categoricals=["discount_use", "month"],
    variable_groups={},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["discount_rate"],
    time_varying_unknown_categoricals=["interest_free", "num_installments", "transaction_notes"],
    time_varying_unknown_reals=["time_idx", "transaction_amount", "last_repayment_money"],
    target_normalizer="auto",  # use softplus and normalize by group
    # categorical_encoders={"add_nan": True},
    allow_missing_timesteps=True,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

for idx, data in val_dataloader:
    print(f"{idx}, {data}")
    break

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
# baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
# MAE()(baseline_predictions.output, baseline_predictions.y)


## Train model
# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=6,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(quantiles=[0.5]),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
