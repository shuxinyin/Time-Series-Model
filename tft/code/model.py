#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: xyshu
# @Time: 2023-08-06 10:15
# @File: model.py
# @Description: implementation of TFT model

import torch
import torch.nn as nn

from net_module import (VariableSelectionNetwork, GatedResidualNetwork,
                        TemporalLayer, InterpretableMultiHeadAttention, GLU)


class TemporalFusionTransformer(nn.Module):
    """Creates a Temporal Fusion Transformer model.

    For simplicity, arguments are passed within a parameters dictionary

    Args:
        col_to_idx (dict): Maps column names to their index in input array
        static_covariates (list): Names of static covariate variables
        time_dependent_categorical (list): Names of time dependent categorical variables
        time_dependent_continuous (list): Names of time dependent continuous variables
        category_counts (dict): Maps column names to the number of categories of each categorical feature
        known_time_dependent (list): Names of known time dependent variables
        observed_time_dependent (list): Names of observed time dependent variables
        batch_size (int): Batch size
        encoder_steps (int): Fixed k time steps to look back for each prediction (also size of LSTM encoder)
        hidden_size (int): Internal state size of different layers
        num_lstm_layers (int): Number of LSTM layers that should be used
        dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
        embedding_dim (int): Dimensionality of embeddings
        num_attention_heads (int): Number of heads for interpretable mulit-head attention
        quantiles (list): Quantiles used for prediction. Also defines model output size
        device (str): Used to decide between CPU and GPU

    """

    def __init__(self, parameters):
        """Uses the given parameters to set up the Temporal Fusion Transformer model

        Args:
          parameters: Dictionary with parameters used to define the model.
        """
        super().__init__()

        # Inputs
        self.col_to_idx = parameters["col_to_idx"]
        self.static_covariates = parameters["static_covariates"]
        self.time_dependent_categorical = parameters["time_dependent_categorical"]
        self.time_dependent_continuous = parameters["time_dependent_continuous"]
        self.category_counts = parameters["category_counts"]
        self.known_time_dependent = parameters["known_time_dependent"]
        self.observed_time_dependent = parameters["observed_time_dependent"]
        self.time_dependent = self.known_time_dependent + self.observed_time_dependent

        # Architecture
        self.batch_size = parameters['batch_size']
        self.encoder_steps = parameters['encoder_steps']
        self.hidden_size = parameters['hidden_layer_size']
        self.num_lstm_layers = parameters['num_lstm_layers']
        self.dropout = parameters['dropout']
        self.embedding_dim = parameters['embedding_dim']
        self.num_attention_heads = parameters['num_attention_heads']

        # Outputs
        self.quantiles = parameters['quantiles']

        # Other
        self.device = parameters['device']

        # Prepare for input transformation (embeddings for categorical variables and linear transformations for continuous variables)

        # Prepare embeddings for the static covariates and static context vectors
        self.static_embeddings = nn.ModuleDict(
            {col: nn.Embedding(self.category_counts[col], self.embedding_dim).to(self.device) for col in
             self.static_covariates})
        self.static_variable_selection = VariableSelectionNetwork(self.embedding_dim, len(self.static_covariates),
                                                                  self.hidden_size, self.dropout, is_temporal=False)

        self.static_context_variable_selection = GatedResidualNetwork(self.hidden_size, self.hidden_size,
                                                                      self.hidden_size, self.dropout, is_temporal=False)
        self.static_context_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                              self.dropout, is_temporal=False)
        self.static_context_state_h = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                           self.dropout, is_temporal=False)
        self.static_context_state_c = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                           self.dropout, is_temporal=False)

        # Prepare embeddings and linear transformations for time dependent variables
        self.temporal_cat_embeddings = nn.ModuleDict(
            {col: TemporalLayer(nn.Embedding(self.category_counts[col], self.embedding_dim)).to(self.device) for col in
             self.time_dependent_categorical})
        self.temporal_real_transformations = nn.ModuleDict(
            {col: TemporalLayer(nn.Linear(1, self.embedding_dim)).to(self.device) for col in
             self.time_dependent_continuous})

        # Variable selection and encoder for past inputs
        self.past_variable_selection = VariableSelectionNetwork(self.embedding_dim, len(self.time_dependent),
                                                                self.hidden_size, self.dropout,
                                                                context_size=self.hidden_size)

        # Variable selection and decoder for known future inputs
        self.future_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                                                  len([col for col in self.time_dependent if
                                                                       col not in self.observed_time_dependent]),
                                                                  self.hidden_size, self.dropout,
                                                                  context_size=self.hidden_size)

        # LSTM encoder and decoder
        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                    num_layers=self.num_lstm_layers, dropout=self.dropout)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                    num_layers=self.num_lstm_layers, dropout=self.dropout)

        # Gated skip connection and normalization
        self.gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size))

        # Temporal Fusion Decoder

        # Static enrichment layer
        self.static_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                      self.dropout, self.hidden_size)

        # Temporal Self-attention layer
        self.multihead_attn = InterpretableMultiHeadAttention(self.num_attention_heads, self.hidden_size)
        self.attention_gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.attention_add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size, self.hidden_size))

        # Position-wise feed-forward layer
        self.position_wise_feed_forward = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                               self.dropout)

        # Output layer
        self.output_gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.output_add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size, self.hidden_size))

        self.output = TemporalLayer(nn.Linear(self.hidden_size, len(self.quantiles)))

    def define_static_covariate_encoders(self, x):
        embedding_vectors = [self.static_embeddings[col](x[:, 0, self.col_to_idx[col]].long().to(self.device)) for col
                             in self.static_covariates]
        static_embedding = torch.cat(embedding_vectors, dim=1)
        static_encoder, static_weights = self.static_variable_selection(static_embedding)

        # Static context vectors
        static_context_s = self.static_context_variable_selection(
            static_encoder)  # Context for temporal variable selection
        static_context_e = self.static_context_enrichment(static_encoder)  # Context for static enrichment layer
        static_context_h = self.static_context_state_h(
            static_encoder)  # Context for local processing of temporal features (encoder/decoder)
        static_context_c = self.static_context_state_c(
            static_encoder)  # Context for local processing of temporal features (encoder/decoder)

        return static_encoder, static_weights, static_context_s, static_context_e, static_context_h, static_context_c

    def define_past_inputs_encoder(self, x, context):
        embedding_vectors = torch.cat(
            [self.temporal_cat_embeddings[col](x[:, :, self.col_to_idx[col]].long()) for col in
             self.time_dependent_categorical], dim=2)
        transformation_vectors = torch.cat(
            [self.temporal_real_transformations[col](x[:, :, self.col_to_idx[col]]) for col in
             self.time_dependent_continuous], dim=2)

        past_inputs = torch.cat([embedding_vectors, transformation_vectors], dim=2)
        past_encoder, past_weights = self.past_variable_selection(past_inputs, context)

        return past_encoder.transpose(0, 1), past_weights

    def define_known_future_inputs_decoder(self, x, context):
        embedding_vectors = torch.cat(
            [self.temporal_cat_embeddings[col](x[:, :, self.col_to_idx[col]].long()) for col in
             self.time_dependent_categorical if col not in self.observed_time_dependent], dim=2)

        transformation_vectors = torch.cat(
            [self.temporal_real_transformations[col](x[:, :, self.col_to_idx[col]]) for col in
             self.time_dependent_continuous if col not in self.observed_time_dependent], dim=2)

        future_inputs = torch.cat([embedding_vectors, transformation_vectors], dim=2)
        future_decoder, future_weights = self.future_variable_selection(future_inputs, context)

        return future_decoder.transpose(0, 1), future_weights

    def define_lstm_encoder(self, x, static_context_h, static_context_c):
        output, (state_h, state_c) = self.lstm_encoder(x, (
            static_context_h.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1),
            static_context_c.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)))

        return output, state_h, state_c

    def define_lstm_decoder(self, x, state_h, state_c):
        output, (_, _) = self.lstm_decoder(x, (state_h.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1),
                                               state_c.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)))

        return output

    def get_mask(self, attention_inputs):
        # mask = torch.cumsum(torch.eye(attention_inputs.shape[1]*self.num_attention_heads, attention_inputs.shape[0]), dim=1)
        mask = torch.cumsum(torch.eye(attention_inputs.shape[0] * self.num_attention_heads, attention_inputs.shape[1]),
                            dim=1)

        return mask.unsqueeze(2).to(self.device)

    def forward(self, x):
        # Static variable selection and static covariate encoders
        static_encoder, static_weights, static_context_s, static_context_e, static_context_h, static_context_c = self.define_static_covariate_encoders(
            x["inputs"])

        # Past input variable selection and LSTM encoder
        past_encoder, past_weights = self.define_past_inputs_encoder(
            x["inputs"][:, :self.encoder_steps, :].float().to(self.device), static_context_s)

        # Known future inputs variable selection and LSTM decoder
        future_decoder, future_weights = self.define_known_future_inputs_decoder(
            x["inputs"][:, self.encoder_steps:, :].float().to(self.device), static_context_s)

        # Pass output from variable selection through LSTM encoder and decoder
        encoder_output, state_h, state_c = self.define_lstm_encoder(past_encoder, static_context_h, static_context_c)
        decoder_output = self.define_lstm_decoder(future_decoder, static_context_h, static_context_c)

        # Gated skip connection before moving into the Temporal Fusion Decoder
        variable_selection_outputs = torch.cat([past_encoder, future_decoder], dim=0)
        lstm_outputs = torch.cat([encoder_output, decoder_output], dim=0)
        gated_outputs = self.gated_skip_connection(lstm_outputs)
        temporal_feature_outputs = self.add_norm(variable_selection_outputs.add(gated_outputs))
        temporal_feature_outputs = temporal_feature_outputs.transpose(0, 1)

        # Temporal Fusion Decoder

        # Static enrcihment layer
        static_enrichment_outputs = self.static_enrichment(temporal_feature_outputs, static_context_e)

        # Temporal Self-attention layer
        mask = self.get_mask(static_enrichment_outputs)
        multihead_outputs, multihead_attention = self.multihead_attn(static_enrichment_outputs,
                                                                     static_enrichment_outputs,
                                                                     static_enrichment_outputs, mask=mask)

        attention_gated_outputs = self.attention_gated_skip_connection(multihead_outputs)
        attention_outputs = self.attention_add_norm(attention_gated_outputs.add(static_enrichment_outputs))

        # Position-wise feed-forward layer
        temporal_fusion_decoder_outputs = self.position_wise_feed_forward(attention_outputs)

        # Output layer
        gate_outputs = self.output_gated_skip_connection(temporal_fusion_decoder_outputs)
        norm_outputs = self.output_add_norm(gate_outputs.add(temporal_feature_outputs))

        output = self.output(norm_outputs[:, self.encoder_steps:, :]).view(-1, 3)

        attention_weights = {
            'multihead_attention': multihead_attention,
            'static_weights': static_weights[Ellipsis, 0],
            'past_weights': past_weights[Ellipsis, 0, :],
            'future_weights': future_weights[Ellipsis, 0, :]
        }

        return output, attention_weights


if __name__ == '__main__':
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
        "static_covariates": ["Region", "Symbol"],
        "time_dependent_categorical": ["day_of_week", "day_of_month", "week_of_year", "month"],
        "time_dependent_continuous": ['log_vol', 'days_from_start', "open_to_close", ],
        "category_counts": {"day_of_week": 7, "day_of_month": 31, "week_of_year": 53, "month": 12, "Region": 4,
                            "Symbol": 31},
        "known_time_dependent": ["day_of_week", "day_of_month", "week_of_year", "month", "days_from_start"],
        "observed_time_dependent": ["log_vol", "open_to_close"]
    }
    model = TemporalFusionTransformer(params)
    model.to(DEVICE)
