import pickle
import operator
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# from plot_keras_history import plot_history
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import multilabel_confusion_matrix
# from keras_contrib.utils import save_load_utils

from keras import layers
from keras import optimizers

from keras.models import Model
# from keras import Input
from tensorflow_addons import metrics
from tensorflow_addons import losses
from tensorflow_addons.layers import CRF
from tensorflow_addons.text.crf import crf_log_likelihood



def build_bilstm(max_len, vocab_size, n_tags, embedding_matrix=None, embedding_dim=None, unit='lstm', num_units=100, dropout=0.1, recurrent_dropout=0.1):
    input = layers.Input(shape=(max_len,))

    if embedding_matrix is not None:
        model = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_matrix.shape[-1], input_length=max_len, mask_zero=True, weights=[embedding_matrix], trainable=False)(input)
    elif embedding_dim is not None:
        model = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len, mask_zero=True, embeddings_initializer='uniform')(input)
    else:
        raise ValueError('Must provide either an embedding matrix or an embedding dimension.')

    if unit == 'lstm':
        model = layers.Bidirectional(layers.LSTM(units=num_units, return_sequences=True, recurrent_dropout=recurrent_dropout))(model)
    elif unit == 'gru':
        model = layers.Bidirectional(layers.GRU(units=num_units, return_sequences=True, recurrent_dropout=recurrent_dropout))(model)
    elif unit == 'rnn':
        model = layers.Bidirectional(layers.SimpleRNN(units=num_units, return_sequences=True, recurrent_dropout=recurrent_dropout))(model)
    else:
        raise ValueError('Invalid unit type. Must be one of lstm, gru, or rnn.')

    model = layers.Dropout(dropout)(model)
    model = layers.TimeDistributed(layers.Dense(n_tags, activation="relu"))(model)

    crf_layer = CRF(units=n_tags)
    output_layer  = crf_layer(model)

    output_model = Model(input, output_layer)

    loss = losses.SigmoidFocalCrossEntropy()
    metric = metrics.F1Score(num_classes=n_tags, average='micro')
    
    opt = optimizers.RMSprop(lr=0.01)

    output_model.compile(optimizer=opt, loss=loss, metrics=[metric])
    output_model.summary()

    return output_model