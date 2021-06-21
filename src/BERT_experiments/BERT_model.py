import tensorflow_hub as hub
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import numpy as np
import random

from keras.layers import Layer, Input, Dense, Dropout, Concatenate
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam


class BERT(Layer):
    def __init__(self, output_representation='pooled_output', hierarchical=False, **kwargs):
        self.bert = None
        self.hierarchical = hierarchical
        super(BERT, self).__init__(**kwargs)

        self.output_representation = output_representation

    def build(self, input_shape):
        self.bert = hub.Module('https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                               trainable=True, name="{}_module".format(self.name))

        # Remove unused layers and set trainable parameters
        self._trainable_weights += [var for var in self.bert.variables
                                    if not "/cls/" in var.name and not "/pooler/" in var.name]
        super(BERT, self).build(input_shape)

    def call(self, x, mask=None):

        inputs = dict(input_ids=x[0], input_mask=x[1], segment_ids=x[2])

        outputs = self.bert(inputs, as_dict=True, signature='tokens')['sequence_output']

        if self.output_representation == 'pooled_output':
            return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, 512, 768)


def custom_loss(y_true, y_pred):
    """
    This function computes the loss with respect to a single linguistic quality score. In Single-Task mode, this
    function calculates the 'mse' between the y_true and y_pred but in Multi-Task mode calculates the 'mse' for a
    specific quality measure ignoring the other outputs of the model
    :param y_true: The ground truth scores
    :param y_pred: The predicted scores
    :return:
    """
    return mean_squared_error(y_pred=y_pred[:, QUALITY_INDEX], y_true=y_true[:, QUALITY_INDEX])


def set_quality_index(mode, quality):
    """
    This function is only used just to set a value in the QUALITY_INDEX variable used in custom loss
    :param mode:
    :param quality:
    :return:
    """
    metrics_indexes = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'Q5': 4}
    global QUALITY_INDEX
    QUALITY_INDEX = 0 if mode == 'Single Task' else metrics_indexes[quality]


def compile_bert(shape, dropout_rate, lr, mode, human_metric):
    """
    Using the above class, creates, compiles the and returns the BERT model ready to be trained
    :param shape: The Input shape (We used 512 as the max bpes that can be fit).
    :param dropout_rate: The dropout rate of the model.
    :param lr: The learning rate of the model.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :param human_metric: The metric for which the model will be trained at.
    :return: The compiler model ready to be used.
    """
    random.seed(11)
    np.random.seed(13)
    tf.set_random_seed(21)

    word_inputs = Input(shape=(shape[1],), name='word_inputs', dtype='int32')
    mask_inputs = Input(shape=(shape[1],), name='pos_inputs', dtype='int32')
    seg_inputs = Input(shape=(shape[1],), name='seg_inputs', dtype='int32')
    doc_encoding = BERT()([word_inputs, mask_inputs, seg_inputs])

    doc_encoding = Dropout(dropout_rate)(doc_encoding)
    model = None

    # Final output (projection) layer
    if mode == 'Single Task':
        outputs = Dense(1, activation='linear', name='outputs')(doc_encoding)
        model = Model(inputs=[word_inputs, mask_inputs, seg_inputs], outputs=[outputs])

    elif mode == 'Multi Task-1':
        outputs = Dense(5, activation='linear', name='outputs')(doc_encoding)
        model = Model(inputs=[word_inputs, mask_inputs, seg_inputs], outputs=[outputs])

    elif mode == 'Multi Task-5':
        output_q1 = Dense(1, activation='linear', name='output_Q1')(doc_encoding)
        output_q2 = Dense(1, activation='linear', name='output_Q2')(doc_encoding)
        output_q3 = Dense(1, activation='linear', name='output_Q3')(doc_encoding)
        output_q4 = Dense(1, activation='linear', name='output_Q4')(doc_encoding)
        output_q5 = Dense(1, activation='linear', name='output_Q5')(doc_encoding)

        model = Model(inputs=[word_inputs, mask_inputs, seg_inputs],
                      outputs=[Concatenate()([output_q1, output_q2, output_q3, output_q4, output_q5])])

    set_quality_index(mode=mode, quality=human_metric)
    model.compile(optimizer=Adam(lr=lr), loss='mse', loss_weights=None, metrics=[custom_loss])

    return model
