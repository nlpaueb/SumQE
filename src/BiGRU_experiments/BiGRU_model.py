import numpy as np
import os
from gensim.models import KeyedVectors

from keras.layers import add, Bidirectional, Concatenate, CuDNNGRU
from keras.layers import Dense, Embedding, Input, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


from .masking import Camouflage, SymmetricMasking
from .attension import Attention
from .dropout import TimestepDropout

from input import INPUT_DIR

EMBEDDINGS_PATH = os.path.join(INPUT_DIR, 'glove-wiki-gigaword-200.bin')


def pretrained_embedding():
    inputs = Input(shape=(None,), dtype='int32')
    embeddings = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)
    word_encodings_weights = np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0)
    embeds = Embedding(len(word_encodings_weights), word_encodings_weights.shape[-1],
                       weights=[word_encodings_weights], trainable=False)(inputs)

    return Model(inputs=inputs, outputs=embeds, name='embedding')


def compile_bigrus_attention(shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr, mode):
    """
    Compiles a Hierarchical RNN based on the given parameters
    :param shape: The input shape
    :param n_hidden_layers: How many stacked Layers you want.
    :param hidden_units_size: size of hidden units, as a list
    :param dropout_rate: The percentage of inputs to dropout
    :param word_dropout_rate: The percentage of timesteps to dropout
    :param lr: learning rate
    :return: Nothing
    """

    # Document Feature Representation
    doc_inputs = Input(shape=(shape[1],), name='doc_inputs')
    pretrained_encodings = pretrained_embedding()
    doc_embs = pretrained_encodings(doc_inputs)

    # Apply variational dropout
    drop_doc_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(doc_embs)
    encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_doc_embs)

    # Bi-GRUs over token embeddings
    for i in range(n_hidden_layers):

        grus = Bidirectional(
            CuDNNGRU(hidden_units_size, return_sequences=True), name='bidirectional_grus_{}'.format(i))(encodings)

        grus = Camouflage(mask_value=0.0)([grus, encodings])

        if i == 0:
            encodings = SpatialDropout1D(dropout_rate)(grus)
        else:
            encodings = add([grus, encodings])
            encodings = SpatialDropout1D(dropout_rate)(encodings)

    # Attention over BI-GRU (context-aware) embeddings
    # Mask encodings before attention
    grus_outputs = SymmetricMasking(mask_value=0, name='masking')([encodings, encodings])

    doc_encoding = Attention(kernel_regularizer=l2(),
                             bias_regularizer=l2(),
                             return_attention=False,
                             name='self_attention')(grus_outputs)
    model = None

    # Final output (projection) layer
    if mode == 'Single Task':
        outputs = Dense(1, activation='linear', name='outputs')(doc_encoding)
        model = Model(inputs=doc_inputs, outputs=[outputs])

    elif mode == 'Multi Task-1':
        outputs = Dense(5, activation='linear', name='outputs')(doc_encoding)
        model = Model(inputs=doc_inputs, outputs=[outputs])

    elif mode == 'Multi Task-5':
        output_q1 = Dense(1, activation='linear', name='output_Q1')(doc_encoding)
        output_q2 = Dense(1, activation='linear', name='output_Q2')(doc_encoding)
        output_q3 = Dense(1, activation='linear', name='output_Q3')(doc_encoding)
        output_q4 = Dense(1, activation='linear', name='output_Q4')(doc_encoding)
        output_q5 = Dense(1, activation='linear', name='output_Q5')(doc_encoding)

        model = Model(inputs=doc_inputs,
                      outputs=[Concatenate()([output_q1, output_q2, output_q3, output_q4, output_q5])])

    # Wrap up model + Compile with optimizer and loss function
    # model = Model(inputs=doc_inputs, outputs=[outputs])
    model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss='mse', loss_weights=None)

    return model
