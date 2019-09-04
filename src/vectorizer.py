import numpy as np
import os
import pickle

from src.BERT_experiments.vocab import BERTTextEncoder

from input import INPUT_DIR

VOCAB_PATH = os.path.join(INPUT_DIR, 'vocab.txt')


class Vectorizer(object):

    def __init__(self):
        pass

    def vectorize_inputs(self, sequence, max_sequence_size=100, **kwargs):
        raise NotImplementedError


class BERTVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequence, max_sequence_size=512, **kwargs):
        """
        Vectorize the sentence from tokens to tokens ids
        :param max_sequence_size: Max sequence that can fit on BERT (max-512)
        :param sequence: List with the tokens of the sentence
        :param kwargs[i]:The index (int) of the sentence on the summary e.g. 1st, 2nd,...
        :return:
        """

        bert_tokenizer = BERTTextEncoder(vocab_file=VOCAB_PATH, do_lower_case=True, max_len=max_sequence_size)

        return bert_tokenizer.encode(sequence, kwargs['i'])

    def transform_to_inputs(self, tokens):
        """
        Pads the lists of tokens, masks and segment in order to have 512 length
        :param tokens: List of the sentence tokens.
        :return: 3d array which can be fit on BERT
        """
        inputs = np.zeros((3, 1, 512), dtype=np.int32)

        if len(tokens) <= 512:
            inputs[0, 0, :len(tokens)] = tokens  # words
            inputs[1, 0, :len(tokens)] = np.ones((len(tokens)), dtype=np.int32)  # masks
            # seg are already to zero

        else:
            inputs[0, 0, :] = tokens[:512]  # words
            inputs[1, 0, :] = np.ones((512), dtype=np.int32)  # masks
            # seg aer already to zero

        return inputs


class W2VVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()
        w2v_index = os.path.join(INPUT_DIR, 'glove-wiki-gigaword-200.index')
        self.indices = {'word': pickle.load(open(w2v_index, 'rb'))}

    def vectorize_inputs(self, sequence, max_sequence_size=300, **kwargs):
        """
        Produce W2V indices for each token in the list of tokens
        :param sequence: list of lists of tokens
        :param max_sequence_size: maximum padding
        """

        word_inputs = np.zeros((1, max_sequence_size, ), dtype=np.int32)

        for j, token in enumerate(sequence[:max_sequence_size]):
            if token in self.indices['word']:
                word_inputs[0][j] = self.indices['word'][token]
            else:
                word_inputs[0][j] = self.indices['word']['UNKNOWN']

        return word_inputs
