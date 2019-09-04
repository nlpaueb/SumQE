from typing import List

from src.BERT_experiments.tokenization import FullTokenizer


class TextEncoder:
    SPECIAL_COUNT = 4
    NUM_SEGMENTS = 2
    BERT_UNUSED_COUNT = 99  # bert pretrained models
    BERT_SPECIAL_COUNT = 4  # they don't have DEL

    def __init__(self, vocab_size: int):
        # NOTE you MUST always put unk at 0, then regular vocab, then special tokens, and then pos
        self.vocab_size = vocab_size
        self.unk_id = 0

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, sent: str) -> List[int]:
        raise NotImplementedError()


class BERTTextEncoder(TextEncoder):
    def __init__(self, vocab_file: str, do_lower_case: bool = True, max_len=512) -> None:
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        super().__init__(len(self.tokenizer.vocab))
        self.max_len = max_len
        self.bert_unk_id = self.tokenizer.vocab['[UNK]']
        self.bert_msk_id = self.tokenizer.vocab['[MASK]']
        self.bert_cls_id = self.tokenizer.vocab['[CLS]']
        self.bert_sep_id = self.tokenizer.vocab['[SEP]']

    def encode(self, sent: str, i: int) -> List[int]:
        if i == 0:
            return [self.bert_cls_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))[:self.max_len - 2] + [self.bert_sep_id]
        else:
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))[:self.max_len - 1] + [self.bert_sep_id]
