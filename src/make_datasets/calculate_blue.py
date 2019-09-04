import sys
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BLEU:
    def __init__(self, length_limit=0, max_ngram=4, use_smoothing=False):
        self.length_limit = length_limit
        self.max_ngram = max_ngram
        self.use_smoothing = use_smoothing
        if self.use_smoothing:
            sf = SmoothingFunction()
            self.sf_methods = {'SM-{}'.format(f[-1]): getattr(sf, f) for f in dir(sf) if callable(getattr(sf, f)) and '__' not in f and f[-1] != '0'}

    def _preprocess_summary(self, summary):
        summary_tokens = [token.lower() for token in word_tokenize(text=summary, language='english')]
        if self.length_limit > 0:
            return summary_tokens[:self.length_limit]
        return summary_tokens

    def get_bleu_score_for_ngram(self, hypothesis, references, ngram=4, smoothing_function=None):
        weights = [1./n for n in range(1, ngram + 1)]
        tokenized_hypothesis = self._preprocess_summary(hypothesis)
        tokenized_references = [self._preprocess_summary(reference) for reference in references]

        return sentence_bleu(references=tokenized_references, hypothesis=tokenized_hypothesis, weights=weights, smoothing_function=smoothing_function)

    def get_bleu_scores(self, hypothesis, references):
        blue_scores = {}

        for n in range(1, self.max_ngram + 1):
            blue_name = 'BLEU-{0:s}'.format(str(n))

            blue_scores[blue_name] = self.get_bleu_score_for_ngram(hypothesis, references, n)

            if self.use_smoothing:
                for sf_name, sf_method in self.sf_methods.items():
                    blue_name = 'BLEU-{0:s}'.format(str(n))
                    if not (n <= 2 and sf_name == 'SM-6'):
                        blue_name = '{0:s}-{1:s}'.format(blue_name, sf_name)
                        try:
                            blue_scores[blue_name] = self.get_bleu_score_for_ngram(hypothesis, references, n, sf_method)
                        except AssertionError:
                            blue_scores[blue_name] = 0.
                            print('{0:s}: assertion error --> setting BLEU to 0.'.format(sf_name), file=sys.stderr)

        return blue_scores
