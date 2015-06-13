import math
import collections


class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        # TODO your code here
        self.lUniCounts = collections.defaultdict(lambda: 0)
        self.lBiCounts = collections.defaultdict(lambda: 0)
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        for sentence in corpus.corpus:
            previous = ''
            for datum in sentence.data:
                token = datum.word
                self.lUniCounts[token] += 1
                if previous:
                    self.lBiCounts[previous + ',' + token] += 1
                previous = token

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0
        previous = ''
        for token in sentence:
            if previous:
                biCount = self.lBiCounts[previous + ',' + token]
                UniCount = self.lUniCounts[previous]
                score += math.log(biCount + 1)
                score -= math.log(UniCount + len(self.lBiCounts))
            previous = token
        return score
