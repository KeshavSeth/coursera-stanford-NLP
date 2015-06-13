import math
import collections


class StupidBackoffLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        # TODO your code here
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.trigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def getWord(self, sentence, idx):
        return sentence.data[idx].word

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        for sentence in corpus.corpus:
            for i in xrange(len(sentence.data)):
                # print i
                self.total += 1
                token = self.getWord(sentence, i)
                self.unigramCounts[token] += 1
                if i > 0:
                    p_token = self.getWord(sentence, i - 1)
                    self.bigramCounts[(p_token, token)] += 1
                    if i > 1:
                        pp_token = self.getWord(sentence, i - 2)
                        self.trigramCounts[(pp_token, p_token, token)] += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0
        ff = sentence[0]
        ss = sentence[1]
        alpha = 0.9
        for token in sentence[2:]:
            tricount = self.trigramCounts[(ff, ss, token)]
            tri_bicount = self.bigramCounts[(ff, ss)]
            bicount = self.bigramCounts[(ss, token)]
            bi_unicount = self.unigramCounts[ss]
            unicount = self.unigramCounts[token]
            if tricount > 0:
                score += math.log(tricount)
                score -= math.log(tri_bicount)
            elif bicount > 0:
                score += math.log(bicount)
                score -= math.log(bi_unicount)
                score += math.log(alpha)
            else:
                score += math.log((unicount + 1))
                score -= math.log(self.total + len(self.unigramCounts))
                score += math.log(alpha * alpha)
            ff, ss = ss, token
        return score
