import math
import collections


class CustomLanguageModel:

    """
        Custome Language Model - Kneser-Ney discounting
        accuracy 0.25
    """

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        # TODO your code here
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.unigramCounts = collections.defaultdict(lambda: 0)
        """ firstCounts will be used as a container for continuation counts """
        self.firstCounts = collections.defaultdict(lambda: 0)
        self.secondCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def count_first_part(self, dict, token):
        # find cardinality of the set where token ends the bigram
        num = 0
        for key in dict.keys():
            if key[1] == token:
                num += 1
        return num

    def count_second_part(self, dict, token):
        # find cardinality of the set where token starts the bigram
        num = 0
        for key in dict.keys():
            if key[0] == token:
                num += 1
        return num

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        for sentence in corpus.corpus:
            previous = ''
            for datum in sentence.data:
                token = datum.word
                self.total += 1
                self.unigramCounts[token] += 1
                if previous:
                    self.bigramCounts[(previous, token)] += 1
                previous = token

        for token in self.unigramCounts.keys():
            self.firstCounts[token] = self.count_first_part(
                self.bigramCounts, token)
            self.secondCounts[token] = self.count_second_part(
                self.bigramCounts, token)

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0
        score_bi = 0.0
        score_uni = 0.0
        d = 0.75
        for i in xrange(len(sentence)):
            if i == 0:
                """
                    for the first word add the unigram probability
                """
                score += math.log(self.unigramCounts[sentence[i]] + 1)
                score -= math.log(self.total + len(self.unigramCounts))
            else:
                """
                    for i greater than 0 we will perform kneser-ney algorithm
                """
                count_bi = self.bigramCounts[(sentence[i - 1], sentence[i])]
                count_uni = self.unigramCounts[sentence[i - 1]]
                # cardinality of set {w' : c(w_i-1, w') > 0}
                count_follows_up = self.secondCounts[sentence[i - 1]]
                # cardinality of set {w_i-1: c(w_i-1, w_i) > 0}
                count_continue = self.firstCounts[sentence[i]]
                """
                second interpolation term
                """

                if count_uni < 1:
                    score += math.log(float(0.000027) / self.total)
                    continue

                assert count_uni > 0
                lambdaa = (float(d) / float(count_uni)) * count_follows_up
                p_continuation = float(
                    count_continue) / (len(self.bigramCounts))
                second_inter_term = lambdaa * p_continuation
                """
                first interpolation term
                """
                first_inter_term = float(
                    max(count_bi - d, 0)) / float(count_uni)

                prob = first_inter_term + second_inter_term
                # if p_continuation == 0.0:
                #     print count_uni
                #     print count_follows_up
                #     print count_continue
                if prob == 0.0:
                    score += math.log(self.unigramCounts[sentence[i]] + 1)
                    score -= math.log(self.total + len(self.unigramCounts))
                else:
                    score += math.log(prob)
        return score
