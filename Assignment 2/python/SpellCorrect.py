import math
from Datum import Datum
from Sentence import Sentence
from HolbrookCorpus import HolbrookCorpus
from UniformLanguageModel import UniformLanguageModel
from UnigramLanguageModel import UnigramLanguageModel
from StupidBackoffLanguageModel import StupidBackoffLanguageModel
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel
from CustomLanguageModel import CustomLanguageModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types

# Modified version of Peter Norvig's spelling corrector
"""Spelling Corrector.

Copyright 2007 Peter Norvig. 
Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re
import collections


class SpellCorrect:

    """Holds edit model, language model, corpus. trains"""

    def __init__(self, lm, corpus):
        """initializes the language model."""
        self.languageModel = lm
        self.editModel = EditModel('../data/count_1edit.txt', corpus)

    def evaluate(self, corpus):
        """Tests this speller on a corpus, returns a SpellingResult"""
        numCorrect = 0
        numTotal = 0
        testData = corpus.generateTestCases()
        for sentence in testData:
            if sentence.isEmpty():
                continue
            errorSentence = sentence.getErrorSentence()
            hypothesis = self.correctSentence(errorSentence)
            if sentence.isCorrection(hypothesis):
                numCorrect += 1
            numTotal += 1
        return SpellingResult(numCorrect, numTotal)

    def correctSentence(self, sentence):
        """Takes a list of words, returns a corrected list of words."""
        if len(sentence) == 0:
            return []
        argmax_i = 0
        argmax_w = sentence[0]
        maxscore = float('-inf')
        maxlm = float('-inf')
        maxedit = float('-inf')

        # skip start and end tokens
        for i in range(1, len(sentence) - 1):
            word = sentence[i]
            editProbs = self.editModel.editProbabilities(word)
            for alternative, editscore in editProbs.iteritems():
                if alternative == word:
                    continue
                sentence[i] = alternative
                lmscore = self.languageModel.score(sentence)
                if editscore != 0:
                    editscore = math.log(editscore)
                else:
                    editscore = float('-inf')
                score = lmscore + editscore
                if score >= maxscore:
                    maxscore = score
                    maxlm = lmscore
                    maxedit = editscore
                    argmax_i = i
                    argmax_w = alternative

            # restores sentence to original state before moving on
            sentence[i] = word
        argmax = list(sentence)  # copy it
        argmax[argmax_i] = argmax_w  # correct it
        return argmax

    def correctCorpus(self, corpus):
        """Corrects a whole corpus, returns a JSON representation of the output."""
        string_list = []  # we will join these with commas,  bookended with []
        sentences = corpus.corpus
        for sentence in sentences:
            uncorrected = sentence.getErrorSentence()
            corrected = self.correctSentence(uncorrected)  # List<String>
            word_list = '["%s"]' % '","'.join(corrected)
            string_list.append(word_list)
        output = '[%s]' % ','.join(string_list)
        return output


def main():
    """Trains all of the language models and tests them on the dev data. Change devPath if you
       wish to do things like test on the training data."""
    trainPath = '../data/holbrook-tagged-train.dat'
    trainingCorpus = HolbrookCorpus(trainPath)

    devPath = '../data/holbrook-tagged-dev.dat'
    devCorpus = HolbrookCorpus(devPath)

    # print "===================================================================="

    # print 'Uniform Language Model: '
    # uniformLM = UniformLanguageModel(trainingCorpus)
    # uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
    # uniformOutcome = uniformSpell.evaluate(devCorpus)
    # print str(uniformOutcome)

    # print "===================================================================="

    # print 'Unigram Language Model: '
    # unigramLM = UnigramLanguageModel(trainingCorpus)
    # unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
    # unigramOutcome = unigramSpell.evaluate(devCorpus)
    # print str(unigramOutcome)

    # print "===================================================================="

    # print 'Laplace Unigram Language Model: '
    # laplaceUnigramLM = LaplaceUnigramLanguageModel(trainingCorpus)
    # laplaceUnigramSpell = SpellCorrect(laplaceUnigramLM, trainingCorpus)
    # laplaceUnigramOutcome = laplaceUnigramSpell.evaluate(devCorpus)
    # print str(laplaceUnigramOutcome)

    # print "===================================================================="

    # print 'Laplace Bigram Language Model: '
    # laplaceBigramLM = LaplaceBigramLanguageModel(trainingCorpus)
    # laplaceBigramSpell = SpellCorrect(laplaceBigramLM, trainingCorpus)
    # laplaceBigramOutcome = laplaceBigramSpell.evaluate(devCorpus)
    # print str(laplaceBigramOutcome)

    # print "===================================================================="

    # print 'Stupid Backoff Language Model: '
    # sbLM = StupidBackoffLanguageModel(trainingCorpus)
    # sbSpell = SpellCorrect(sbLM, trainingCorpus)
    # sbOutcome = sbSpell.evaluate(devCorpus)
    # print str(sbOutcome)

    print "===================================================================="

    print 'Custom Language Model: '
    customLM = CustomLanguageModel(trainingCorpus)
    customSpell = SpellCorrect(customLM, trainingCorpus)
    customOutcome = customSpell.evaluate(devCorpus)
    print str(customOutcome)

    print "===================================================================="

if __name__ == "__main__":
    main()
