# This script analyzes a text file to determine how frequently it uses certain
# words relative to the usual frequencies in English according to the Brown
# corpus. You can use this with the modifier parameter to skew the results toward
# the vocabulary of a certain author or text.

import codecs
import json
import math
import nltk
import re
import sys

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <infile.txt> <outfile.json>')

infile = codecs.open(sys.argv[1], 'r', 'utf8')
outfile = codecs.open(sys.argv[2], 'w', 'utf8')

re_word = re.compile(r"^[a-zA-Z']+$")

corpus = infile.read()
corpus_tokens = nltk.tokenize.word_tokenize(corpus)
corpus_tokens = [w.lower() for w in corpus_tokens
                 if re_word.match(w)]
corpus_freq = nltk.FreqDist(corpus_tokens)
n_corpus = sum(corpus_freq.values())

brown_tokens = [w.lower() for w in nltk.corpus.brown.words()
                if re_word.match(w)]
brown_freq = nltk.FreqDist(brown_tokens)
n_brown = sum(brown_freq.values())

scores = {}
for w in corpus_freq.keys() & brown_freq.keys():
    scores[w] = math.log((corpus_freq[w] / n_corpus) / (brown_freq[w] / n_brown))

json.dump(scores, outfile)
