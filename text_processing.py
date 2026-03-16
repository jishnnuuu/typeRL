# text_processing.py

import re
import numpy as np
from collections import Counter

from bigrams import BIGRAM_BAG, BIGRAM_SET


def preprocess_text(text):
    """
    Clean the sentence before bigram extraction.
    """
    text = text.lower()
    # remove punctuation and spaces
    text = re.sub(r'[^a-z]', '', text)
    return text


def extract_bigrams(text):
    """
    Extract bigrams using sliding window.
    """
    clean_text = preprocess_text(text)
    bigrams = []
    
    for i in range(len(clean_text) - 1):
        bigram = clean_text[i:i+2]
        bigrams.append(bigram)
    return bigrams


def count_tracked_bigrams(sentence):
    """
    Count occurrences of bigrams that exist in BIGRAM_BAG.
    """
    extracted = extract_bigrams(sentence)
    counts = Counter()
    
    for b in extracted:
        if b in BIGRAM_SET:   # fast lookup
            counts[b] += 1
    return counts


def counts_to_vector(counts):
    """
    Convert bigram counts into vector aligned with BIGRAM_BAG.
    """
    vector = np.zeros(len(BIGRAM_BAG))
    for i, b in enumerate(BIGRAM_BAG):
        if b in counts:
            vector[i] = counts[b]
    return vector