""" 
We define a bag of bigrams, which are pairs of consecutive characters in a text.
We created 30 bigrams based on their frequency in English text. 
"""

# bigrams.py

BIGRAM_BAG = [
    "th","he","in","er","an","re","on","at","en","nd",
    "ti","es","or","te","of","ed","is","it","al","ar",
    "st","to","nt","ng","se","ha","as","ou","io","le"
]

def get_bigram_index(bigram):
    """
    Return index of bigram in the bag.
    """
    return BIGRAM_BAG.index(bigram)
