""" 
We define a bag of bigrams, which are pairs of consecutive characters in a text.
We created 30 bigrams based on their frequency in English text. 
"""

# bigrams.py

BIGRAM_BAG = [
    # very common English bigrams
    "th","he","in","er","an","re","on","at","en","nd",
    # common transitions
    "ti","es","or","te","of","ed","is","it","al","ar",
    # additional frequent patterns
    "st","to","nt","ng","se","ha","as","ou","io","le",
    # additional useful typing patterns
    "ve","co","me","de","hi","ri","ro","ic","ne","ea"
]

# fast lookup structure
BIGRAM_SET = set(BIGRAM_BAG)

def get_bigram_index(bigram):
    """
    Return index of bigram in the bag.
    """
    return BIGRAM_BAG.index(bigram)
