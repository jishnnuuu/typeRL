"""
it builds
(bigram, difficulty) → list of sentences

so the lookup is O(1)
"""


import csv
import random
from collections import defaultdict

class SentenceDataset:
    def __init__(self, filename):
        self.data = defaultdict(list)
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                bigram = row["target_bigram"]
                difficulty = int(row["difficulty"])
                sentence = row["sentence"]
                self.data[(bigram, difficulty)].append(sentence)
                
    def sample(self, bigram, difficulty):
        key = (bigram, difficulty)
        
        if key not in self.data or len(self.data[key]) == 0:
            raise ValueError(f"No sentences for {bigram}, difficulty {difficulty}")
        
        return random.choice(self.data[key])