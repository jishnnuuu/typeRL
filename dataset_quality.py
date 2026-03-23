import csv
from collections import defaultdict
import numpy as np

from text_processing import count_tracked_bigrams
from bigrams import BIGRAM_BAG


def analyze_dataset(filename="typing_dataset_cleaned.csv"):
    total_sentences = 0
    bigram_counts = defaultdict(list)
    length_stats = []
    
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["sentence"]
            bigram = row["target_bigram"]
            
            counts = count_tracked_bigrams(sentence)
            count = counts.get(bigram, 0)
            
            bigram_counts[bigram].append(count)
            length_stats.append(len(sentence.split()))
            
            total_sentences += 1
    print(f"\nTotal sentences: {total_sentences}")
    
    # --- Bigram count stats ---
    print("\nBigram occurrence stats:")
    all_counts = []
    
    for b, vals in bigram_counts.items():
        avg = np.mean(vals)
        min_c = np.min(vals)
        max_c = np.max(vals)
        
        print(f"{b}: avg={avg:.2f}, min={min_c}, max={max_c}")
        
        all_counts.extend(vals)
    
    count = 0
    for i in range(len(all_counts)):
        if(all_counts[i]<3):
            count+=1
    
    print(f"bad rows : {count}")
    
    print("\nOverall:")
    print(f"Avg count: {np.mean(all_counts):.2f}")
    print(f"Min count: {np.min(all_counts)}")
    print(f"Max count: {np.max(all_counts)}")
    
    # --- Sentence length ---
    print("\nSentence length stats:")
    print(f"Avg length: {np.mean(length_stats):.2f}")
    print(f"Min length: {np.min(length_stats)}")
    print(f"Max length: {np.max(length_stats)}")


if __name__ == "__main__":
    analyze_dataset()