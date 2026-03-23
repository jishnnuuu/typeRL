import csv
from text_processing import count_tracked_bigrams


def find_bad_samples(filename="typing_dataset_cleaned.csv"):
    bad_rows = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sentence = row["sentence"]
            bigram = row["target_bigram"]
            difficulty = row["difficulty"]
            count = count_tracked_bigrams(sentence).get(bigram, 0)
            
            if count < 3:
                bad_rows.append({
                    "index": idx,
                    "sentence": sentence,
                    "bigram": bigram,
                    "difficulty": difficulty,
                    "count": count
                })
    print(f"Total bad samples: {len(bad_rows)}\n")
    for row in bad_rows[:]:  # preview first 20
        print(row)
    return bad_rows


if __name__ == "__main__":
    bad_rows = find_bad_samples()