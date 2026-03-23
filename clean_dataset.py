import csv
from text_processing import count_tracked_bigrams


def clean_dataset(input_file="typing_dataset.csv", output_file="typing_dataset_cleaned.csv"):
    good_rows = []
    removed = 0
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["sentence"]
            bigram = row["target_bigram"]
            count = count_tracked_bigrams(sentence).get(bigram, 0)
            if count >= 3:
                good_rows.append(row)
            else:
                removed += 1
    print(f"Removed rows: {removed}")
    print(f"Remaining rows: {len(good_rows)}")
    # write cleaned dataset
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "target_bigram", "difficulty"])
        writer.writeheader()
        writer.writerows(good_rows)
    print(f"\nCleaned dataset saved to: {output_file}")


if __name__ == "__main__":
    clean_dataset()