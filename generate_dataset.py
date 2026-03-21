import os
import csv
import time
from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from bigrams import BIGRAM_BAG

# 1. Load the hidden .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Did you create the .env file?")

client = Groq(api_key=API_KEY)

def build_prompt(bigram, difficulty):
    difficulty_map = {
        0: "very simple words, 3-4 letters only",
        1: "simple common words, basic vocabulary",
        2: "normal everyday sentences",
        3: "slightly complex sentences with varied vocabulary",
        4: "advanced vocabulary and complex sentence structure"
    }
    
    return f"""
Generate ONE natural English sentence for typing practice.
Requirements:
- Must contain the letter combination: {bigram}
- The bigram must appear at least 5 times
- Sentence length: 8 to 12 words
- Difficulty: {difficulty_map[difficulty]}
- Typing optimization: Avoid rare symbols; stick to letters, commas, and periods.
- Natural flow: Do not use nonsense words to force the bigram.
- No artificial repetition
- Avoid repeating the same word more than twice
- Return ONLY the sentence text.
"""

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30)
)
def generate_sentence(bigram, difficulty):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": build_prompt(bigram, difficulty)}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip().strip('"')

def generate_dataset(samples_per_bigram=20):
    data = []
    # Using a list to track progress
    total_bigrams = len(BIGRAM_BAG)
    
    for b_idx, bigram in enumerate(BIGRAM_BAG):
        print(f"\nProcessing Bigram {b_idx+1}/{total_bigrams}: '{bigram}'")
        
        for difficulty in range(5):
            for i in range(samples_per_bigram):
                try:
                    sentence = generate_sentence(bigram, difficulty)
                    data.append({
                        "sentence": sentence,
                        "target_bigram": bigram,
                        "difficulty": difficulty
                    })
                    # Use end='\r' to keep the terminal clean
                    print(f"   Diff {difficulty} | Sample {i+1}/{samples_per_bigram} created.", end='\r')
                    
                    # 1.5s sleep helps stay under the 30 RPM free-tier limit
                    time.sleep(1.5) 
                except Exception as e:
                    print(f"\nError on '{bigram}': {e}")
                    continue
    return data

def save_to_csv(data, filename="typing_dataset.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "target_bigram", "difficulty"])
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    start_time = time.time()
    try:
        results = generate_dataset(samples_per_bigram=20)
        save_to_csv(results)
        elapsed = (time.time() - start_time) / 60
        print(f"\n\nDone! {len(results)} sentences saved in {elapsed:.1f} minutes.")
    except KeyboardInterrupt:
        print("\nStopping and saving partial progress...")
        save_to_csv(results)