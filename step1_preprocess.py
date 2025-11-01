import os
import re
import pandas as pd

# Input & output files
INPUT_CSV = "airline_faq.csv"
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "faq_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1 — Load
df = pd.read_csv(INPUT_CSV)
print("Loaded CSV with shape:", df.shape)
print(df.head())

# Step 2 — Basic cleanup
def clean_text(t):
    if pd.isna(t): return ""
    t = str(t).strip().replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.lower()

df["question"] = df["Question"].apply(clean_text)
df["answer"] = df["Answer"].apply(clean_text)

# Step 3 — Combine into single text field for embeddings
df["text"] = df["question"] + "\n\n" + df["answer"]

# Step 4 — Remove duplicates and empties
df = df.drop_duplicates(subset=["question", "answer"])
df = df[(df["question"] != "") & (df["answer"] != "")].reset_index(drop=True)

# Save cleaned dataset
df.to_csv(OUTPUT_CSV, index=False)
print("✅ Cleaned dataset saved to:", OUTPUT_CSV)
print(df.head())
