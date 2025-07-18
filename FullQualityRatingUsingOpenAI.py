import csv
import time
import re
import openai
from typing import Tuple
import concurrent.futures
from dotenv import load_dotenv
import os

# === CONFIGURATION ===
CSV_PATH = r"F:\PaperBA\selectedAWARDS_nochmehr\download_metadata"
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY_UNI")
openai.api_key = API_KEY

MODEL = "gpt-4o-mini"
MAX_WORKERS = 5
MAX_ENTRIES = 5  # limit for quick testing
Debug = False  # Set to False for production use

DEFAULT_JOURNAL_SCORE = 0.2
DEFAULT_TYPE_SCORE = 0.1
DEFAULT_SURVEY_BONUS = 0.0

SYSTEM_PROMPT = """
You are an AI assistant tasked with scoring research paper entries based on their publication venue, type, and determining if it's a survey paper.

**Instructions:**
You will be given information for a single paper: its title, journal, publisher, and publication type.
You need to determine three scores for this paper: a "Journal/Publisher Score", a "Publication Type Score", and a "Survey Bonus".

1.  **Journal/Publisher Score:**
    * Match "journal" or "publisher" against JOURNAL_RANKINGS.
    * Use journal if available; fallback to publisher.
    * The score is based on the jornals h5 ranking. If the jornal is not listed compare its h5 ranking with the h5 ranking of the listed journals and assign a suitable score on the scale (1, 0.8, 0.6, 0.4, 0.2, 0.01).

2.  **Publication Type Score:**
    * Match the provided type against FORM_SCORES.
    * If unknown, assign 0.1 or the value matching the entry.

3.  **Survey Bonus:**
    * If the title suggests it's a survey/review paper (e.g., "A Review of...", "Survey of...", "Systematic Review" etc.), assign 0.2.
    * Otherwise, assign 0.0.

**Scoring Rubrics:**

JOURNAL_RANKINGS = {
    #Top 10 h5
    "Nature": 1.0,
    "Science": 1.0,
    "NeurIPS": 1.0,
    #Top 25 h5
    "ICML": 0.8,
    "International Conference on Machine Learning": 0.8,
    "CVPR": 0.8,
    "IEEE": 0.8,
    #Top 50 h5
    "EMNLP": 0.6,
    "Springer": 0.6,
    "Elsevier": 0.6,
    "ACM": 0.6,
    #Top 100 h5
    "MDPI": 0.4,
    "Hindawi": 0.4,
    "all other journals if no h5 ranking": 0.2,
    "known predator journals": 0.01
}

FORM_SCORES = {
    "journal-article": 1.0,
    "proceedings-article": 0.8,
    "posted-content": 0.6,
    "book-chapter": 0.6,
    "monograph": 0.6,
    "reference-entry": 0.3,
    "other": 0.2
}

**Output Format:**
Journal/Publisher Score: [Score]
Publication Type Score: [Score]
Survey Bonus: [Score]
"""

USER_PROMPT_TEMPLATE = """
Please score the following paper based on the instructions:
Title: '{title}'
Journal: '{journal}'
Publisher: '{publisher}'
Publication Type: '{pub_type}'
"""

def fetch_scores_from_llm(entry_data: dict) -> Tuple[dict, float, float, float]:

    # === Holt Scores für ein einzelnes Paper von GPT ===
    # Nutzt Titel, Journal, Publisher und Publikationstyp als Input.
    # Extrahiert daraus:
    # - Journal/Publisher Score (basierend auf Ranking)
    # - Publication Type Score
    # - Survey Bonus (ob Titel auf Review/Survey schließen lässt)
    # Gibt Metadaten und alle drei Scores zurück. Fängt Fehler ab und loggt sie.

    title     = entry_data.get("title", "[Kein Titel]")
    journal   = entry_data.get("journal", "")
    publisher = entry_data.get("publisher", "")
    pub_type  = entry_data.get("pub_type", "")

    print(f"\U0001F4E8 Submitting: {title[:60]}...(Journal: {journal}, Publisher: {publisher}, Type: {pub_type})")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        journal=journal or "N/A",
        publisher=publisher or "N/A",
        pub_type=pub_type or "N/A"
    )

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0
        )

        reply = response.choices[0].message.content.strip()

        journal_score = float(re.search(r"Journal/Publisher Score:\s*([0-9.]+)", reply).group(1)) \
                        if re.search(r"Journal/Publisher Score:", reply) else DEFAULT_JOURNAL_SCORE

        type_score    = float(re.search(r"Publication Type Score:\s*([0-9.]+)", reply).group(1)) \
                        if re.search(r"Publication Type Score:", reply) else DEFAULT_TYPE_SCORE

        survey_bonus  = float(re.search(r"Survey Bonus:\s*([0-9.]+)", reply).group(1)) \
                        if re.search(r"Survey Bonus:", reply) else DEFAULT_SURVEY_BONUS

        print(f"✅ Scores → Journal:{journal_score:.2f}  Type:{type_score:.2f}  Survey:{survey_bonus:.2f}")
        print("score was for title:", title)
        return entry_data, round(journal_score, 2), round(type_score, 2), round(survey_bonus, 2)

    except Exception as e:
        print(f"⚠️ LLM error for '{title[:60]}': {e}")
        return entry_data, DEFAULT_JOURNAL_SCORE, DEFAULT_TYPE_SCORE, DEFAULT_SURVEY_BONUS

# === Main ===
entries_to_process = []
original_headers   = []

try:
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        original_headers = reader.fieldnames or []

        for i, row in enumerate(reader):
            if Debug and i >= MAX_ENTRIES:
                break
            entries_to_process.append({
                "identifier": f"Entry {i+1}",
                "title":      row.get("title", "[Kein Titel]"),
                "journal":    row.get("source_display_name", ""),
                "publisher":  row.get("publisher_name", ""),
                "pub_type":   row.get("type_crossref", ""),
                "original":   row
            })
except Exception as e:
    print(f"🚫 CSV load error: {e}")

if not entries_to_process:
    print("🚫 No entries to process.")
    exit()

print(f"\n🔧 Processing {len(entries_to_process)} entries with {MAX_WORKERS} threads...\n")
start_time = time.time()

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_entry = {executor.submit(fetch_scores_from_llm, e): e for e in entries_to_process}
    for future in concurrent.futures.as_completed(future_to_entry):
        original_entry, journal_score, type_score, survey_bonus = future.result()
        results.append({
            "original": original_entry["original"],
            "journal_publisher_score": journal_score,
            "publication_type_score":  type_score,
            "survey_bonus":            survey_bonus
        })

print(f"\n🏁 Done in {time.time() - start_time:.1f} s.")




# ---------- WRITE OUTPUT ----------
output_csv = "scored_papers_output_threaded_Awards_nochmehr.csv"
print(f"💾 Writing {len(results)} rows to {output_csv}")

extra_cols = ["journal_publisher_score", "publication_type_score", "survey_bonus"]
final_headers = original_headers.copy()
for col in extra_cols:
    if col not in final_headers:
        final_headers.append(col)

try:
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=final_headers)
        writer.writeheader()

        for res in results:
            row = res["original"].copy()
            row["journal_publisher_score"] = res["journal_publisher_score"]
            row["publication_type_score"]  = res["publication_type_score"]
            row["survey_bonus"]             = res["survey_bonus"]
            writer.writerow(row)

    print("✅ CSV saved successfully.")

except Exception as e:
    print(f"🚫 Write error: {e}")
