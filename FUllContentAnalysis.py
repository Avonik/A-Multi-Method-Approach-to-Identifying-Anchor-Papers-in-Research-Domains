import pandas as pd
import ast
import os
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import concurrent.futures
from functools import partial
from tqdm import tqdm
import time


"""
Skripit für die Konetxt-analyse mit OpenaiLLM (Bewertung des Zitationskontextes)
Mit multithread und Maßnahmen zur Tokenreduktion sowie robustheit gegen Fehler.
"""

# === CONFIGURATION ===
CSV_PATH = r"F:\PaperBA\FinalData\dataJoined.csv"

#nur wenn debug = true ist! :
END_ROW = 6000
START_ROW = 0
DEBUG = False

# Set the maximum number of parallel threads for processing papers
MAX_WORKERS = 6

load_dotenv()

# === HELPER FUNCTIONS ===


# === Gibt einen sauberen Titel zurück, oder einen Platzhalter, wenn leer oder ungültig ===
# Wird verwendet, um fehlende oder fehlerhafte Titel im Datensatz abzufangen.
def safe_title(val, placeholder="(untitled)"):
    """
    Liefert einen String-Titel; falls leer/NaN → Platzhalter.
    """
    if isinstance(val, str) and val.strip():
        return val.strip()
    return placeholder


# === Ruft das OpenAI-Modell auf und behandelt Rate-Limit-Fehler durch automatische Wiederholungen ===
# Führt einen Chat-Completion-Call mit System- und User-Prompt durch.
# Bei Fehlschlägen werden bis zu 'retries'-Versuche mit Wartezeit gemacht.
def llm_call(client, system_prompt, user_prompt, retries=5, delay_seconds=60):
    """
    Call the OpenAI chat completion API with retry on rate limit errors.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.01
            )
            if DEBUG and hasattr(response, "usage") and response.usage:
                u = response.usage
                print(f"[DEBUG] Tokens used - prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"[WARN] Rate limit hit. Attempt {attempt}/{retries}. Waiting {delay_seconds} seconds...")
            if attempt < retries:
                time.sleep(delay_seconds)
                continue
            else:
                return None  # lieber Paper skippen als Crash


# === Prüft, ob ein Satz eine Zitation enthält ===
# Erkennt sowohl numerische Zitationen ([1], [2, 5]) als auch Autor-Jahr-Zitationen (Müller et al., 2020).
def sentence_contains_citation(sentence):
    """Returns True if the sentence looks like it contains a citation marker (numeric or author-year)."""
    if re.search(r"\[\d+(?:,\s*\d+)*\]", sentence):
        return True
    if re.search(r"\([A-Z][A-Za-z]+[^)]*\d{4}\)", sentence):
        return True
    return False

## === Versucht, den Referenzteil (Literature Section) eines Volltexts zu extrahieren ===
# Sucht nach typischen Überschriften wie "References" oder "Literatur".
# Falls keine Überschrift gefunden wird, nutzt es die letzten 10 % des Textes als Fallback.
def extract_references_section(full_text: str) -> str:
    """Versucht die References-Section zu isolieren; Fallback: letzte 10 % des Dokuments."""
    lines = full_text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*(References|Bibliography|Literature|Literatur|Literature|Literaturverzeichnis)\s*$", line, re.IGNORECASE):
            return "\n".join(lines[i + 1 :])
    start = int(len(lines) * 0.9)
    return "\n".join(lines[start:])

# === Sichere Auswertung von Listen-Strings aus CSV-Zellen ===
# Nutzt `ast.literal_eval`, um z.B . Referenzlisten aus Text zu rekonstruieren.
# Gibt bei Syntaxfehlern oder leeren Werten eine leere Liste zurück.
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val) if pd.notna(val) else []
    except (ValueError, SyntaxError):
        return []

# === Holt sicher einen Wert aus einer Dictionary-Zeile, mit Fallback-Wert ===
# Fängt AttributeError ab, wenn zB. `row` kein echtes Dict ist.
def safe_get(row, key, default=""):
    try:
        return row.get(key, default)
    except AttributeError:
        return default

# ----------------------------------------------------------------

## === Zentrales Verarbeitungsskript für ein einzelnes Paper ===
# Prüft PDF-Verfügbarkeit, extrahiert Text, filtert relevante Sätze mit Zitaten,
# bestimmt das Zitationsformat (numeric / author-year) über LLM,
# reduziert die Zitatsätze basierend auf diesem Format,
# sendet anschließend alles zusammen an ein zweites LLM zur Impact-Bewertung der Zitate.
# Gibt für jedes zitierte Paper: Titel, OpenAlex-ID, Impact-Score und kurze Erklärung zurück.
def process_paper(paper_row_tuple, paper_dict, client):
    try:  # GANZES Paper in einen Schutzmantel packen
        _, row = paper_row_tuple
        citing_id = safe_get(row, "openalex_id", "")
        citing_pdf_path = str(safe_get(row, "local_pdf_path", "")).strip()
        citing_title = safe_title(safe_get(row, "title", ""))
        local_output_records = []

        if not citing_pdf_path or citing_pdf_path.lower() == "nan" or not os.path.exists(citing_pdf_path):
            print(f"[SKIP] Citing Paper: {citing_id} ('{citing_title}'). Reason: PDF not found at '{citing_pdf_path}'.")
            return None

        if DEBUG:
            total_refs = len(safe_get(row, "referenced_works", []))
            present = [cid for cid in row["referenced_works"] if cid in paper_dict]
            with_title = [cid for cid in present if str(paper_dict[cid]['title']).strip()]
            print(f"CSV refs: {total_refs} | in dataset: {len(present)} | with title: {len(with_title)}")

        cited_local_ids = [cid for cid in safe_get(row, "referenced_works", []) if cid in paper_dict]
        if not cited_local_ids:
            return None

        reference_entries = []
        for cid in cited_local_ids:
            cited_paper = paper_dict[cid]
            ref_title  = safe_title(cited_paper["title"])
            ref_authors = cited_paper.get("authors", "")
            reference_entries.append(
                f"* {ref_title}, Authors: {ref_authors}, ID: {cid}"
            )

        if not reference_entries:
            if DEBUG:
                print(f"[DEBUG] {citing_id}: no valid referenced papers with titles remaining.")
            return None

        reference_text = "\n".join(reference_entries)
        print(f"[INFO] Processing: {citing_id} ('{citing_title}')")

        # ----------- PDF einlesen ------------
        try:
            with fitz.open(citing_pdf_path) as doc:
                pages_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            print(f"[SKIP] Citing Paper: {citing_id} ('{citing_title}'). Reason: Could not read PDF. Error: {e}")
            return None

        sentences = re.split(r"(?<=[.!?])\s+", pages_text)
        relevant_sentences = [s for s in sentences if sentence_contains_citation(s)]
        if DEBUG:
            print(f"[DEBUG] {citing_id}: total sentences {len(sentences)}, relevant {len(relevant_sentences)}")

        references_section = extract_references_section(pages_text)
        text_excerpt = "\n".join(sentences[:10])[:1000]
        citation_sentences_for_llm = "\n".join(relevant_sentences) if relevant_sentences else pages_text

        # ----------- Reference-Section filtern ------------
        target_author_names = []
        for cid in cited_local_ids:
            cited_paper = paper_dict[cid]
            authors = cited_paper.get('authors', "")
            if isinstance(authors, str) and authors.strip():
                for name in authors.split(','):
                    try:
                        parts = name.strip().split()
                        if parts:
                            last_name = parts[-1]
                            target_author_names.append(last_name)
                    except Exception:
                        continue  # nie crashen wegen komischem Autorstring

        all_refs = references_section.split("\n")
        filtered_refs = [line for line in all_refs if any(author.lower() in line.lower() for author in target_author_names)]
        references_section = "\n".join(filtered_refs)  # nur gefilterte References

        # ----------- CALL 1 ------------
        citation_format_prompt = (
            "You will be given a Literature section of an academic paper. Your task is to identify the format in which references to other papers occur. Are they in [Author, Year] format, numeric format (e.g., [1], [2]), or something else? "
            "IF the citation uses a numeric format you *MUST* give the one number associated with each of given papers. e.g like this 'The references in the text occur in numeric format, with the associated numbers being: [24] for 'This is a paper title' and [5] for 'Another Paper Title'."
            "IF the citation uses an author format you *MUST* give the author and year associated with each of the given papers like this 'The references in the text occur in author and year format, with the associated identifiers being: [Mustermann, 2010] for 'This is a paper title' and [Heinz et all, 1921] for 'Another Paper Title'.. "
            "Output one short sentence"
        )
        user_prompt_1 = (
            f"[REFERENCE_SECTION]\n{references_section}\n\n"
            f"[TEXT_EXCERPT]\n{text_excerpt}\n\n"
            f"Papers to find: {reference_text}"
        )
        format_response = llm_call(client, citation_format_prompt, user_prompt_1)
        if not format_response:
            print(f"[SKIP] Citing Paper: {citing_id}. Reason: No response from format detection.")
            return None

        sentences_for_llm = citation_sentences_for_llm
        try:
            if "numeric format" in format_response.lower():
                nums = re.findall(r"\[(\d+)\]", format_response)
                if nums:
                    bracket_pattern = re.compile(r"\[(.*?)\]")
                    filtered = []
                    for s in relevant_sentences:
                        matches = bracket_pattern.findall(s)
                        if any(any(n.strip() == num for n in m.split(',')) for m in matches for num in nums):
                            filtered.append(s)
                    if filtered:
                        sentences_for_llm = "\n".join(filtered)
            elif "author and year format" in format_response.lower():
                author_entries = re.findall(r"\[([^\]]+?)\]", format_response)
                author_names = []
                for entry in author_entries:
                    author_part = entry.split(",")[0]
                    parts = re.split(r"\band\b|&|et al\.?|et all\.?", author_part, flags=re.IGNORECASE)
                    author_names.extend([p.strip().split()[0] for p in parts if p.strip()])
                filtered = [s for s in relevant_sentences if any(a.lower() in s.lower() for a in author_names)]
                if filtered:
                    sentences_for_llm = "\n".join(filtered)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] {citing_id}: Problem during sentence filtering ({e}) – proceeding with unfiltered sentences.")

        # ----------- CALL 2 ------------
        system_prompt_combined = """
            You are a meticulous academic research assistant. Your task is to (1) locate every sentence where a set of target papers is explicitly cited and (2) assign an impact score to each cited paper.

            **Inputs you will receive:**

            1. **Citation Format**: A short description showing how citation markers map to the target papers.
            2. **Sentences**: A list of sentences (plain text) that contain potential citation markers.
            3. **Target Papers**: A list of target papers with their titles and OpenAlex IDs.

            **Scoring rubric (impact_score):**
              - 1.0 → paper is *fundamental* to the citing paper
              - 0.8 → paper is somewhat usefull for the citing paper
              - 0.6 → paper is mentioned for context / incidental mention, e.g., only in literature section
              - 0.2 → critical / disapproving comment / paper is disproven

            **Your output must be a JSON dictionary** where:
              * Each **key** is the exact paper title.
              * Each **value** is a dictionary with keys:
                  - "impact_score": one of 1.0, 0.8, 0.4, 0.2
                  - "explanation": brief justification (max 25 words)
                  - "id": the OpenAlex ID (as given)

            **Important constraints:**

            * Use only the provided sentences to decide whether a paper is cited.
            * Match markers exactly as described in the citation format.
            * Explanation must be concise and directly reference the sentence content.
            * Return **ONLY** the JSON – no extra text.
        """
        combined_prompt = json.dumps({
            "citation_format": format_response.strip(),
            "sentences": sentences_for_llm,
            "target_papers": reference_entries
        })
        combined_response = llm_call(client, system_prompt_combined, combined_prompt)
        if not combined_response:
            print(f"[SKIP] Citing Paper: {citing_id} ('{citing_title}'). Reason: No response in combined call.")
            return None

        try:
            eval_data = json.loads(re.sub(r"^```json\s*|\s*```$", "", combined_response.strip(), flags=re.DOTALL))
        except json.JSONDecodeError:
            print(f"[SKIP] Citing Paper: {citing_id}. Reason: JSON parse error.")
            return None

        for cited_title, data in eval_data.items():
            cited_paper_openalex_id = data.get("id")
            if not cited_paper_openalex_id:
                print(f"[WARN] No ID in response for cited title '{cited_title}' in citing paper {citing_id}.")
                continue

            local_output_records.append({
                "citing_paper_id": citing_id,
                "citing_paper_title": citing_title,
                "cited_paper_id": cited_paper_openalex_id,
                "cited_paper_title": cited_title,
                "impact_score": data.get("impact_score", None),
                "explanation": data.get("explanation", "")
            })

        if DEBUG:
            print(f"[DEBUG] {citing_id}: Records produced -> {len(local_output_records)}")

        return local_output_records

    except Exception as e:
        print(f"[SKIP] Citing Paper: {safe_get(row,'openalex_id','<unknown>')} ('{safe_get(row,'title','<no title>')}'). Reason: Fatal error in process_paper. Error: {e}")
        return None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"CSV konnte nicht geladen werden: {e}")

    df["referenced_works"] = df["referenced_works"].apply(safe_literal_eval)
    paper_dict = {row["openalex_id"]: row for _, row in df.iterrows()}

    open_ai_key = os.getenv("OPENAI_API_KEY_UNI")
    if not open_ai_key:
        raise RuntimeError("OPENAI_API_KEY_UNI fehlt in .env!")

    client = OpenAI(api_key=open_ai_key)

    papers_to_process = list(df.iterrows())
    if DEBUG:
        papers_to_process = papers_to_process[START_ROW:END_ROW]

    all_output_records = []
    process_func = partial(process_paper, paper_dict=paper_dict, client=client)

    print(f"--- Starting processing for {len(papers_to_process)} papers using up to {MAX_WORKERS} threads ---")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_func, paper): paper for paper in papers_to_process}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing papers"):
            try:
                paper_result = future.result()
                if paper_result:
                    all_output_records.extend(paper_result)
            except Exception as e:
                # Selbst hier noch alles abfangen, damit kein Thread-Crash den Run killt
                paper = futures[future]
                print(f"[WARN] Unhandled exception for paper {paper[1].get('openalex_id','<unknown>')} – {e}")

    if all_output_records:
        output_df = pd.DataFrame(all_output_records)
        output_filename = "citation_analysis_results_for_final_data.csv"
        try:
            output_df.to_csv(output_filename, index=False)
            print(f"\n[DONE] Saved {len(all_output_records)} results to '{output_filename}'")
        except Exception as e:
            print(f"[ERROR] Could not save CSV ({e}).")
    else:
        print("\n[INFO] No results were generated to save.")
