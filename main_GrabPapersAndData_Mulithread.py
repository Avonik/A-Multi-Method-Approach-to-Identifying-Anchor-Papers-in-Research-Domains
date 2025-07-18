import requests
import os
import re
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Dieses Skripit sammeslt erst "max_results" Paper von OpenAlex, die den Filter "search_query_title" erfüllen.
Dann werden diese Paper parallel heruntergeladen und validiert, dass die Dateien auch groß genug sind und nicht defekt.
Das Herunterladen erfolgt aus drei Quellen: 
1. OpenAlex (beste OA-Location) 2. arXiv (wenn DOI zu arXiv passt) 3. Unpaywall (wenn OpenAlex und arXiv nichts liefern)
Die Metadaten der Paper werden in "metadata_file" gespeichert.
"""

# ==== Konfiguration ====
search_query_title = ""
additional_filters = "concepts.id:C119857082,is_oa:true"
#neural networks: C50644808
#machine learning: C119857082
max_results = 10000
workers = 20
unpaywall_email = "Your_email"
download_dir = r"F:\\PaperBA\\10000xML_oa"
metadata_file = os.path.join(download_dir, "download_metadata.csv")
os.makedirs(download_dir, exist_ok=True)
MIN_FILE_SIZE_BYTES = 25 * 1024

# === PDF Download und Validierung ===
def download_and_validate_pdf(pdf_url, target_filename, source_info="Unknown"):
    """
    Lädt ein PDF von der gegebenen URL herunter und validiert:
    - ob es sich um ein echtes PDF handelt (erkennt an '%PDF')
    - ob es eine Mindestgröße überschreitet (nicht leer oder beschädigt)
    - speichert es lokal ab, wenn gültig
    """

    if not pdf_url:
        return False
    try:
        pdf_response = requests.get(pdf_url, timeout=30, stream=True)
        pdf_response.raise_for_status()
        content_type = pdf_response.headers.get('Content-Type', '').lower()
        content_buffer = b""
        is_pdf_content = False
        for chunk in pdf_response.iter_content(chunk_size=8192):
            if not content_buffer and b"%PDF" in chunk[:1024]:
                is_pdf_content = True
            content_buffer += chunk
        if not is_pdf_content:
            return False
        with open(target_filename, "wb") as f:
            f.write(content_buffer)
        if os.path.exists(target_filename) and os.path.getsize(target_filename) > MIN_FILE_SIZE_BYTES:
            return True
        else:
            if os.path.exists(target_filename):
                os.remove(target_filename)
            return False
    except Exception:
        return False

# === Ein Paper verarbeiten ===
def process_paper(paper):
    """
        Verarbeitet ein einzelnes Paper:
        - extrahiert Metadaten
        - klassifiziert den Publikationstyp
        - versucht das PDF über OpenAlex, arXiv oder Unpaywall herunterzuladen
        - gibt Metadaten inkl. PDF-Pfad zurück, wenn Download erfolgreich
    """

    openalex_id = paper.get("id")
    doi = paper.get("doi")
    title = paper.get("title", "Kein Titel")
    if not doi:
        return None

    doi_clean = doi.replace("https://doi.org/", "")
    filename_safe = doi_clean.replace("/", "_").replace(":", "_")
    local_pdf_path = os.path.join(download_dir, f"{filename_safe}.pdf")

    referenced_works = paper.get("referenced_works", [])
    current_metadata = {
        "openalex_id": openalex_id,
        "doi": doi,
        "title": title,
        "publication_year": paper.get("publication_year"),
        "venue": paper.get("host_venue", {}).get("display_name", ""),
        "authors": ", ".join([a.get("author", {}).get("display_name", "N/A") for a in paper.get("authorships", [])]),
        "cited_by_count": paper.get("cited_by_count", 0),
        "referenced_works": referenced_works,
        "num_referenced_works": len(referenced_works),
        "local_pdf_path": "",
        "pdf_source_url": "",
        "download_source": ""
    }

    # ==== Erweiterte Metadaten ====
    primary_location = paper.get("primary_location") or {}
    source = primary_location.get("source")
    if not source:
        for loc in paper.get("locations", []):
            source = loc.get("source")
            if source:
                break

    source_name = source.get("display_name", "Keine Quelle") if source else "Keine Quelle"
    publisher_id = source.get("host_organization") if source else "Unbekannt"
    publisher_name = source.get("host_organization_name") if source else "Unbekannt"
    pub_type = paper.get("type_crossref", "Unbekannt")
    doi_lower = doi_clean.lower()

    # Klassifikation
    if "arxiv" in doi_lower or "arxiv" in source_name.lower():
        pub_label = "Preprint (arXiv)"
    elif "cvpr" in doi_lower:
        pub_label = "Conference (CVPR)"
    elif "iccv" in doi_lower:
        pub_label = "Conference (ICCV)"
    elif "ieee" in doi_lower:
        pub_label = "Conference (IEEE)"
    elif "springer" in doi_lower:
        pub_label = "Springer Book Chapter"
    elif "acm" in doi_lower:
        pub_label = "ACM Publication"
    elif pub_type == "journal-article":
        pub_label = "Peer-reviewed"
    else:
        pub_label = "Unbekannt"



    current_metadata.update({
        "source_display_name": source_name,
        "publisher_id": publisher_id,
        "publisher_name": publisher_name,
        "type_crossref": pub_type,
        "classification": pub_label,
    })

    # === Download PDF ===
    pdf_downloaded_successfully = False
    best_oa_location = paper.get("best_oa_location")
    if best_oa_location and best_oa_location.get("is_oa") and best_oa_location.get("pdf_url"):
        pdf_url = best_oa_location["pdf_url"]
        if download_and_validate_pdf(pdf_url, local_pdf_path):
            current_metadata["pdf_source_url"] = pdf_url
            current_metadata["download_source"] = "OpenAlex"
            pdf_downloaded_successfully = True

    arxiv_match = re.match(r"10\\.48550/arxiv\\.(\\d{4}\\.\\d{4,5}(v\\d+)?)", doi_lower)
    if not pdf_downloaded_successfully and arxiv_match:
        arxiv_id = arxiv_match.group(1)
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if download_and_validate_pdf(arxiv_url, local_pdf_path):
            current_metadata["pdf_source_url"] = arxiv_url
            current_metadata["download_source"] = "arXiv"
            pdf_downloaded_successfully = True

    if not pdf_downloaded_successfully:
        unpaywall_api_url = f"https://api.unpaywall.org/v2/{doi_clean}?email={unpaywall_email}"
        try:
            r = requests.get(unpaywall_api_url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                oa_location = data.get("best_oa_location")
                if oa_location and oa_location.get("url_for_pdf"):
                    pdf_url = oa_location["url_for_pdf"]
                    if download_and_validate_pdf(pdf_url, local_pdf_path):
                        current_metadata["pdf_source_url"] = pdf_url
                        current_metadata["download_source"] = "Unpaywall"
                        pdf_downloaded_successfully = True
        except:
            pass

    if pdf_downloaded_successfully:
        current_metadata["local_pdf_path"] = local_pdf_path
        return current_metadata
    else:
        return None

# === Suche starten ===
def fetch_all_openalex_results(filters, max_total_results=1000, per_page=200):
    """
        Ruft mit Cursor-Pagination alle passenden Paper von der OpenAlex-API ab (bis zu max_total_results).
        Nutzt TQDM zur Fortschrittsanzeige.
    """

    all_results, cursor, total_fetched = [], "*", 0
    base_url = "https://api.openalex.org/works"
    with tqdm(total=max_total_results, desc="Lade OpenAlex-Ergebnisse") as pbar:
        while total_fetched < max_total_results:
            url = f"{base_url}?filter={filters}&per-page={per_page}&cursor={cursor}&mailto={unpaywall_email}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            total_fetched += len(results)
            pbar.update(len(results))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
    return all_results





# === Hauptskript ===
filters = f"title.search:{search_query_title}"
if additional_filters:
    filters += f",{additional_filters}"
print(f"🔍 Suche nach Papers mit Filter: {filters}")

papers = fetch_all_openalex_results(filters, max_total_results=max_results)
if not papers:
    print("❌ Keine Paper gefunden.")
    exit()

print("🚀 Starte parallele Verarbeitung mit Threading...")
downloaded_papers_metadata, processed_dois = [], set()
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(process_paper, p): p for p in papers}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Verarbeite Paper"):
        result = future.result()
        if result:
            doi_clean = result["doi"].replace("https://doi.org/", "")
            if doi_clean not in processed_dois:
                downloaded_papers_metadata.append(result)
                processed_dois.add(doi_clean)

if downloaded_papers_metadata:
    print(f"\n💾 Speichere Metadaten in {metadata_file}...")
    fieldnames = [
        "openalex_id", "doi", "title", "authors", "publication_year",
        "venue", "source_display_name", "publisher_id", "publisher_name",
        "type_crossref", "classification",
        "cited_by_count", "referenced_works", "num_referenced_works",
        "local_pdf_path", "pdf_source_url", "download_source"
    ]
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in downloaded_papers_metadata:
            writer.writerow(row)
    print("✅ Metadaten gespeichert.")
else:
    print("⚠️  Keine gültigen Paper heruntergeladen. Keine Metadaten-Datei erstellt.")

print("\n🏁 Skript abgeschlossen.")
