# A Multi-Method Approach to Identifying Anchor Papers in Research Domains

This repository contains all scripts and components used in the bachelor thesis project titled  
**“A Multi-Method Approach to Identifying Anchor Papers in Research Domains”**.

The goal is to collect, analyze, and score research papers based on citation structure, content, and metadata, in order to identify statistically significant and conceptually central ("anchor") publications within a given research domain.

---

## 📦 Repository Structure 


| File | Purpose |
|------|---------|
| `GrabPapersAndData_Multithread.py` | Downloads papers and metadata from OpenAlex, fetches PDFs via OpenAlex / arXiv / Unpaywall. |
| `FullQualityRatingUsingOpenAI.py` | Assigns quality scores to papers using LLMs based on metadata. |
| `FullContentAnalysis.py` | Analyzes citation contexts using LLMs to infer the functional role (impact score) of cited papers. |
| `networkExportForGephi.py` | Exports citation graph (with weights and attributes) as GEXF for use in Gephi. |
| `LinearReg-WithConfInt.py` | Trains a regression model with confidence intervals for feature interpretation. |
| `GradientBoostModell.py` | Trains a gradient boosting model to predict anchor-paper status based on features. |


---

## 🚀 Pipeline Overview

The pipeline consists of 5 steps:

1. **Paper Collection**
   - Run `GrabPapersAndData_Multithread.py`
   - Downloads metadata and PDFs (if available) for all papers in the selected OpenAlex concept.
   - Script will return a folder full with PDFs and a metadata csv.

2. **Quality Rating**
   - Run `FullQualityRatingUsingOpenAI.py`
   - Uses OpenAI’s LLM to score papers based on metadata.
   - Returns a new csv.

3. **Citation Context Analysis**
   - Run `FullContentAnalysis.py`
   - For each citing paper, uses LLMs to extract and score the relevance of cited papers based on in-text mentions.
   - Returns a new csv.
   
4. **Network Export**
   - Run `networkExportForGephi.py`
   - Takes csvs from step 2 and 3 and generates a GEXF network for further visualisation and analysis in [Gephi](https://gephi.org/).
   - You can now calculate network metrics in Gephi.

5. **Model Training**
   - Export csv with full network metrics calculated in Gephi (sample provided in repo)
   - Run `GradientBoostModell.py` or `LinearReg-WithConfInt.py`
   - Trains predictive/explorative models for anchor-paper identification based on metadata and citation context scores.


---

## ⚙️ Setup & Requirements

### Dependencies

- Python 3.10+
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

## API Key
Add your own OpenAI API key in a .env file:
```bash
OPENAI_API_KEY_UNI=your_api_key_here
```

## ✒️ Notes

* Make sure to customise the configs (first few rows) of the python scripts to you liking / usecase.


## 🖼️ Images

This is what your Graph could look like:

<img width="1469" height="849" alt="Zitationsgraph" src="https://github.com/user-attachments/assets/37884cb2-464f-4a51-8447-85ae2bc7c1f9" />

<img width="1468" height="853" alt="Zitationsgraph2" src="https://github.com/user-attachments/assets/0c8bcb91-54d9-4e05-8637-fcbeb461904e" />


