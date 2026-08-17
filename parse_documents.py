import os
import re
import json
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONFIG — map each PDF to the source metadata you want cited in answers

DATA_DIR = "./data"

SOURCES = [
    {
        "filename": "CDC_About_VTE_DVT.pdf",
        "source_name": "CDC",
        "title": "About Venous Thromboembolism (Blood Clots)",
        "url": "https://www.cdc.gov/blood-clots/about/index.html",
    },
    {
        "filename": "NHS_DVT.pdf",
        "source_name": "NHS",
        "title": "DVT (deep vein thrombosis)",
        "url": "https://www.nhs.uk/conditions/deep-vein-thrombosis-dvt/",
    },
    {
        "filename": "DVT_Symptoms_and_causes.pdf",
        "source_name": "Mayo Clinic",
        "title": "Deep vein thrombosis (DVT) - Symptoms & causes",
        "url": "https://www.mayoclinic.org/diseases-conditions/deep-vein-thrombosis/symptoms-causes/syc-20352557",
    },
    {
        "filename": "DVT_Diagnosis_and_treatment.pdf",
        "source_name": "Mayo Clinic",
        "title": "Deep vein thrombosis (DVT) - Diagnosis & treatment",
        "url": "https://www.mayoclinic.org/diseases-conditions/deep-vein-thrombosis/diagnosis-treatment/drc-20352563",
    },
]


# 2. DOCUMENT LOADING — extract raw text page by page

def load_pdf_text(filepath: str) -> str:
    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n".join(pages_text)


# 3. TEXT CLEANING — remove noise so embeddings aren't polluted by junk

def clean_text(raw_text: str) -> str:
   
    text = raw_text
    text = re.sub(r"\b\d+\s+of\s+\d+\b", " ", text)
    text = re.sub(r"[•◦▪‣]", "-", text)
    text = re.sub(r"\(cid:\d+\)", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    return text.strip()

# 4. CHUNKING — split into overlapping, semantically coherent chunks

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80):
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# 5. PIPELINE — run load -> clean -> chunk for every source, attach metadata

def build_chunks():
    all_chunks = []
    chunk_id = 0

    for source in SOURCES:
        filepath = os.path.join(DATA_DIR, source["filename"])
        if not os.path.exists(filepath):
            print(f"[WARN] Missing file, skipping: {filepath}")
            continue

        print(f"Loading: {source['filename']}")
        raw_text = load_pdf_text(filepath)

        print(f"Cleaning text ({len(raw_text)} raw chars)...")
        cleaned = clean_text(raw_text)

        print(f"Chunking ({len(cleaned)} clean chars)...")
        chunks = chunk_text(cleaned)

        for chunk in chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_id:04d}",
                "text": chunk,
                "source_name": source["source_name"],  
                "title": source["title"],
                "url": source["url"],
            })
            chunk_id += 1

        print(f" -> {len(chunks)} chunks from {source['source_name']}\n")

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()

    os.makedirs("./output", exist_ok=True)
    out_path = "./output/chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(chunks)} total chunks written to {out_path}")
    print("\nSample chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False) if chunks else "No chunks produced.")