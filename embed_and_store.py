import json
import chromadb
from sentence_transformers import SentenceTransformer

# 1. CONFIG

CHUNKS_PATH = "./output/chunks.json"          
CHROMA_DIR = "./chroma_db"                    
COLLECTION_NAME = "dvt_medical_sources"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# 2. LOAD CHUNKS produced by parse_documents.py

def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 3. EMBED + STORE — encode every chunk and add it to a Chroma collection

def build_vector_store(chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to Chroma (persistent, on disk) ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source_name": c["source_name"],
            "title": c["title"],
            "url": c["url"],
        }
        for c in chunks
    ]

    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Writing to Chroma collection ...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    return collection, model


# 4. QUICK TEST — run a sample retrieval query to confirm it works end-to-end

def test_query(collection, model, query: str, top_k: int = 3):
    print(f"\nTest query: \"{query}\"")
    print("-" * 70)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        print(f"\n[{rank}] source: {meta['source_name']}  |  distance: {dist:.4f}")
        print(f"    url: {meta['url']}")
        preview = doc[:200].replace("\n", " ")
        print(f"    text: {preview}...")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    collection, model = build_vector_store(chunks)

    test_query(collection, model, "What are the symptoms of DVT?")
    test_query(collection, model, "How can I prevent a blood clot on a long flight?")
    test_query(collection, model, "What is the link between DVT and pulmonary embolism?")