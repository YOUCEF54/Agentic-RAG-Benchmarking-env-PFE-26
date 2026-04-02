import os
import time

import lancedb
import requests
from dotenv import load_dotenv
from pdf_oxide import PdfDocument
from semantic_text_splitter import TextSplitter
from sentence_transformers import SentenceTransformer


# ── CSV setup ──────────────────────────────────────────────────────────────────
CSV_FILE = "profile_results_py.csv"
CSV_HEADER = "run,pdf_read_ms,chunking_ms,model_embedding_ms,db_insert_ms,search_ms\n"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write(CSV_HEADER)

# ── One-time setup ─────────────────────────────────────────────────────────────
load_dotenv()
api_key    = os.getenv("API_KEY")
model_name = os.getenv("MODEL")
pdf_folder = os.getenv("PDF_FOLDER", "./pdfs")

# Collect all PDF paths from the folder
pdf_files = [
    os.path.join(pdf_folder, f)
    for f in os.listdir(pdf_folder)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in folder: {pdf_folder}")

print(f"Found {len(pdf_files)} PDF(s): {[os.path.basename(p) for p in pdf_files]}")

print("Loading embedding model (once)...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

# ── 100-iteration profiling loop ───────────────────────────────────────────────
NUM_RUNS = 100
context = ""

for run in range(1, NUM_RUNS + 1):
    print(f"\n{'─' * 50}")
    print(f"Run {run}/{NUM_RUNS}")

    # 1. PDF read & extract — all files in /pdfs ────────────────────────────────
    t0 = time.perf_counter()

    file_content = ""
    for pdf_path in pdf_files:
        doc = PdfDocument(pdf_path)
        page_count = doc.page_count() if callable(doc.page_count) else doc.page_count
        for i in range(page_count):
            page_text = doc.extract_text(i)
            if page_text:
                file_content += page_text + "\n\n"

    pdf_read_ms = (time.perf_counter() - t0) * 1000
    print(f"  PDF Read & Extract:    {pdf_read_ms:.4f} ms  ({len(pdf_files)} file(s))")

    # 2. Text chunking ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    splitter = TextSplitter(500)
    chunks = splitter.chunks(file_content)

    chunking_ms = (time.perf_counter() - t0) * 1000
    print(f"  Text Chunking:         {chunking_ms:.4f} ms  ({len(chunks)} chunks)")

    # 3. Bulk embedding ────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    vectors = embed_model.encode(chunks)

    embedding_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model/Embedding:       {embedding_ms:.4f} ms")

    # 4. DB insertion ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    db = lancedb.connect("./lance_db")
    data = [
        {"id": str(i), "text": chunks[i], "vector": vectors[i].tolist()}
        for i in range(len(chunks))
    ]
    table = db.create_table("docs", data, mode="overwrite")

    db_insert_ms = (time.perf_counter() - t0) * 1000
    print(f"  DB Init & Insertion:   {db_insert_ms:.4f} ms  ({table.count_rows()} rows)")

    # 5. Vector search ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    query_text = f"{BGE_QUERY_PREFIX}{QUESTION}"
    query_embedding = embed_model.encode(query_text)
    results = table.search(query_embedding.tolist()).limit(2).to_list()
    context = "\n\n".join(row["text"] for row in results)

    search_ms = (time.perf_counter() - t0) * 1000
    print(f"  Search:                {search_ms:.4f} ms")

    # ── Append this run to CSV ─────────────────────────────────────────────────
    with open(CSV_FILE, "a") as f:
        f.write(f"{run},{pdf_read_ms:.4f},{chunking_ms:.4f},{embedding_ms:.4f},{db_insert_ms:.4f},{search_ms:.4f}\n")

# ── Single LLM call after all profiling runs ───────────────────────────────────
print(f"\n{'─' * 50}")
print("Running LLM query (single call, not profiled)...")

answer_prompt = (
    "Use the following context to answer the question.\n\n"
    f"Context:\n{context}\n\nQuestion: {QUESTION}"
)
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
body    = {"model": model_name, "messages": [{"role": "user", "content": answer_prompt}]}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
try:
    print("\nAnswer:", response.json()["choices"][0]["message"]["content"])
except Exception:
    print("\nAnswer:", response.text)

print(f"\nProfiling complete. Results saved to '{CSV_FILE}'.")
