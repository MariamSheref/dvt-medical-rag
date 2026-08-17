import json
from collections import Counter

with open("./output/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("=" * 70)
print(f"TOTAL CHUNKS: {len(chunks)}")
print("=" * 70)

# 1) How many chunks came from each source? (should be > 0 for both)
counts = Counter(c["source_name"] for c in chunks)
print("\nChunks per source:")
for source, count in counts.items():
    print(f"  {source}: {count} chunks")

# 2) Show 3 sample chunks (first, middle, last) so you can eyeball quality
print("\n" + "=" * 70)
print("SAMPLE CHUNKS")
print("=" * 70)

sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
for i in sample_indices:
    c = chunks[i]
    print(f"\n--- Chunk {c['id']}  (source: {c['source_name']}) ---")
    print(f"Title: {c['title']}")
    print(f"URL:   {c['url']}")
    print(f"Length: {len(c['text'])} chars")
    print("Text:")
    print(c["text"])
    print("-" * 70)

# 3) Check for leftover extraction artifacts that should have been cleaned
print("\n" + "=" * 70)
print("ARTIFACT CHECK (should all say 'OK — none found')")
print("=" * 70)

import re
checks = {
    "cid: font codes":     r"\(cid:\d+\)",
    "'X of Y' page numbers": r"\b\d+\s+of\s+\d+\b",
    "non-ASCII characters": r"[^\x20-\x7E\n]",
    "triple+ newlines":     r"\n{3,}",
}

all_text = "\n".join(c["text"] for c in chunks)
for label, pattern in checks.items():
    matches = re.findall(pattern, all_text)
    status = "OK — none found" if not matches else f"FOUND {len(matches)} — e.g. {matches[:3]}"
    print(f"  {label}: {status}")

# 4) Check chunk sizes are within a sane range (no empty or giant chunks)
lengths = [len(c["text"]) for c in chunks]
print("\n" + "=" * 70)
print("CHUNK SIZE STATS")
print("=" * 70)
print(f"  Min length: {min(lengths)} chars")
print(f"  Max length: {max(lengths)} chars")
print(f"  Avg length: {sum(lengths) / len(lengths):.0f} chars")

empty_chunks = [c for c in chunks if len(c["text"].strip()) == 0]
print(f"  Empty chunks: {len(empty_chunks)}  (should be 0)")