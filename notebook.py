from pathlib import Path
from pypdf import PdfReader


# ==========================================
# STEP 1: FIND PDF FILES
# ==========================================

project_folder = Path(__file__).parent
pdf_folder = project_folder / "research_articles"

pdf_files = list(pdf_folder.rglob("*.pdf"))

print("================================")
print("DOCUCHAT AI")
print("================================")

print("PDF files found:", len(pdf_files))


# ==========================================
# STEP 2: EXTRACT TEXT FROM PDFs
# ==========================================

documents = []

for pdf_file in pdf_files:

    print("\nReading:", pdf_file.name)

    try:
        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append({
            "filename": pdf_file.name,
            "text": text
        })

        print("Characters extracted:", len(text))

    except Exception as e:

        print("Error reading:", pdf_file.name)
        print("Error:", e)


print("\n================================")
print("TEXT EXTRACTION COMPLETED")
print("Documents:", len(documents))
print("================================")


# ==========================================
# STEP 3: CHUNK THE TEXT
# ==========================================

chunk_size = 500
chunk_overlap = 50

chunks = []

for document in documents:

    text = document["text"]

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append({
                "filename": document["filename"],
                "text": chunk
            })

        start = end - chunk_overlap


print("\n================================")
print("CHUNKING COMPLETED")
print("================================")

print("Total chunks:", len(chunks))


# ==========================================
# STEP 4: SHOW SOME CHUNKS
# ==========================================

print("\n========== SAMPLE CHUNKS ==========")

for i in range(min(5, len(chunks))):

    print("\nChunk", i + 1)

    print("Source:", chunks[i]["filename"])

    print("Text:")
    print(chunks[i]["text"][:300])

    print("----------------------------------")
    # ==========================================
# STEP 3: CREATE EMBEDDINGS
# ==========================================

from sentence_transformers import SentenceTransformer

print("\n================================")
print("LOADING EMBEDDING MODEL")
print("================================")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ==========================================
# CREATE TEXT LIST
# ==========================================

chunk_texts = [chunk["text"] for chunk in chunks]


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True
)

print("\n================================")
print("EMBEDDINGS CREATED")
print("================================")

print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)
# ==========================================
# STEP 4: CREATE FAISS VECTOR DATABASE
# ==========================================

import faiss
import numpy as np

print("\n================================")
print("CREATING FAISS VECTOR DATABASE")
print("================================")


# Convert embeddings to NumPy array
embeddings = np.array(embeddings).astype("float32")


# Get embedding dimension
dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)


# Create FAISS index
index = faiss.IndexFlatL2(dimension)


# Add embeddings to FAISS
index.add(embeddings)


print("FAISS index created successfully.")

print("Total vectors stored:", index.ntotal)
# ==========================================
# STEP 4.1: TEST VECTOR SEARCH
# ==========================================

query = "What are quasi-periodic eruptions?"

print("\n================================")
print("TESTING SEARCH")
print("================================")

# Convert question into embedding
query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# Search top 5 relevant chunks
distances, indices = index.search(query_embedding, 5)


print("\nQuestion:", query)

print("\nTop 5 relevant chunks:")

for rank, idx in enumerate(indices[0]):

    print("\n-----------------------------")
    print("Rank:", rank + 1)
    print("Source:", chunks[idx]["filename"])
    print("Distance:", distances[0][rank])
    print("Text:")
    print(chunks[idx]["text"][:500])
    # ==========================================
# STEP 5: LOAD GENAI MODEL
# ==========================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("\n================================")
print("LOADING GENAI MODEL")
print("================================")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("GenAI model loaded successfully!")


# ==========================================
# STEP 5.1: ASK QUESTION
# ==========================================

question = "What are quasi-periodic eruptions?"

print("\nUser Question:")
print(question)


# ==========================================
# STEP 5.2: RETRIEVE RELEVANT CHUNKS
# ==========================================

question_embedding = model.encode([question])

question_embedding = np.array(question_embedding).astype("float32")

distances, indices = index.search(question_embedding, 5)


# ==========================================
# STEP 5.3: BUILD CONTEXT
# ==========================================

context = ""

for idx in indices[0]:

    context += chunks[idx]["text"] + "\n\n"


# ==========================================
# STEP 5.4: CREATE PROMPT
# ==========================================

prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""


# ==========================================
# STEP 5.5: GENERATE ANSWER
# ==========================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
)

outputs = generator_model.generate(
    **inputs,
    max_new_tokens=200
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print("\n================================")
print("DOCUCHAT AI ANSWER")
print("================================")

print(answer)
