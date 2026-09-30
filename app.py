import streamlit as st
from pathlib import Path
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuChat AI")
st.write("Ask questions from your research papers.")


# =========================================================
# FIND PDF FILES
# =========================================================

def get_pdf_files():

    project_folder = Path(__file__).parent
    pdf_folder = project_folder / "research_articles"

    if not pdf_folder.exists():
        return []

    return sorted(
        pdf_folder.rglob("*.pdf"),
        key=lambda x: x.name.lower()
    )


# =========================================================
# LOAD PDF DOCUMENTS
# =========================================================

@st.cache_data
def load_documents(pdf_signature):

    pdf_files = get_pdf_files()

    documents = []

    for pdf_file in pdf_files:

        try:

            reader = PdfReader(pdf_file)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            if text.strip():

                documents.append({
                    "filename": pdf_file.name,
                    "text": text
                })

        except Exception as e:

            print(f"Error reading {pdf_file.name}: {e}")

    return documents


# =========================================================
# CREATE CHUNKS
# =========================================================

@st.cache_data
def create_chunks(documents):

    chunks = []

    chunk_size = 1000
    chunk_overlap = 150

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append({
                    "filename": document["filename"],
                    "text": chunk
                })

            start = end - chunk_overlap

            if start >= len(text):
                break

    return chunks


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =========================================================
# CREATE FAISS INDEX
# =========================================================

@st.cache_resource
def create_index(chunks, chunk_signature):

    embedding_model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# LOAD GENERATIVE AI MODEL
# =========================================================

@st.cache_resource
def load_genai_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def get_question_terms(question):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower()
    )

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "why",
        "how",
        "does",
        "do",
        "can",
        "and",
        "or",
        "with",
        "from",
        "this",
        "that",
        "it",
        "be",
        "used",
        "mean",
        "called"
    }

    terms = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    return terms


# =========================================================
# KEYWORD MATCH SCORE
# =========================================================

def keyword_score(question, text):

    terms = get_question_terms(question)

    if not terms:
        return 0.0

    text_lower = text.lower()

    matched = 0

    for term in terms:

        if term in text_lower:
            matched += 1

    return matched / len(terms)


# =========================================================
# PHRASE MATCH SCORE
# =========================================================

def phrase_score(question, text):

    question_clean = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        question.lower()
    )

    question_clean = re.sub(
        r"\s+",
        " ",
        question_clean
    ).strip()

    text_clean = text.lower()

    # Remove common question phrases
    phrases_to_remove = [
        "what is",
        "what are",
        "why is",
        "why are",
        "how does",
        "how do",
        "explain",
        "define"
    ]

    concept = question_clean

    for phrase in phrases_to_remove:

        if concept.startswith(phrase):

            concept = concept[len(phrase):].strip()

            break

    if len(concept) < 3:
        return 0.0

    if concept in text_clean:
        return 1.0

    return 0.0


# =========================================================
# DEFINITION BONUS
# =========================================================

def definition_bonus(question, text):

    question_terms = get_question_terms(question)

    if not question_terms:
        return 0.0

    text_lower = text.lower()

    # Definition-style phrases commonly used in papers
    definition_patterns = [
        "is defined as",
        "are defined as",
        "refers to",
        "is the process of",
        "is a process",
        "can be defined",
        "is described as",
        "means that",
        "is known as"
    ]

    has_definition_language = any(
        pattern in text_lower
        for pattern in definition_patterns
    )

    if not has_definition_language:
        return 0.0

    matched = sum(
        1
        for term in question_terms
        if term in text_lower
    )

    if matched == 0:
        return 0.0

    return min(
        1.0,
        0.5 + 0.1 * matched
    )


# =========================================================
# CURRENT PDF FILES
# =========================================================

pdf_files = get_pdf_files()

pdf_signature = tuple(
    (
        str(pdf),
        pdf.stat().st_size,
        pdf.stat().st_mtime_ns
    )
    for pdf in pdf_files
)


# =========================================================
# SHOW PDF COUNT
# =========================================================

st.info(
    f"Current PDF files found: {len(pdf_files)}"
)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

with st.spinner(
    "Reading research papers..."
):

    documents = load_documents(
        pdf_signature
    )


if not documents:

    st.error(
        "No PDF files were found in "
        "'research_articles'."
    )

    st.stop()


# =========================================================
# CREATE CHUNKS
# =========================================================

with st.spinner(
    "Preparing research content..."
):

    chunks = create_chunks(
        documents
    )


# =========================================================
# CHUNK SIGNATURE
# =========================================================

chunk_signature = (
    len(chunks),
    sum(
        len(chunk["text"])
        for chunk in chunks
    )
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

with st.spinner(
    "Loading embedding model..."
):

    embedding_model = load_embedding_model()


# =========================================================
# CREATE FAISS INDEX
# =========================================================

with st.spinner(
    "Creating searchable research index..."
):

    index = create_index(
        chunks,
        chunk_signature
    )


# =========================================================
# LOAD GENAI MODEL
# =========================================================

with st.spinner(
    "Loading GenAI model..."
):

    tokenizer, generator_model = (
        load_genai_model()
    )


# =========================================================
# STATUS
# =========================================================

st.success(
    f"Loaded {len(documents)} PDFs "
    f"and {len(chunks)} text chunks."
)


# =========================================================
# DOCUMENT LIST
# =========================================================

with st.expander(
    "📚 View loaded research papers"
):

    for document in documents:

        st.write(
            f"• {document['filename']}"
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your research papers..."
)


# =========================================================
# ANSWER
# =========================================================

if question:

    st.chat_message(
        "user"
    ).write(question)


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching research papers and generating answer..."
        ):

            # =================================================
            # QUESTION EMBEDDING
            # =================================================

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )

            question_embedding = np.asarray(
                question_embedding,
                dtype="float32"
            )

            faiss.normalize_L2(
                question_embedding
            )


            # =================================================
            # SEMANTIC SEARCH
            # =================================================

            SEARCH_K = min(
                30,
                len(chunks)
            )

            semantic_scores, semantic_indices = (
                index.search(
                    question_embedding,
                    SEARCH_K
                )
            )


            # =================================================
            # COMBINE SEMANTIC + KEYWORD + DEFINITION
            # =================================================

            candidates = []

            for semantic_score, idx in zip(
                semantic_scores[0],
                semantic_indices[0]
            ):

                text = chunks[idx][
                    "text"
                ]

                semantic_score = float(
                    semantic_score
                )

                semantic_score = max(
                    0.0,
                    min(
                        1.0,
                        semantic_score
                    )
                )

                key_score = keyword_score(
                    question,
                    text
                )

                phrase_match = phrase_score(
                    question,
                    text
                )

                definition_match = (
                    definition_bonus(
                        question,
                        text
                    )
                )


                # ---------------------------------------------
                # FINAL HYBRID SCORE
                # ---------------------------------------------

                final_score = (
                    0.50 * semantic_score
                    +
                    0.25 * key_score
                    +
                    0.15 * phrase_match
                    +
                    0.10 * definition_match
                )


                candidates.append(
                    {
                        "idx": idx,
                        "score": final_score,
                        "semantic": semantic_score,
                        "keyword": key_score,
                        "phrase": phrase_match,
                        "definition": definition_match
                    }
                )


            # =================================================
            # SORT BY FINAL SCORE
            # =================================================

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )


            # =================================================
            # SELECT RESULTS
            # Try to use different papers when relevant
            # =================================================

            selected = []

            source_count = {}

            for candidate in candidates:

                idx = candidate["idx"]

                filename = chunks[idx][
                    "filename"
                ]

                current_source_count = (
                    source_count.get(
                        filename,
                        0
                    )
                )


                # Don't let one paper dominate
                if current_source_count >= 3:
                    continue


                selected.append(
                    candidate
                )

                source_count[filename] = (
                    current_source_count + 1
                )


                if len(selected) >= 8:
                    break


            # =================================================
            # BUILD CONTEXT
            # =================================================

            context_parts = []

            sources = []

            max_context_chars = 6500

            current_chars = 0

            for candidate in selected:

                idx = candidate["idx"]

                chunk = chunks[idx]

                text = chunk[
                    "text"
                ].strip()

                if not text:
                    continue


                if (
                    current_chars
                    + len(text)
                    > max_context_chars
                ):
                    break


                context_parts.append(
                    f"""
Research paper:
{chunk['filename']}

Research content:
{text}
"""
                )

                current_chars += len(text)

                sources.append(
                    chunk["filename"]
                )


            sources = list(
                dict.fromkeys(
                    sources
                )
            )


            context = "\n\n".join(
                context_parts
            )


            # =================================================
            # GENERATION PROMPT
            # =================================================

            prompt = f"""
You are DocuChat AI, a research assistant.

Answer the user's question using ONLY the research
information below.

IMPORTANT:

1. Give a direct answer to the question.
2. Explain the concept clearly.
3. Use your own words.
4. Prefer information that directly defines or explains
   the concept asked about.
5. Combine relevant information when several research
   papers discuss the same concept.
6. Do not answer using an unrelated topic just because
   it appears in the context.
7. Do not output a paper title.
8. Do not output only keywords.
9. Do not invent facts.
10. Give 3 to 5 complete sentences.

If the research information does not contain enough
information to answer the question, say:

"I could not find enough information in the research papers."

Research information:

{context}

User question:

{question}

Answer:
"""


            # =================================================
            # TOKENIZATION
            # =================================================

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )


            # =================================================
            # GENERATE
            # =================================================

            outputs = generator_model.generate(
                **inputs,
                max_new_tokens=180,
                min_new_tokens=40,
                do_sample=False
            )


            # =================================================
            # DECODE
            # =================================================

            answer = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.markdown(
                "### 🤖 Answer"
            )

            st.write(
                answer
            )


            # =================================================
            # DISPLAY SOURCES ONLY
            # =================================================

            st.markdown(
                "### 📄 Sources"
            )

            for source in sources:

                st.write(
                    f"- {source}"
                )