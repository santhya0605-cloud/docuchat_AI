 DocuChat AI — LLM-Powered Document Question Answering System

 1.Project Overview

DocuChat AI is a Retrieval-Augmented Generation (RAG) based document question-answering system that allows users to ask questions about information contained in PDF documents.

The system processes PDF documents, extracts their text, divides the text into smaller chunks, converts the chunks into embeddings, and retrieves relevant content based on the user's question. The retrieved information is then used to generate a context-based answer.

2. How It Works

The project follows these main steps:

1. PDF Upload / Document Collection
   - PDF documents are collected as the knowledge source.

2. Text Extraction
   - Text is extracted from the PDF documents using `PyPDF`.

3. Text Chunking
   - Extracted text is divided into smaller overlapping chunks.
   - This makes it easier to search for relevant information.

4. Text Embeddings
   - Text chunks are converted into numerical vector representations using a Sentence Transformer model.

5. Similarity Search
   - The embeddings are used to identify chunks that are most relevant to the user's question.

6. Context Retrieval
   - Relevant document chunks are retrieved and provided as context.

7. Answer Generation
   - An LLM uses the retrieved context to generate a response to the user's question.

3.Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- Hugging Face Transformers
- RAG (Retrieval-Augmented Generation)
- Natural Language Processing (NLP)

4. Project Structure

```text
DocuChat-AI/
│
├── app.py
├── notebook.py
├── requirements.txt
├── .gitignore
└── research_articles/
    └── PDF documents
