# ProcureAI — BIS Compliance RAG Engine

A 100% air-gapped Retrieval-Augmented Generation (RAG) engine designed to extract technical specifications and Indian Standard (IS) codes from messy government tender PDFs.

## Architecture
1. **`extract_clean_pdf.py`**: Ingests raw PDFs, strips watermarks, applies OCR typo corrections, and extracts tables into structured Markdown.
2. **`ingest.py`**: Chunks the Markdown structurally and builds a local ChromaDB vector store using `all-MiniLM-L6-v2`.
3. **`app.py`**: A Streamlit frontend that queries the vector store and uses a local Llama 3.2 model (via Ollama) to generate audit-ready compliance reports.

## How to Run
1. Place your raw PDF standards into a folder named `raw_pdfs/`.
2. Run the extraction pipeline: `python extract_clean_pdf.py`
3. Build the vector database: `python ingest.py`
4. Launch the dashboard: `streamlit run app.py`