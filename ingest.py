import os
import shutil
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data" 
CHROMA_PATH = "chroma_db"

def parse_markdown_with_metadata(file_path):
    """Extracts the custom metadata block and returns the clean text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    metadata = {}
    meta_match = re.search(r'---METADATA---\n(.*?)\n--------------', content, flags=re.DOTALL)
    
    if meta_match:
        meta_lines = meta_match.group(1).strip().split('\n')
        for line in meta_lines:
            if ':' in line:
                key, val = line.split(':', 1)
                metadata[key.strip()] = val.strip()
                
        content = content.replace(meta_match.group(0), '')
        
    return content, metadata

def main():
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing old database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} folder not found.")
        return

    markdown_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.md')]
    
    headers_to_split_on = [
        ("##", "Page")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    all_chunks = []
    
    print(f"Loading and intelligently chunking {len(markdown_files)} documents...")
    for file_path in markdown_files:
        content, file_metadata = parse_markdown_with_metadata(file_path)
        
        doc_chunks = markdown_splitter.split_text(content)
        for chunk in doc_chunks:
            chunk.metadata.update(file_metadata)
            all_chunks.append(chunk)

    print(f"Created {len(all_chunks)} table-safe chunks with attached cross-reference metadata.")
    print("Generating embeddings and building the final ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("✅ Air-gapped database built successfully! Ready for production.")

if __name__ == "__main__":
    main()