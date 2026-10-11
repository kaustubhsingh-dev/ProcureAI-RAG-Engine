import os
import shutil
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
    # Secondary splitter to keep chunks strictly under all-MiniLM-L6-v2 context limits (256 tokens)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    all_chunks = []
    
    print(f"Loading and intelligently chunking {len(markdown_files)} documents...")
    for file_path in markdown_files:
        content, file_metadata = parse_markdown_with_metadata(file_path)
        
        # 1. Structural page split
        page_chunks = markdown_splitter.split_text(content)
        
        # 2. Token-safe sub-chunking
        for page_chunk in page_chunks:
            sub_chunks = text_splitter.split_documents([page_chunk])
            for sub in sub_chunks:
                sub.metadata.update(file_metadata)
                all_chunks.append(sub)

    print(f"Created {len(all_chunks)} sub-chunks fitted for vector embedding context limits.")
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