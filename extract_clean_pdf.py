import os
import re
import shutil
import fitz   
import pdfplumber
from pathlib import Path

INPUT_FOLDER = "raw_pdfs"    
OUTPUT_FOLDER = "data"       
ARCHIVE_FOLDER = "archive_pdfs"

# --- OCR TYPO CORRECTIONS ---
# Add new entries here as new document-specific typos are discovered.
# Pattern -> Replacement (case-insensitive)
KNOWN_OCR_FIXES = {
    r'\bCat-bon\b': 'Carbon',
    r'\bCat-bot\b': 'Carbon',
    r'\bCatbon\b': 'Carbon',
    r'\bfimm\+\b': 'following',
    r'\~e4\~5': 'Fe 415',
    r'\~e yJj': 'Fe 500',
}

def clean_text(text):
    """Strips watermarks, applies known OCR typo fixes, and normalizes spacing."""
    watermark_pattern = r"Free Standard provided by BIS via BSB Edge Private Limited.*?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\."
    text = re.sub(watermark_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    reaffirmed_pattern = r"\(Reaffirmed\s*\d{4}\)"
    text = re.sub(reaffirmed_pattern, "", text, flags=re.IGNORECASE)
    
    # Apply documented OCR corrections
    for pattern, replacement in KNOWN_OCR_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Systemic spacing fixes
    text = re.sub(r'(?<!\n)\n([a-z])', r' \1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def format_table(table_data):
    """Converts a list of lists (table) into strict Markdown format."""
    if not table_data or len(table_data) == 0:
        return ""
    
    clean_rows = []
    for row in table_data:
        clean_row = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
        if any(clean_row): 
            clean_rows.append(clean_row)
            
    if not clean_rows:
        return ""
        
    table_str = "\n"
    for i, row in enumerate(clean_rows):
        line = "| " + " | ".join(row) + " |"
        table_str += line + "\n"
        if i == 0:
            separator = "|" + "|".join(["---"] * len(row)) + "|"
            table_str += separator + "\n"
            
    return table_str + "\n"

def extract_metadata(full_text):
    """
    Hunts for Clause 2 (References / Normative References).
    Uses a HYBRID regex that captures BOTH:
      - Explicit "IS xxx" references
      - Bare "xxx:yyyy" references (colon-year = definitely a standard, not a random number)
    This avoids the earlier year-range filter that would incorrectly exclude valid codes like IS 2062.
    """
    refs = "None identified"
    
    patterns = [
        r'2\.?\s*(?:NORMATIVE\s+)?REFERENCES?(.*?)(?:\n\s*3\.?\s*(?:TERMINOLOGY|TERMS|DEFINITIONS|MATERIAL|GENERAL)|4\.?\s*(?:GENERAL|MATERIAL|REQUIREMENTS))',
        r'2\.?\s*(?:NORMATIVE\s+)?REFERENCES?(.*?)(?:\n\s*3\.)',
        r'REFERENCES?(.*?)(?:TERMINOLOGY|TERMS\s+AND\s+DEFINITIONS)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            block = match.group(1)[:4000]
            
            # Pattern A: Explicit "IS xxx" or "IS: xxx"
            explicit = re.findall(r'\bIS\s*[:]?\s*(\d{1,5})\b', block, flags=re.IGNORECASE)
            
            # Pattern B: Bare "xxx:yyyy" (colon followed by a 4-digit year → definitely a standard)
            with_year = re.findall(r'\b(\d{1,5})\s*:\s*(?:19|20)\d{2}\b', block)
            
            # Merge and deduplicate while preserving order
            seen = set()
            flat_codes = []
            for c in explicit + with_year:
                if c not in seen:
                    seen.add(c)
                    flat_codes.append(c)
            
            if flat_codes:
                # Sort numerically for consistent output
                unique_codes = sorted(set(flat_codes), key=int)
                formatted = [f"IS {code}" for code in unique_codes]
                refs = ", ".join(formatted[:15])
                break
    
    return refs

def process_pdf(pdf_path, output_path, archive_path):
    print(f"Processing: {os.path.basename(pdf_path)}...")
    
    try:
        page_tables = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                formatted_tables = [format_table(t) for t in tables if t] if tables else []
                page_tables[page_num] = formatted_tables

        doc = fitz.open(pdf_path)

        full_document_text = "\n".join([doc[i].get_text() for i in range(len(doc))])
        normative_refs = extract_metadata(clean_text(full_document_text))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"---METADATA---\n")
            f.write(f"status: Active\n")
            f.write(f"normative_references: {normative_refs}\n")
            f.write(f"source: {os.path.basename(pdf_path)}\n")
            f.write(f"--------------\n\n")
            
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                raw_text = page.get_text()
                cleaned_text = clean_text(raw_text)
                
                f.write(f"\n{'#'*40}\n")
                f.write(f"## PAGE {page_num + 1} / {total_pages}\n")
                f.write(f"{'#'*40}\n\n")
                
                if cleaned_text:
                    f.write(cleaned_text + "\n\n")
                    
                tables = page_tables.get(page_num, [])
                if tables:
                    f.write("\n### EXTRACTED DATA TABLES ###\n")
                    for table_str in tables:
                        f.write(table_str)
                f.write("\n")
        
        doc.close()

        # Safe move: overwrite if archive already has a copy
        if os.path.exists(archive_path):
            os.remove(archive_path)
        shutil.move(pdf_path, archive_path)
        print(f"✅ Saved: {output_path} | Archived: {archive_path} | References: {normative_refs}")

    except Exception as e:
        print(f"❌ ERROR processing {os.path.basename(pdf_path)}: {e}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{INPUT_FOLDER}'. Nothing to process.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process.\n")
    for pdf_file in pdf_files:
        output_file = os.path.join(OUTPUT_FOLDER, pdf_file.stem + ".md")
        archive_file = os.path.join(ARCHIVE_FOLDER, pdf_file.name)
        process_pdf(str(pdf_file), output_file, archive_file)
    
    print("\n🎉 All PDFs processed. Converted files are in 'data' and source files are safely archived.")

if __name__ == "__main__":
    main()