import os
import re
import fitz   
import pdfplumber
from pathlib import Path

INPUT_FOLDER = "raw_pdfs"    
OUTPUT_FOLDER = "data"       

def clean_text(text):
    """Strips watermarks, fixes OCR line breaks, and corrects common table header typos."""
    watermark_pattern = r"Free Standard provided by BIS via BSB Edge Private Limited.*?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\."
    text = re.sub(watermark_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    reaffirmed_pattern = r"\(Reaffirmed\s*\d{4}\)"
    text = re.sub(reaffirmed_pattern, "", text, flags=re.IGNORECASE)
    
    text = re.sub(r'Cat-bon', 'Carbon', text, flags=re.IGNORECASE)
    text = re.sub(r'Cat-bot', 'Carbon', text, flags=re.IGNORECASE)
    text = re.sub(r'Catbon', 'Carbon', text, flags=re.IGNORECASE)
    text = re.sub(r'Sulphur', 'Sulphur', text, flags=re.IGNORECASE)  
    text = re.sub(r'Phosphorus', 'Phosphorus', text, flags=re.IGNORECASE)
    text = re.sub(r'fimm\+', 'following', text, flags=re.IGNORECASE)
    text = re.sub(r'\~e4\~5', 'Fe 415', text, flags=re.IGNORECASE)
    text = re.sub(r'\~e yJj', 'Fe 500', text, flags=re.IGNORECASE)
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
    """STRESS-TEST FIX: Hunts for Clause 2 and extracts mandatory IS Codes"""
    refs = "None identified"
    match = re.search(r'2\.?\s*Normative References(.*?)(?:3\.?\s*Terms and Definitions|4\.?\s*General)', full_text, flags=re.IGNORECASE | re.DOTALL)
    
    if match:
        block = match.group(1)
        is_codes = re.findall(r'IS\s*\d+(?:\s*\(\s*Part\s*\d+\s*\))?', block, flags=re.IGNORECASE)
        if is_codes:
            is_codes = list(set([code.replace('\n', '').strip() for code in is_codes]))
            refs = ", ".join(is_codes)
            
    return refs

def process_pdf(pdf_path, output_path):
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
        os.remove(pdf_path)
        print(f"✅ Saved: {output_path} | Associated References: {normative_refs}")

    except Exception as e:
        print(f"❌ ERROR processing {os.path.basename(pdf_path)}: {e}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{INPUT_FOLDER}'. Nothing to process.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process.\n")
    for pdf_file in pdf_files:
        output_file = os.path.join(OUTPUT_FOLDER, pdf_file.stem + ".md")
        process_pdf(str(pdf_file), output_file)
    
    print("\n🎉 All PDFs processed. Converted files are in 'data'.")

if __name__ == "__main__":
    main()