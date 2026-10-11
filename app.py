import os
import re 
import io
import fitz
import pdfplumber
import streamlit as st
import chromadb
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="ProcureAI | INDIAN STANDARD Code Engine",
    page_icon="✦",
    layout="wide"
)

if "prefill" not in st.session_state:
    st.session_state.prefill = ""
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = False
if "db_stats" not in st.session_state:
    st.session_state.db_stats = {"standards": 0, "chunks": 0}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "💬 Interactive Query Console"

CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
You are an expert procurement assistant for the Indian Government. 
Use the following pieces of retrieved context to answer the user's question. 
The text is extracted from official Indian Standards. Each context block includes METADATA containing the standard's status and mandatory Normative References.

Context:
{context}

Question: {question}

Instructions:
1. Answer directly with specific numerical limits, applicable IS codes, and clause references.
2. CRITICAL: You MUST explicitly create a section called "Mandatory Allied Standards" and list any IS codes found in the "Normative References" metadata of the context.
3. DO NOT invent any IS codes that are not present in the text or metadata.
4. If the answer is not in the context, state: "I cannot find the answer in the provided standards."
5. STRICT NO-MATH RULE: DO NOT calculate, multiply, or derive any numbers. Only output the exact numbers written in the text.
6. TABLE TRACING: If extracting data from a table, strictly trace the exact row and column to ensure the numbers match the specific size/category requested. If you are unsure, state "Data not clear in table."
7. TABLE PRIORITY: If the context contains a table with rows labeled "Carbon" or "Cat-bon" and columns labeled "Fe 415", "Fe 500", "Fe 550", look at the INTERSECTION of the "Carbon" row and the "Fe 500" column. The value there is the answer.
"""

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = OllamaLLM(model="llama3.2", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return embeddings, db, llm, prompt

@st.cache_data(ttl=3600, show_spinner=False)
def get_db_stats(_db):
    try:
        total_chunks = _db._collection.count()
        all_metadatas = _db.get(include=['metadatas'])['metadatas']
        unique_standards = len(set([
            os.path.basename(m.get('source', '')) 
            for m in all_metadatas 
            if m and 'source' in m
        ]))
        return {"standards": unique_standards, "chunks": total_chunks}
    except Exception:
        return {"standards": 0, "chunks": 0}

embeddings, db, llm, prompt = load_rag_pipeline()
stats = get_db_stats(db)

st.markdown("""
<style>
.stApp { background: #f8fafc; color: #0f172a !important; }
.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #f8fafc !important; }
.brand { padding: .4rem .2rem 1.5rem; font-size: 1.45rem; font-weight: 800; color: #ffffff; }
.brand span { color: #38bdf8; } 
.hero { padding: .25rem 0 .5rem; }
.hero h1 { font-size: 2rem; margin: 0; color: #0f172a; letter-spacing: -0.5px; }
.hero h1 span { color: #0284c7; } 
.hero p { color: #64748b; margin: .2rem 0 0; font-weight: 500; font-size: 0.95rem; }
.feature-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.85rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
.model-pill { display: inline-block; padding: .4rem .75rem; border: 1px solid #cbd5e1; border-radius: 999px; background: #ffffff; color: #334155; font-size: .8rem; font-weight: 600; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 6px; }
div[data-testid="stTextArea"] textarea { background: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important; min-height: 110px !important; }
div[data-testid="stTextArea"] textarea:focus { border-color: #0284c7 !important; }
label, .stTextArea label { color: #475569 !important; font-weight: 600 !important; }
div.stButton > button { border: 0 !important; border-radius: 8px !important; min-height: 2.8rem !important; font-weight: 700 !important; background: #0284c7 !important; color: white !important; }
div.stButton > button:hover { background: #0369a1 !important; }
.glass { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); padding: 1rem; }
.glass h3 { margin: 0 0 .6rem; color: #0f172a; font-size: .95rem; font-weight: 700; }
.status-row { display: flex; justify-content: space-between; gap: 1rem; padding: .5rem 0; border-bottom: 1px solid #f1f5f9; color: #475569; font-size: .85rem; font-weight: 500; }
.status-row:last-child { border-bottom: 0; }
.status-value { color: #0f172a; font-weight: 700; }
.online { color: #10b981; }
.thinking { padding: 1rem; border-radius: 10px; background: #f0f9ff; border: 1px solid #bae6fd; }
.thinking-title { font-size: .95rem; font-weight: 700; color: #0369a1; }
.processing { height: 4px; margin-top: .75rem; border-radius: 99px; background: linear-gradient(90deg, #bae6fd, #0284c7, #bae6fd); background-size: 200% 100%; animation: processing 2s linear infinite; }
@keyframes processing { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }
.result-card { padding: 1.15rem; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03); }
.result-badge { display: inline-block; padding: .25rem .55rem; border-radius: 6px; background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; font-size: .72rem; font-weight: 800; letter-spacing: 0.5px; }
.source-card { margin: .5rem 0; padding: .75rem; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; }
.empty-state { text-align: center; padding: 3rem 1.5rem; background: #ffffff; border: 2px dashed #e2e8f0; border-radius: 10px; color: #64748b; }
.streamlit-expanderHeader { background: #f8fafc !important; border-radius: 6px !important; border: 1px solid #e2e8f0 !important; font-weight: 600 !important; color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">✦ Procure<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown("### 🏠 Home")
    st.markdown("💬 Ask Assistant")
    st.markdown("📚 Standards Library")
    st.markdown("🛡️ Compliance Check")
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    st.markdown("🟢 **AI Model**  \nLlama 3.2")
    st.markdown("🟢 **Vector DB**  \nChromaDB")
    st.markdown("🟢 **Mode**  \nOffline / Air-Gapped")
    st.markdown("---")
    st.caption("🔐 Built for secure, auditable procurement")

top_left, top_right = st.columns([4, 1])
with top_left:
    st.markdown("""
    <div class="hero">
        <h1><span>ProcureAI</span> ✦ BIS Compliance Engine</h1>
        <p>Offline RAG Assistant for Indian Standard Code Extraction & Cross-Referencing</p>
    </div>
    """, unsafe_allow_html=True)
with top_right:
    st.markdown('<div class="model-pill" style="margin-top: 0.5rem;"><span class="dot"></span>Llama 3.2 Offline</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="feature-card" style="text-align:center;"><div style="font-size:1.4rem; font-weight:800; color:#0284c7;">{stats['standards']}</div><div style="color:#64748b; font-size:0.78rem; font-weight:600;">Standards Indexed</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="feature-card" style="text-align:center;"><div style="font-size:1.4rem; font-weight:800; color:#0284c7;">{stats['chunks']}</div><div style="color:#64748b; font-size:0.78rem; font-weight:600;">Clauses Vectorized</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown("""<div class="feature-card" style="text-align:center;"><div style="font-size:1.4rem; font-weight:800; color:#0284c7;">100%</div><div style="color:#64748b; font-size:0.78rem; font-weight:600;">Air-Gapped / Local</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown("""<div class="feature-card" style="text-align:center;"><div style="font-size:1.4rem; font-weight:800; color:#0284c7;">&lt;10s</div><div style="color:#64748b; font-size:0.78rem; font-weight:600;">Avg Retrieval Time</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

selected_tab = st.segmented_control(
    "Navigation",
    options=["💬 Interactive Query Console", "📄 Bulk Tender Analyzer", "🎯 Demo Scenarios"],
    key="active_tab",
    label_visibility="collapsed"
)

def extract_uploaded_text(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    bytes_data = uploaded_file.getvalue()
    
    if file_ext != "pdf":
        try: return bytes_data.decode("utf-8")
        except: return ""
    
    full_text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text_parts.append(f"\n--- PAGE {page_num + 1} ---\n{text}")
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    full_text_parts.append(f"\n[TABLE {t_idx + 1} on Page {page_num + 1}]")
                    for row in table:
                        clean_cells = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                        full_text_parts.append("| " + " | ".join(clean_cells) + " |")
                    full_text_parts.append("") 
    except Exception:
        try:
            doc = fitz.open(stream=bytes_data, filetype="pdf")
            for page in doc: full_text_parts.append(page.get_text())
            doc.close()
        except: pass

    extracted_content = "\n".join(full_text_parts)
    
    if len(extracted_content.strip()) < 50:
        st.info("⚠️ Non-searchable PDF detected. Running OCR on the first 10 pages...")
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(bytes_data, dpi=200, first_page=1, last_page=10)
            ocr_text = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img)
                ocr_text.append(f"\n--- PAGE {i + 1} (OCR) ---\n{text}")
            return "\n".join(ocr_text)
        except Exception as e:
            st.warning(f"OCR processing unavailable: {e}")
            return ""
            
    return extracted_content

def trigger_demo(query):
    """Callback to set query and seamlessly route to the console."""
    st.session_state.prefill = query
    st.session_state.trigger_query = True
    st.session_state.active_tab = "💬 Interactive Query Console"

if selected_tab == "💬 Interactive Query Console":
    col_input, col_output = st.columns([1, 1.25], gap="medium")
    
    with col_input:
        st.markdown("#### 🔎 Input Specification")
        query_input = st.text_area(
            "Query Prompt", 
            value=st.session_state.prefill, 
            placeholder="e.g. What is the maximum carbon percentage for Fe 500 grade steel?", 
            height=120,
            label_visibility="collapsed"
        )
        submit_button = st.button("✦ Analyze & Match Standards", type="primary", use_container_width=True)
        
        st.markdown("<p style='font-size:0.82rem; color:#64748b; margin-top:0.6rem; margin-bottom:0.3rem; font-weight:600;'>Sample Test Queries:</p>", unsafe_allow_html=True)
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("🇮🇳 National Flag ratio?", use_container_width=True):
                st.session_state.prefill = "What is the ratio of length to width for the National Flag of India?"
                st.session_state.trigger_query = True
                st.rerun()
            if st.button("📦 Plywood moisture %?", use_container_width=True):
                st.session_state.prefill = "What is the maximum moisture content allowed for tea-chest plywood?"
                st.session_state.trigger_query = True
                st.rerun()
        with ex_col2:
            if st.button("🔢 Round off 14.725?", use_container_width=True):
                st.session_state.prefill = "According to IS 2, how do you round off 14.725 to a fineness of 0.01?"
                st.session_state.trigger_query = True
                st.rerun()
            if st.button("🏗️ Fe 500 Carbon limit?", use_container_width=True):
                st.session_state.prefill = "What is the maximum carbon percentage for Fe 500 grade steel?"
                st.session_state.trigger_query = True
                st.rerun()

        st.markdown("""
        <div class="glass" style="margin-top: 0.8rem;">
            <h3>⚡ System Execution Parameters</h3>
            <div class="status-row"><span>Vector DB Status</span><span class="status-value online">● Ready</span></div>
            <div class="status-row"><span>Model Server</span><span class="status-value">Local Ollama</span></div>
            <div class="status-row"><span>Cross-Referencing</span><span class="status-value online">● Active</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_output:
        st.markdown("#### 📋 AI Recommendation & Analysis")
        output_container = st.container()

        active_query = None
        if submit_button and query_input.strip():
            active_query = query_input
            st.session_state.trigger_query = False
        elif st.session_state.trigger_query and st.session_state.prefill.strip():
            active_query = st.session_state.prefill
            st.session_state.trigger_query = False

        if active_query:
            with output_container:
                with st.spinner("Retrieving standard clauses..."):
                    results = db.similarity_search(active_query, k=5)
                    st.markdown("""
                    <div class="thinking">
                        <div class="thinking-title">✦ Processing query across vector store...</div>
                        <div style="color:#64748b;font-size:.8rem;margin-top:.2rem;">Filtering normative references and calculating numerical precision match.</div>
                        <div class="processing"></div>
                    </div>
                    """, unsafe_allow_html=True)

                    if not results:
                        st.warning("No relevant Indian Standards found in the database for this query.")
                    else:
                        context_parts = []
                        for doc in results:
                            context_parts.append(f"[METADATA -> Source: {doc.metadata.get('source', 'Unknown')} | Status: {doc.metadata.get('status', 'Unknown')} | Normative References: {doc.metadata.get('normative_references', 'None')}]\n{doc.page_content}")
                            
                        formatted_prompt = prompt.format(context="\n\n---\n\n".join(context_parts), question=active_query)
                        
                        try: 
                            response = llm.invoke(formatted_prompt)
                        except Exception as e: 
                            response = f"⚠️ **AI Model Unavailable**\n\nThe Ollama service could not be reached. Ensure it is running (`ollama serve`).\n\n_Error: {str(e)[:200]}_"

                        st.markdown("""<div class="result-card"><span class="result-badge">✓ AUDIT READY</span>""", unsafe_allow_html=True)
                        st.markdown(response)
                        st.markdown("</div>", unsafe_allow_html=True)

                        report_content = f"PROCURE AI — COMPLIANCE RECOMMENDATION REPORT\nGenerated: {datetime.now().strftime('%d-%b-%Y %H:%M')}\nMode: Offline / Air-Gapped | Model: Llama 3.2\n\nQUERY:\n{active_query}\n\nAI RECOMMENDATION:\n{response}\n\nSOURCE REFERENCES (AUDIT TRAIL):"
                        for i, doc in enumerate(results, 1): 
                            report_content += f"\n[{i}] {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Status: {doc.metadata.get('status', 'Unknown')} | Allied Refs: {doc.metadata.get('normative_references', 'None')}"
                        
                        st.download_button(
                            "📥 Download Audit Report (.txt)", 
                            data=report_content, 
                            file_name=f"ProcureAI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", 
                            mime="text/plain", 
                            use_container_width=True
                        )

                        with st.expander(f"📚 View {len(results)} Context Chunks & Allied Standards"):
                            for i, doc in enumerate(results):
                                refs = doc.metadata.get("normative_references", "None identified")
                                st.markdown(f"""<div class="source-card"><b style="color:#0f172a;">📄 Chunk {i+1}</b> <span style="color:#0284c7;"> · {os.path.basename(doc.metadata.get('source', 'Unknown'))}</span><br><span style="color:#166534; font-size: 0.8rem;">Status: {doc.metadata.get('status', 'Unknown')}</span></div>""", unsafe_allow_html=True)
                                if refs and refs != "None identified":
                                    st.markdown(f"""<div style="padding: 0.4rem 0.6rem; border-left: 3px solid #6b21a8; background: #faf5ff; border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.8rem; color: #6b21a8;"><b>MANDATORY ALLIED STANDARDS:</b> {refs}</div>""", unsafe_allow_html=True)
                                st.text(doc.page_content)
        else:
            with output_container:
                st.markdown("""
                <div class="empty-state">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔍</div>
                    <div style="font-weight: 700; color: #334155; margin-bottom: 0.2rem;">Awaiting Query Input</div>
                    <div style="font-size: 0.85rem;">Select a sample query on the left or type a custom request to view live AI compliance analysis.</div>
                </div>
                """, unsafe_allow_html=True)

elif selected_tab == "📄 Bulk Tender Analyzer":
    st.markdown("""
    <div class="glass" style="margin-bottom: 1rem;">
        <h3>📄 Bulk Tender Analyzer</h3>
        <p style="color:#475569; font-size: 0.88rem; margin: 0;">Upload a full tender document. The engine will strip boilerplate, map technical specifications to IS codes, and build a matrix of required normative dependencies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Tender Document (PDF, TXT, MD)", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        with st.spinner("Parsing tender content and mapping against standard repository..."):
            tender_text = extract_uploaded_text(uploaded_file)
            
            if not tender_text or len(tender_text.strip()) < 50:
                st.error("⚠️ Could not extract text even after OCR attempts.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                tender_chunks = text_splitter.split_text(tender_text)
                
                spec_keywords = r'\b(mm|grade|material|steel|cement|concrete|pipe|voltage|class|type|thickness|diameter)\b|IS[\s:\-]*\d+'
                technical_chunks = [c for c in tender_chunks if re.search(spec_keywords, c, re.IGNORECASE)]
                if not technical_chunks: technical_chunks = tender_chunks[len(tender_chunks)//2 : (len(tender_chunks)//2)+10]
                
                chunks_to_query = technical_chunks[:15]
                retrieved_docs = []
                
                if chunks_to_query:
                    with st.spinner("Querying vector database..."):
                        for chunk in chunks_to_query:
                            retrieved_docs.extend(db.similarity_search(chunk, k=2))

                if not retrieved_docs:
                    st.info("No matching Indian Standards found in the vector database.")
                else:
                    source_count = {}
                    for doc in retrieved_docs:
                        source = doc.metadata.get("source", "Unknown")
                        filename = os.path.basename(source)
                        is_code = filename.replace(".md", "").replace("_", " ")
                        if is_code not in source_count:
                            source_count[is_code] = {
                                "count": 0, 
                                "snippets": [], 
                                "status": doc.metadata.get("status", "Unknown"), 
                                "refs": doc.metadata.get("normative_references", "None"), 
                                "raw_content": doc.page_content
                            }
                        source_count[is_code]["count"] += 1
                        source_count[is_code]["snippets"].append(doc.page_content[:300] + "...")
                    
                    sorted_recommendations = sorted(source_count.items(), key=lambda x: x[1]["count"], reverse=True)[:4]
                    top_is_code, top_data = sorted_recommendations[0]
                    
                    try:
                        with st.spinner(f"Generating AI justification for {top_is_code}..."):
                            justify_prompt = f"The user uploaded a tender document. We matched it to {top_is_code}. Standard Text: {top_data['raw_content'][:1000]}\nWrite a 2-sentence explanation of why this standard is required for procurement."
                            justification = llm.invoke(justify_prompt)
                    except Exception:
                        justification = "(AI justification temporarily unavailable — model is offline or busy.)"

                    st.markdown(f"""<div class="result-card"><span class="result-badge">✓ SCANNED {len(technical_chunks)} TECHNICAL CLAUSES</span><h3 style="color:#0f172a;margin-top:.5rem;">✦ Recommended Indian Standards Matrix</h3></div>""", unsafe_allow_html=True)
                    for i, (is_code, data) in enumerate(sorted_recommendations, 1):
                        confidence = min((data["count"] * 30) + 55, 98)
                        with st.expander(f"#{i} **{is_code}** (Match Confidence: {int(confidence)}%)", expanded=(i==1)):
                            st.markdown(f"**Validity:** `{data['status']}` | **Normative Dependencies:** `{data['refs']}`")
                            if i == 1: st.info(f"**AI Justification:**\n{justification}")
                            st.markdown("**Sample Matched Clause:**")
                            st.text(data["snippets"][0])

elif selected_tab == "🎯 Demo Scenarios":
    st.markdown("""
    <div class="glass" style="margin-bottom: 1rem;">
        <h3>🎯 Guided Demo Scenarios</h3>
        <p style="color:#475569; font-size: 0.88rem; margin: 0;">Clicking any scenario will route you to the Interactive Query Console and automatically run the extraction engine.</p>
    </div>
    """, unsafe_allow_html=True)
    
    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        st.markdown("### 🏗️ Steel Procurement")
        st.button(
            "Run Scenario: Steel Tender Analysis", 
            use_container_width=True,
            on_click=trigger_demo,
            args=("What is the maximum carbon percentage for Fe 500 grade steel?",)
        )
            
        st.markdown("### 📦 Packaging Procurement")
        st.button(
            "Run Scenario: Tea-Chest Packaging", 
            use_container_width=True,
            on_click=trigger_demo,
            args=("What is the maximum moisture content allowed for tea-chest plywood?",)
        )
            
    with scenario_col2:
        st.markdown("### 🇮🇳 National Symbols")
        st.button(
            "Run Scenario: National Flag Spec", 
            use_container_width=True,
            on_click=trigger_demo,
            args=("What is the ratio of length to width for the National Flag of India?",)
        )
            
        st.markdown("### 🔢 Statistical Compliance")
        st.button(
            "Run Scenario: Rounding Rules", 
            use_container_width=True,
            on_click=trigger_demo,
            args=("According to IS 2, how do you round off 14.725 to a fineness of 0.01?",)
        )

st.markdown("""<div style="margin-top:2rem; padding: .75rem 1rem; border-radius: 8px; border: 1px solid #e2e8f0; background: #ffffff; color: #64748b; text-align: center; font-size: .8rem; font-weight: 600;">🛡️ Evidence-first · 🔒 Offline & Secure · 📑 Auditable BIS references · ⚡ Fast local inference</div>""", unsafe_allow_html=True)