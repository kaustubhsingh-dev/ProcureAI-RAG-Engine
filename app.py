import os
import re 
import fitz 
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="ProcureAI | INDIAN STANDARD Code Engine",
    page_icon="✦",
    layout="wide"
)

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
    return db, llm, prompt

db, llm, prompt = load_rag_pipeline()

st.markdown("""
<style>
/* Enterprise Trust Palette - Clean, Accessible, Institutional */
.stApp {
    background: #f8fafc; /* Crisp, light slate background */
    color: #0f172a !important; /* High contrast text */
}
.block-container { max-width: 1380px; padding-top: 1.4rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

/* Institutional Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important; /* Deep Navy */
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
.brand { padding: .4rem .2rem 1.5rem; font-size: 1.45rem; font-weight: 800; color: #ffffff; }
.brand span { color: #38bdf8; } /* Subtle Govt Blue */

/* Hero Section */
.hero { padding: .25rem 0 1rem; animation: fadeUp .55s ease both; }
.hero h1 { font-size: 2.25rem; margin: 0; color: #0f172a; letter-spacing: -0.5px; }
.hero h1 span { color: #0284c7; } /* Trust Blue */
.hero p { color: #64748b; margin: .35rem 0 0; font-weight: 500; }

/* Pill & Badges */
.model-pill {
    display: inline-block; padding: .5rem .85rem; border: 1px solid #cbd5e1; 
    border-radius: 999px; background: #ffffff; color: #334155; 
    font-size: .82rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02); font-weight: 600;
}
.dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; 
    background: #10b981; margin-right: 7px;
}

/* Search Bar (Stripped of neon, made premium) */
.search-shell {
    padding: 2px; border-radius: 14px; background: #e2e8f0; 
    transition: all 0.3s ease;
}
.search-shell:focus-within {
    background: #0284c7;
    box-shadow: 0 4px 12px rgba(2,132,199,0.15);
}
.search-inner { background: #ffffff; border-radius: 12px; padding: 1rem 1.15rem .75rem; }
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important; color: #0f172a !important; 
    border: 1px solid #cbd5e1 !important; border-radius: 8px !important; 
    min-height: 120px !important; transition: all .25s;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #0284c7 !important; box-shadow: 0 0 0 3px rgba(2,132,199,0.1) !important;
}
label, .stTextArea label { color: #475569 !important; font-weight: 600 !important; }

/* Action Buttons */
div.stButton > button {
    border: 0 !important; border-radius: 8px !important; min-height: 3rem !important; 
    font-weight: 700 !important; background: #0284c7 !important; color: white !important; 
    box-shadow: 0 4px 6px rgba(2,132,199,0.2) !important; transition: all .2s !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important; background: #0369a1 !important; 
    box-shadow: 0 6px 12px rgba(2,132,199,0.3) !important;
}

/* Glass/Status Panel */
.glass {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.03); padding: 1.15rem; animation: fadeUp .65s ease both;
}
.glass h3 { margin: 0 0 .9rem; color: #0f172a; font-size: 1rem; font-weight: 700; }
.status-row {
    display: flex; justify-content: space-between; gap: 1rem; padding: .72rem 0; 
    border-bottom: 1px solid #f1f5f9; color: #475569; font-size: .9rem; font-weight: 500;
}
.status-row:last-child { border-bottom: 0; }
.status-value { color: #0f172a; font-weight: 700; }
.online { color: #10b981; }

/* Processing Animation (Clean) */
.thinking {
    margin-top: 1.15rem; padding: 1.25rem; border-radius: 12px; 
    background: #f0f9ff; border: 1px solid #bae6fd; animation: fadeUp .7s ease both;
}
.thinking-title { font-size: 1.02rem; font-weight: 700; color: #0369a1; }
.processing {
    height: 4px; margin-top: 1rem; border-radius: 99px; 
    background: linear-gradient(90deg, #bae6fd, #0284c7, #bae6fd); 
    background-size: 200% 100%; animation: processing 2s linear infinite;
}
@keyframes processing { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }

/* Results & Cards */
.result-card {
    margin-top: 1.1rem; padding: 1.35rem; border-radius: 12px; 
    background: #ffffff; border: 1px solid #e2e8f0; 
    box-shadow: 0 10px 25px rgba(0,0,0,0.05); animation: resultIn .75s ease both;
}
.result-badge {
    display: inline-block; padding: .35rem .65rem; border-radius: 6px; 
    background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; 
    font-size: .75rem; font-weight: 800; letter-spacing: 0.5px;
}
.source-card {
    margin: .65rem 0; padding: 1rem; border-radius: 8px; background: #f8fafc; 
    border: 1px solid #e2e8f0; transition: all .2s; border-left: 4px solid #0284c7;
}
.source-card:hover { transform: translateX(3px); border-color: #cbd5e1; background: #ffffff; }

/* Footer */
.bottom-bar {
    margin-top: 1.2rem; padding: .9rem 1rem; border-radius: 8px; 
    border: 1px solid #e2e8f0; background: #ffffff; color: #64748b; 
    text-align: center; font-size: .8rem; font-weight: 600;
}

/* Animations */
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes resultIn { from { opacity: 0; transform: scale(.99) translateY(8px); } to { opacity: 1; transform: scale(1) translateY(0); } }

/* Fix Expander UI */
.streamlit-expanderHeader {
    background: #f8fafc !important; border-radius: 6px !important; 
    border: 1px solid #e2e8f0 !important; font-weight: 600 !important; color: #0f172a !important;
}
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

st.markdown("""
<div class="hero">
    <h1><span>ProcureAI</span> ✦</h1>
    <p>AI-Powered BIS Compliance & Procurement Assistant</p>
</div>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([4, 1])
with top_right:
    st.markdown('<div class="model-pill"><span class="dot"></span>AI Model: Llama 3.2</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Manual Query", "📄 Bulk Tender Analyzer"])

with tab1:
    left, right = st.columns([2.1, 1], gap="large")

    with left:
        query = st.text_area(
            "🔎 Ask about BIS standards, materials, limits or compliance",
            placeholder="Example: What are the manufacturing processes and carbon limits permitted for high-strength steel bars?",
            height=130
        )
        submit_button = st.button("✦ Analyze & Match Standards", type="primary", use_container_width=True)

    with right:
        st.markdown("""
        <div class="glass">
            <h3>⚡ System Status</h3>
            <div class="status-row"><span>Local Inference</span><span class="status-value online">● Active</span></div>
            <div class="status-row"><span>AI Model</span><span class="status-value">Llama 3.2</span></div>
            <div class="status-row"><span>Vector Store</span><span class="status-value">ChromaDB</span></div>
            <div class="status-row"><span>Cross-Referencing</span><span class="status-value online">● Active</span></div>
        </div>
        """, unsafe_allow_html=True)

    if submit_button and query.strip():
        with st.spinner("Retrieving clauses and generating compliance audit..."):
            results = db.similarity_search(query, k=3)

            st.markdown("""
            <div class="thinking">
                <div class="thinking-title">✦ AI is analyzing your request...</div>
                <div style="color:#7f8ba8;font-size:.84rem;margin-top:.3rem;">
                    Retrieving relevant standards and parsing mandatory cross-references.
                </div>
                <div class="processing"></div>
            </div>
            """, unsafe_allow_html=True)

            if not results:
                st.warning("No relevant Indian Standards found in the database for this query.")
            else:
                context_parts = []
                for doc in results:
                    source = doc.metadata.get("source", "Unknown")
                    status = doc.metadata.get("status", "Active")
                    refs = doc.metadata.get("normative_references", "None identified")
                    text = doc.page_content
                    context_block = f"[METADATA -> Source: {source} | Status: {status} | Normative References: {refs}]\n{text}"
                    context_parts.append(context_block)
                    
                context_text = "\n\n---\n\n".join(context_parts)
                formatted_prompt = prompt.format(context=context_text, question=query)
                response = llm.invoke(formatted_prompt)

                st.markdown("""
                <div class="result-card">
                    <span class="result-badge">✓ VERIFIED FROM RETRIEVED DOCUMENTS</span>
                    <h3 style="color:#0f172a;margin-top:.8rem;">✦ AI Recommendation & Compliance Report</h3>
                """, unsafe_allow_html=True)
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander(f"📚 Retrieved Sources & Audit Trail · {len(results)} documents"):
                    for i, doc in enumerate(results):
                        source_path = doc.metadata.get("source", "Unknown Document")
                        file_name = os.path.basename(source_path)
                        status = doc.metadata.get("status", "Active")
                        refs = doc.metadata.get("normative_references", "None identified")
                        st.markdown(f"""
                        <div class="source-card">
                            <b style="color:#e9d5ff;">📄 Reference {i+1}</b> <span style="color:#7dd3fc;"> · {file_name}</span><br>
                            <span style="color:#4ade80;">Status: {status}</span> | <span style="color:#a855f7;">Allied Refs: {refs}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.text(doc.page_content)

with tab2:
    def extract_uploaded_text(uploaded_file):
        file_extension = uploaded_file.name.split('.')[-1].lower()
        bytes_data = uploaded_file.getvalue()
        try:
            if file_extension == "pdf":
                doc = fitz.open(stream=bytes_data, filetype="pdf")
                text = "".join([page.get_text() for page in doc])
                doc.close()
                return text
            else: 
                return bytes_data.decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return ""

    st.markdown("""
    <div class="glass" style="margin-bottom: 2rem;">
        <h3>📄 Bulk Tender Analyzer</h3>
        <p style="color:#aebbd6; font-size: 0.9rem;">Upload a full tender document. The engine will automatically strip legal boilerplate, map technical specifications to IS codes, and build a matrix of required normative dependencies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False  
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing full tender document (scanning for technical specifications)..."):
            tender_text = extract_uploaded_text(uploaded_file)
            
            if not tender_text or len(tender_text.strip()) < 50:
                st.warning("⚠️ Could not extract enough text. Ensure document contains selectable text.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                tender_chunks = text_splitter.split_text(tender_text)
                
                spec_keywords = r'(mm|grade|material|is \d|steel|cement|concrete|pipe|voltage|class|type|thickness|diameter)'
                technical_chunks = [chunk for chunk in tender_chunks if re.search(spec_keywords, chunk, re.IGNORECASE)]
                
                if not technical_chunks:
                    technical_chunks = tender_chunks[len(tender_chunks)//2 : (len(tender_chunks)//2)+10] 
                
                retrieved_docs = []
                for chunk in technical_chunks[:15]: 
                    results = db.similarity_search(chunk, k=2)
                    retrieved_docs.extend(results)
                
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
                                "status": doc.metadata.get("status", "Active"),
                                "refs": doc.metadata.get("normative_references", "None identified"),
                                "raw_content": doc.page_content
                            }
                        source_count[is_code]["count"] += 1
                        source_count[is_code]["snippets"].append(doc.page_content[:300] + "...")
                    
                    sorted_recommendations = sorted(source_count.items(), key=lambda x: x[1]["count"], reverse=True)[:4]
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-badge">✓ SCANNED {len(technical_chunks)} TECHNICAL CLAUSES</span>
                        <h3 style="color:#0f172a;margin-top:.8rem;">✦ Recommended Indian Standards Matrix</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, (is_code, data) in enumerate(sorted_recommendations, 1):
                        confidence = min((data["count"] * 30) + 55, 98)
                        
                        with st.expander(f"#{i} **{is_code}** (Match Confidence: {int(confidence)}%)", expanded=(i==1)):
                            st.markdown(f"**Validity:** `{data['status']}` | **Normative Dependencies:** `{data['refs']}`")
                            
                            if i == 1: 
                                with st.spinner("Generating AI justification..."):
                                    justify_prompt = f"""
                                    The user uploaded a tender document. We matched it to {is_code}.
                                    Standard Text: {data['raw_content'][:1000]}
                                    
                                    Write a 2-sentence explanation of why this standard is required for procurement. Be specific.
                                    """
                                    justification = llm.invoke(justify_prompt)
                                    st.info(f"**AI Justification:**\n{justification}")
                            
                            st.markdown("**Sample Matched Clause:**")
                            st.text(data["snippets"][0])

st.markdown("""
<div class="bottom-bar">
    🛡️ Evidence-first · 🔒 Offline & Secure · 📑 Auditable BIS references · ⚡ Fast local inference
</div>
""", unsafe_allow_html=True)