"""
app.py — Streamlit RAG Chatbot UI (Google Gemini — FREE!)
Upload PDF/DOCX/TXT files and chat with your documents.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_core import load_documents, chunk_documents, create_vectorstore, build_rag_chain

# ─────────────────────────────────────────────────────────────
# Load .env (if present)
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f0f4ff;
        border-left: 4px solid #4285F4;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #333;
        margin-top: 4px;
    }
    .stChatMessage { border-radius: 10px; }
    .gemini-badge {
        background: linear-gradient(90deg, #4285F4, #34A853);
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown('<span class="gemini-badge">✨ Powered by Gemini — FREE</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Gemini API Key
    st.markdown("### 🔑 Google Gemini API Key")
    st.markdown(
        "Get your **free** key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)",
        unsafe_allow_html=True,
    )
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        placeholder="AIza...",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # File Upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

    st.markdown("---")

    # Process Button
    process_btn = st.button(
        "🚀 Process Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_btn:
        if not api_key_input:
            st.error("❌ Please enter your Gemini API key first.")
        else:
            with st.spinner("Loading and chunking documents..."):
                docs = load_documents(uploaded_files)

            if not docs:
                st.error("❌ Could not load any documents. Check file formats.")
            else:
                with st.spinner(f"Splitting {len(docs)} page(s) into chunks..."):
                    chunks = chunk_documents(docs)

                with st.spinner(f"Embedding {len(chunks)} chunks with Gemini..."):
                    try:
                        vectorstore = create_vectorstore(chunks, api_key_input)
                        st.session_state.rag_chain = build_rag_chain(vectorstore, api_key_input)
                        st.session_state.docs_processed = True
                        st.session_state.messages = []
                        st.success(f"✅ Ready! {len(chunks)} chunks indexed.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in ["messages", "rag_chain", "docs_processed"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Built with LangChain + Gemini 2.5 Flash + ChromaDB + Streamlit</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload your documents and ask anything in <b>English, தமிழ், or Tanglish</b> — powered by <b>Google Gemini (Free)</b> 🆓</p>',
    unsafe_allow_html=True,
)

if not st.session_state.docs_processed:
    st.info("👈 Upload documents from the sidebar and click **Process Documents** to get started.")
else:
    st.success("✅ Documents are ready. Start chatting below in English, தமிழ், or Tanglish!")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Chat Display
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    filename = src.metadata.get("source_filename", "Unknown")
                    page = src.metadata.get("page", "")
                    page_info = f" — Page {page + 1}" if page != "" else ""
                    snippet = src.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><b>[{i}] {filename}{page_info}</b><br>{snippet}…</div>',
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question in English / தமிழ் / Tanglish (e.g., 'ithula ena iruku', 'hy', 'explain this')...",
    disabled=not st.session_state.docs_processed,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                result = st.session_state.rag_chain({"question": prompt})
                answer = result.get("answer", "Sorry, I couldn't find an answer.")
                sources = result.get("source_documents", [])

                st.markdown(answer)

                # Deduplicate sources
                seen = set()
                unique_sources = []
                for src in sources:
                    key = src.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(src)

                if unique_sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, src in enumerate(unique_sources, 1):
                            filename = src.metadata.get("source_filename", "Unknown")
                            page = src.metadata.get("page", "")
                            page_info = f" — Page {page + 1}" if page != "" else ""
                            snippet = src.page_content[:300].replace("\n", " ")
                            st.markdown(
                                f'<div class="source-box"><b>[{i}] {filename}{page_info}</b><br>{snippet}…</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": unique_sources,
                })

            except Exception as e:
                err_msg = f"❌ Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": [],
                })
