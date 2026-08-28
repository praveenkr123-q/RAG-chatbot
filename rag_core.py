"""
rag_core.py — Multilingual RAG logic (Google Gemini — google-genai 2.x SDK)
Handles document loading, chunking, embedding, and retrieval.

Supported Languages: English, Tamil (தமிழ்), Tanglish (Tamil in English letters).
Embedding model: gemini-embedding-001
Generation model: gemini-2.5-flash / gemini-3.6-flash / gemini-flash-latest
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_core")


# ──────────────────────────────────────────────
# 1. Custom Embeddings — google-genai 2.x SDK
# ──────────────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings using the google-genai 2.x SDK.
    Model: gemini-embedding-001
    """

    MODEL = "gemini-embedding-001"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided to initialize GeminiEmbeddings.")
        self.api_key = api_key
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize google-genai Client: {e}")
            raise

    def _embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Embed a single string using gemini-embedding-001."""
        try:
            response = self.client.models.embed_content(
                model=self.MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise ValueError("Embedding API returned empty embedding values.")
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Gemini API Embedding error for model {self.MODEL}: {e}")
            raise RuntimeError(f"Gemini Embedding failed ({self.MODEL}): {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for indexing."""
        return [self._embed(text, task_type="RETRIEVAL_DOCUMENT") for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query string for retrieval."""
        return self._embed(text, task_type="RETRIEVAL_QUERY")


# ──────────────────────────────────────────────
# 2. Document Loading
# ──────────────────────────────────────────────

def load_documents(uploaded_files) -> List[Document]:
    """
    Load documents from Streamlit UploadedFile objects.
    Supports: PDF, DOCX, TXT
    """
    docs = []

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                continue

            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_filename"] = uploaded_file.name
            docs.extend(loaded)
        except Exception as e:
            logger.error(f"Error loading file {uploaded_file.name}: {e}")
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return docs


# ──────────────────────────────────────────────
# 3. Chunking
# ──────────────────────────────────────────────

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ──────────────────────────────────────────────
# 4. Vector Store (ChromaDB)
# ──────────────────────────────────────────────

def create_vectorstore(chunks: List[Document], api_key: str) -> Chroma:
    """Embed chunks using gemini-embedding-001 and store in ChromaDB."""
    embeddings = GeminiEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
    )
    return vectorstore


# ──────────────────────────────────────────────
# 5. Multilingual Gemini RAG Chain
# ──────────────────────────────────────────────

class GeminiRAGChain:
    """
    Multilingual Conversational RAG Chain powered directly by Google GenAI SDK.
    Supports English, Tamil (தமிழ்), and Tanglish.
    """

    MODELS_TO_TRY = [
        "gemini-2.5-flash",
        "gemini-3.6-flash",
        "gemini-flash-latest",
    ]

    def __init__(self, vectorstore: Chroma, api_key: str):
        self.vectorstore = vectorstore
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.chat_history: List[Dict[str, str]] = []

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute RAG query with retrieved context, conversation history,
        and multilingual capability (English, Tamil, Tanglish).
        """
        question = inputs.get("question", "").strip()
        if not question:
            return {"answer": "Please ask a question.", "source_documents": []}

        # 1. Retrieve relevant document chunks
        retrieved_docs: List[Document] = []
        if self.vectorstore:
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(question)
            except Exception as e:
                logger.warning(f"Retrieval warning: {e}")

        # 2. Build context string
        context_snippets = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source_filename", "Document")
            page = doc.metadata.get("page", None)
            page_info = f", Page {page + 1}" if page is not None else ""
            context_snippets.append(f"[Source {i}: {source}{page_info}]\n{doc.page_content}")

        context_text = "\n\n---\n\n".join(context_snippets) if context_snippets else "No specific document context found."

        # 3. Build System Instruction for Multilingual Support (Tamil / Tanglish / English)
        system_instruction = (
            "You are an intelligent, friendly AI assistant designed for document Q&A and conversation.\n"
            "You have access to the user's uploaded documents provided in the Document Context below.\n\n"
            "=== LANGUAGE & COMMUNICATION RULES ===\n"
            "1. You MUST understand and reply fluently in:\n"
            "   - Tanglish (Tamil words written in English letters, e.g., 'ithula ena iruku', 'solunga bro', 'enaku ithu pathi explain pannu', 'epdi iruka')\n"
            "   - Tamil (தமிழ் எழுத்துகளில், e.g., 'வணக்கம்', 'இந்த ஆவணத்தில் என்ன உள்ளது?')\n"
            "   - English\n"
            "2. ALWAYS match the language and style of the user:\n"
            "   - If user asks in Tanglish -> Reply in natural, friendly, easy-to-understand Tanglish!\n"
            "   - If user asks in Tamil -> Reply in clear Tamil (தமிழ்)!\n"
            "   - If user asks in English -> Reply in clear English!\n"
            "3. FOR GREETINGS & CASUAL MESSAGES (e.g. 'hy', 'hi', 'hello', 'vanakkam', 'who are you', 'how are you', 'epdi iruka'):\n"
            "   - Greet back warmly and politely in their language!\n"
            "   - Tell them you are ready to answer any questions about their uploaded documents.\n"
            "4. FOR DOCUMENT QUESTIONS:\n"
            "   - Use the Document Context below to answer accurately, concisely, and helpfully.\n"
            "   - If the answer is not in the documents, politely inform them in their chosen language.\n\n"
            f"=== DOCUMENT CONTEXT ===\n"
            f"{context_text}"
        )

        # 4. Format conversation history
        history_prompts = []
        for turn in self.chat_history[-6:]:  # keep last 6 turns
            history_prompts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")

        history_text = "\n".join(history_prompts)
        if history_text:
            full_prompt = f"Previous Conversation:\n{history_text}\n\nCurrent User Question: {question}"
        else:
            full_prompt = f"User Question: {question}"

        # 5. Generate content with automatic model fallback
        last_error = None
        answer_text = ""

        for model in self.MODELS_TO_TRY:
            try:
                logger.info(f"Generating content using model: {model}")
                response = self.client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                )
                if response and response.text:
                    answer_text = response.text
                    break
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
                continue

        if not answer_text:
            if last_error:
                answer_text = f"Error generating response: {last_error}"
            else:
                answer_text = "Sorry, I could not generate a response. Please try again."

        # 6. Save to history
        self.chat_history.append({"user": question, "assistant": answer_text})

        return {
            "answer": answer_text,
            "source_documents": retrieved_docs,
        }


def build_rag_chain(vectorstore: Chroma, api_key: str) -> GeminiRAGChain:
    """
    Build a multilingual Gemini RAG Chain supporting English, Tamil, and Tanglish.
    """
    return GeminiRAGChain(vectorstore=vectorstore, api_key=api_key)
