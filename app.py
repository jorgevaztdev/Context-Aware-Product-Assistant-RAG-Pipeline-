import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Page Configuration
st.set_page_config(page_title="AI PM Assistant", layout="wide")
st.title("🤖 Context-Aware Product Assistant")
st.markdown("Upload a document (PRD, Notes) and ask questions about it.")

# Persistent state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# 2. Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    st.info("Using Ollama server for LLM responses")
    embedder_choice = st.radio(
        "Embeddings engine",
        ["Local (SentenceTransformers)", "Remote (Ollama: nomic-embed-text)"],
        index=0,
        help="Local embeddings are faster and avoid network overhead. Remote uses Ollama on the server."
    )
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

# 3. Main Application Logic
# Initialize the "Brain" (LLM) and "Embedder" (Translator)
ollama_base_url = "http://192.168.68.121:11434"

@st.cache_resource(show_spinner=False)
def get_embeddings(choice: str):
    if choice == "Local (SentenceTransformers)":
        return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return OllamaEmbeddings(base_url=ollama_base_url, model="nomic-embed-text")

embeddings = get_embeddings(embedder_choice)

# LLM via Ollama
llm = Ollama(base_url=ollama_base_url, model="qwen2.5:0.5b", temperature=0)

# File Uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])

if uploaded_file is not None:
    # Read the file based on its type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            # Handle PDF files
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if len(text.strip()) < 100:
                st.warning(f"⚠️ PDF appears to contain very little text ({len(text)} characters). It might be scanned or image-based.")
            else:
                st.info(f"📄 Extracted {len(text):,} characters from {len(pdf_reader.pages)} pages")
        elif file_type == 'docx':
            # Handle DOCX files
            doc = Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            # Handle text-based files (txt, md)
            bytes_data = uploaded_file.read()
            # Try UTF-8 first, then fall back to latin-1
            try:
                text = bytes_data.decode("utf-8")
            except UnicodeDecodeError:
                text = bytes_data.decode("latin-1")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()
    
    # 4. The "RAG" Process: Chunking
    # We split text because AI models have a limit on how much they can read at once.
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Even larger chunks = fewer embeddings
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.success(f"✅ Processed {len(chunks)} text chunks in {time.time() - start_time:.2f}s")

    # 5. Vector Store - Optimized with Parallel Embedding
    st.info(f"🔄 Creating embeddings for {len(chunks)} chunks...")
    if embedder_choice == "Local (SentenceTransformers)":
        st.caption("Using local model 'all-MiniLM-L6-v2'. First-time use may download weights and take longer.")
    
    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()
    time_text = st.empty()
    
    embed_start = time.time()
    
    # Batched embedding (works for both local SentenceTransformers and Ollama)
    all_embeddings = [None] * len(chunks)
    # SentenceTransformers performs well with batches of 32-128 on CPU
    batch_size = 64 if embedder_choice == "Local (SentenceTransformers)" else 50

    completed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        for j, emb in enumerate(batch_embeddings):
            all_embeddings[i + j] = emb

        completed += len(batch_embeddings)
        progress = completed / len(chunks)

        elapsed = time.time() - embed_start
        if completed > 0:
            est_total = (elapsed / completed) * len(chunks)
            est_remaining = max(est_total - elapsed, 0)
            time_text.text(f"⏱️ {elapsed:.1f}s elapsed | ~{est_remaining:.1f}s remaining | {completed/elapsed:.1f} chunks/sec")

        status_text.text(f"📦 Completed {completed}/{len(chunks)} chunks ({int(progress*100)}%)")
        progress_bar.progress(progress, text=f"{int(progress*100)}%")
    
    # Create FAISS index
    status_text.text("🔨 Building FAISS index...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(chunks, all_embeddings)),
        embedding=embeddings
    )
    # Persist for multi-question chat
    st.session_state["vector_store"] = vector_store
    
    embed_time = time.time() - embed_start
    progress_bar.progress(1.0, text="Complete!")
    status_text.empty()
    time_text.empty()
    
    st.success(f"✅ Document ready! {len(chunks)} embeddings in {embed_time:.2f}s ({len(chunks)/embed_time:.1f} chunks/sec)")
    st.success("💬 Ask your questions below:")

    # 6. Chat UI for multiple questions
    if st.session_state.get("vector_store") and st.session_state.get("qa_chain") is None:
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["vector_store"].as_retriever()
        )

    # Show history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the uploaded document")
    if user_input and st.session_state.get("qa_chain"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state["qa_chain"].invoke(user_input)
                answer = result.get("result", "No answer produced.")
                st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

elif st.session_state.get("vector_store"):
    # If a vector store exists from a previous upload, enable chat directly
    st.success("A document is loaded. You can ask multiple questions below.")
    if st.session_state.get("qa_chain") is None:
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["vector_store"].as_retriever()
        )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 

    user_input = st.chat_input("Ask a question about the loaded document")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state["qa_chain"].invoke(user_input)
                answer = result.get("result", "No answer produced.")
                st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
