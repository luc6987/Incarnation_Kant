import streamlit as st
import os
import time
from typing import Any
import sys
from pathlib import Path

# --- Environment Setup (Must be before other imports) ---
# Set cache directories to project folder to avoid home dir quota issues
project_root = Path(__file__).parent.parent
venv_cache = project_root / ".venv" / ".cache"
venv_tmp = project_root / ".venv" / ".tmp"

# Create directories if they don't exist
venv_cache.mkdir(parents=True, exist_ok=True)
venv_tmp.mkdir(parents=True, exist_ok=True)

# Force environment variables
os.environ["HF_HOME"] = str(venv_cache / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(venv_cache / "huggingface")
os.environ["TMPDIR"] = str(venv_tmp)
os.environ["TMP"] = str(venv_tmp)
os.environ["TEMP"] = str(venv_tmp)
# -----------------------------------------------------

# --- ChromaDB SQLite Fix ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import settings
from src.database.retriever import KantRetriever
from src.hyde.generator import HydeGenerator
from src.rearticulation.engine import Rearticulator
from loguru import logger
# from langdetect import detect # Removed in favor of LLM detection
import time
import concurrent.futures

# Page Config
st.set_page_config(
    page_title="Digital Kant v2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        font-family: "Georgia", serif;
    }
    .german-quote {
        background-color: #f0f2f6;
        padding: 15px;
        border-left: 5px solid #4a4a4a;
        font-style: italic;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .source-tag {
        font-size: 0.8em;
        color: #666;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/43/Immanuel_Kant_%28painted_portrait%29.jpg", caption="Immanuel Kant (1724-1804)")
    st.title("Settings")
    
    # Try to get API key from env or settings
    default_api_key = os.environ.get("GOOGLE_API_KEY", settings.llm.api_key or "")
    api_key = st.text_input("Google API Key", type="password", value=default_api_key)
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        settings.llm.api_key = api_key
        # Reload settings if needed, or just rely on env var being set before Client init
    
    st.markdown("---")
    st.markdown("### Debug Info")
    if st.checkbox("Show Debug Logs"):
        st.session_state.show_debug = True
    else:
        st.session_state.show_debug = False

    if st.button("🔄 Reload System & Config"):
        st.cache_resource.clear()
        import importlib
        import src.config
        importlib.reload(src.config)
        st.rerun()


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# Load Resources (Cached)
@st.cache_resource
def load_system():
    try:
        retriever = KantRetriever()
        hyde = HydeGenerator()
        rearticulator = Rearticulator()
        return retriever, hyde, rearticulator
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None

# Main Interface
st.title("🏛️ Digital Kant")
st.caption("A Trans-temporal Digital Incarnation of Immanuel Kant")

# Check for API Key
if not os.environ.get("GOOGLE_API_KEY") and not settings.llm.api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to start.")
    st.stop()

# Load System
with st.spinner("Awakening the Philosopher... (Loading Models)"):
    retriever, hyde, rearticulator = load_system()
    if retriever and hyde and rearticulator:
        st.session_state.system_ready = True
    else:
        st.stop()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show Translated Sources for Assistant if available
        if msg["role"] == "assistant" and "details" in msg:
             with st.expander("📜 Referenced Texts (Translated)", expanded=False):
                for i, doc in enumerate(msg["details"]):
                    meta = doc.get('metadata', {})
                    work = meta.get('work_title', 'Unknown')
                    vol = meta.get('volume', '?')
                    page = meta.get('page', '?')
                    loc = f"AA {vol}, p. {page}"
                    # Use translated content if available, fallback to content
                    content = doc.get('translated_content', doc.get('content', ''))
                    url = doc.get('source_url', '')
                    
                    header = f"{i+1}. {work} ({loc})"
                    if url:
                        st.markdown(f"**[{header}]({url})**")
                    else:
                        st.markdown(f"**{header}**")
                    
                    st.markdown(content)
                    st.markdown("---")

        st.markdown(msg["content"], unsafe_allow_html=True)
        
        # Bottom expander for original german context (optional, kept for depth)
        if "details" in msg:
            with st.expander("Philosophical Context (Original German Source)"):
                for doc in msg["details"]:
                    meta = doc.get('metadata', {})
                    work_title = meta.get('work_title', 'Unknown Work')
                    volume = meta.get('volume', '??')
                    page = meta.get('page', '??')
                    korpora_url = doc.get('source_url', '')
                    original_context = doc.get('original_context', '')
                    
                    st.markdown(f"**{work_title}** (AA {volume}, p. {page})")
                    if korpora_url:
                        st.markdown(f"🔗 [Korpora Link]({korpora_url})")
                        
                    if original_context:
                        st.code(original_context, language="text")
                    else:
                        st.markdown(f"""
                        <div class="german-quote">
                            {doc['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()

# Input
if prompt := st.chat_input("Ask Immanuel Kant a question... (in any language)"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant Logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stats container
        stats_placeholder = st.empty()
        
        try:
            # Detect language
            # try:
            #     detected_lang = detect(prompt)
            # except:
            #     detected_lang = "unknown"
            
            # Use LLM for more accurate detection
            with st.spinner("Listening... (Detecting Language)"):
                detected_lang = rearticulator.detect_language(prompt)


            # 1. HyDE
            with st.status("Contemplating (HyDE Translation)...", expanded=True) as status:
                st.write("Translating intent to 18th Century German...")
                german_hypothesis = hyde.generate_hypothesis(prompt)
                
                with st.expander("View Generated Hypothesis", expanded=False):
                    st.markdown(german_hypothesis)

                # 2. Retrieval
                st.write(f"Searching Memory (fetching 50 candidates)...")
                # Get a larger pool (top_k=6) and filter dynamically
                raw_docs = retriever.retrieve(german_hypothesis, top_k=6, search_type="mmr", mmr_lambda=0.5, fetch_k=50)
                
                # Adaptive Filtering Logic
                docs = []
                if raw_docs:
                    best_score = raw_docs[0]['score']
                    for doc in raw_docs:
                        # Logic:
                        # 1. Keep if score is close to the best (e.g., within 20-30% drop)
                        # 2. Absolute quality floor (e.g., score > 0.3)
                        # 3. Always keep at least top 2 if they exist
                        relative_quality = doc['score'] / best_score if best_score > 0 else 0
                        
                        if (relative_quality > 0.75 and doc['score'] > 0.4) or len(docs) < 2:
                            docs.append(doc)
                        else:
                            break # Assume sorted by relevance (mostly true for MMR result list)

                st.write(f"Refined to {len(docs)} most relevant & diverse passages (Adaptive k).")
                
                # Translate docs in parallel
                st.write(f"Translating insights to {detected_lang}...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_doc = {
                        executor.submit(rearticulator.translate_text, doc['content'], detected_lang): doc 
                        for doc in docs
                    }
                    for future in concurrent.futures.as_completed(future_to_doc):
                        doc = future_to_doc[future]
                        try:
                            doc['translated_content'] = future.result()
                        except Exception as exc:
                            logger.error(f"Translation generated an exception: {exc}")
                            doc['translated_content'] = doc['content']

                # Show retrieved documents in status
                for i, doc in enumerate(docs):
                    meta = doc.get('metadata', {})
                    score = doc.get('score', 0)
                    
                    # Extract enriched fields
                    work_title = meta.get('work_title', 'Unknown Work')
                    volume = meta.get('volume', '??')
                    page = meta.get('page', '??')
                    line_range = meta.get('line_range', '??')
                    korpora_url = doc.get('source_url', '')
                    original_context = doc.get('original_context', '')
                    
                    with st.expander(f"Passage {i+1} (Score: {score:.2f}) - {work_title}"):
                        # Header Info
                        st.markdown(f"**Location:** AA {volume}, Page {page}, Lines {line_range}")
                        
                        if korpora_url:
                            st.markdown(f"🔗 [View on Korpora]({korpora_url})")
                        
                        # Display original vector content (German)
                        st.markdown("**Vector Content (Original German):**")
                        st.text(doc.get('content', ''))

                status.update(label="Reasoning Complete", state="complete", expanded=False)
            
            # Display initial stats before generation
            hyde_stats = hyde.client.usage_stats
            # We only know partial stats for now
            stats_text = f"🌐 Lang: {detected_lang} | 🤖 Model: {settings.llm.model_name} | 🔄 Requests: {hyde_stats['requests']} (HyDE)"
            stats_placeholder.caption(stats_text)
            
            # 3. Re-articulation (Streamed)
            
            # Display Translated Sources in Expander
            with st.expander("📜 Referenced Texts (Translated)", expanded=False):
                for i, doc in enumerate(docs):
                    meta = doc.get('metadata', {})
                    work = meta.get('work_title', 'Unknown')
                    vol = meta.get('volume', '?')
                    page = meta.get('page', '?')
                    loc = f"AA {vol}, p. {page}"
                    content = doc.get('translated_content', doc.get('content', ''))
                    url = doc.get('source_url', '')
                    
                    header = f"{i+1}. {work} ({loc})"
                    if url:
                        st.markdown(f"**[{header}]({url})**")
                    else:
                        st.markdown(f"**{header}**")
                        
                    st.markdown(content)
                    st.markdown("---")

            full_response = ""
            message_placeholder.markdown("Thinking...")
            
            stream_gen = rearticulator.rearticulate(prompt, docs, stream=True)
            
            for chunk in stream_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Update final stats
            # Combine stats from both clients
            reart_stats = rearticulator.client.usage_stats
            total_reqs = hyde_stats['requests'] + reart_stats['requests']
            # Note: Streaming usage update might be delayed or not captured in client yet if not handled in _stream_response
            # But let's show what we have.
            
            final_stats = (
                f"🌐 Lang: {detected_lang} | "
                f"🤖 Model: {settings.llm.model_name} | "
                f"📊 Tokens: {hyde_stats['total_tokens'] + reart_stats['total_tokens']} (approx) | "
                f"🔄 Requests: {total_reqs}"
            )
            stats_placeholder.caption(final_stats)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "details": docs
            })
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error in chat loop: {e}")

