import os
import sys

# --- ChromaDB SQLite Fix ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from src.config import settings, PROJECT_ROOT

class KantRetriever:
    """Retrieval engine for Digital Kant."""
    
    def __init__(self):
        self.persist_directory = settings.database.persist_directory
        self.collection_name = settings.database.collection_name
        self.model_name = settings.database.embedding_model
        
        # Ensure embedding model is loaded (uses same cache logic as ingest if possible)
        # We rely on standard HF_HOME env var which should be set in .env or system
        
        # Explicitly set HF_HOME to .venv cache if not set, matching where we downloaded the model
        if not os.environ.get("HF_HOME"):
            possible_cache = Path(PROJECT_ROOT) / ".venv" / ".cache" / "huggingface"
            if possible_cache.exists():
                os.environ["HF_HOME"] = str(possible_cache.absolute())
                logger.info(f"🔧 Set HF_HOME to: {os.environ['HF_HOME']}")
        
        self._init_embedding_model()
        self._init_db()
        
    def _init_embedding_model(self):
        """Initialize local embedding model."""
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or (hasattr(import_torch(), "cuda") and import_torch().cuda.is_available()) else "cpu"
        logger.info(f"📥 Loading embedding model {self.model_name} on {device}...")
        
        try:
            self.embedding_fn = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.success("✅ Embedding model loaded.")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def _init_db(self):
        """Connect to ChromaDB."""
        logger.info(f"🔌 Connecting to ChromaDB at {self.persist_directory}...")
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.success(f"✅ Connected to collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise

    def _mmr(self, query_embedding: List[float], doc_embeddings: List[List[float]], k: int, lambda_mult: float) -> List[int]:
        """Calculate Maximal Marginal Relevance (MMR).
        
        MMR = argmax [lambda * Sim(Di, Q) - (1-lambda) * max(Sim(Di, Dj))]
        
        Args:
            query_embedding: The embedding of the query (list of floats).
            doc_embeddings: List of embeddings for candidate documents.
            k: Number of documents to select.
            lambda_mult: Trade-off parameter (0.5 is balanced).
            
        Returns:
            List of indices of the selected documents in the doc_embeddings list.
        """
        # Convert to numpy arrays
        query_emb = np.array([query_embedding]) # (1, d)
        doc_embs = np.array(doc_embeddings) # (n, d)
        
        # Calculate cosine similarity between query and all documents
        # Ensure normalization for cosine similarity
        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        d_norm = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        
        # Avoid division by zero
        # Use np.maximum to safely avoid zero division while keeping array shape/type
        q_norm = np.maximum(q_norm, 1e-9)
        d_norm = np.maximum(d_norm, 1e-9)
        
        # Similarity with query: (1, d) @ (d, n) -> (1, n)
        sim_to_query = (query_emb @ doc_embs.T) / (q_norm @ d_norm.T)
        sim_to_query = sim_to_query.flatten() # (n,)
        
        # Initialize
        selected_indices = []
        candidate_indices = list(range(len(doc_embeddings)))
        
        # Select k documents
        for _ in range(min(k, len(candidate_indices))):
            best_mmr = -float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                # Relevance part
                relevance = sim_to_query[idx]
                
                # Diversity part (max similarity to already selected)
                if not selected_indices:
                    diversity = 0
                else:
                    # Get embeddings of selected docs: (m, d)
                    selected_embs_arr = doc_embs[selected_indices]
                    current_emb = doc_embs[idx].reshape(1, -1) # (1, d)
                    
                    # Sim with selected
                    s_norm = np.linalg.norm(selected_embs_arr, axis=1, keepdims=True)
                    c_norm = np.linalg.norm(current_emb, axis=1, keepdims=True)
                    
                    # Safety
                    s_norm = np.maximum(s_norm, 1e-9)
                    c_norm = np.maximum(c_norm, 1e-9)

                    sim_to_selected = (current_emb @ selected_embs_arr.T) / (c_norm @ s_norm.T)
                    diversity = np.max(sim_to_selected)
                
                # MMR Score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            # Add best to selected
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            
        return selected_indices

    def retrieve(self, query: str, top_k: int = None, search_type: str = None, fetch_k: int = None, mmr_lambda: float = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query string (usually 18th century German from HyDE).
            top_k: Number of documents to return.
            search_type: "similarity" or "mmr".
            fetch_k: Initial candidates to fetch for MMR.
            mmr_lambda: Diversity penalty for MMR (0.5 is balanced).
            
        Returns:
            List of dictionaries containing document content and metadata.
        """
        k = top_k or settings.database.top_k
        stype = search_type or settings.database.search_type
        fk = fetch_k or settings.database.fetch_k
        lam = mmr_lambda or settings.database.mmr_lambda
        
        if stype == "mmr":
            logger.info(f"🔍 Retrieving top {k} (MMR pool={fk}) for query: {query[:50]}...")
        else:
            logger.info(f"🔍 Retrieving top {k} (Similarity) for query: {query[:50]}...")
        
        try:
            # Embed query
            query_embedding = self.embedding_fn.embed_query(query)
            
            # Determine initial fetch count
            initial_k = fk if stype == "mmr" else k
            
            # Include embeddings if using MMR
            include_fields = ["documents", "metadatas", "distances"]
            if stype == "mmr":
                include_fields.append("embeddings")
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                include=include_fields
            )
            
            # Extract basic results
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Post-processing for MMR
            # Ensure results["embeddings"][0] is not None and handle array truthiness safely
            if stype == "mmr" and "embeddings" in results:
                # results["embeddings"] should be a list of list of floats or None
                # Check for None explicitly first
                if results["embeddings"] is None:
                    logger.warning("⚠️ MMR requested but no embeddings returned by ChromaDB.")
                # Check length
                elif len(results["embeddings"]) == 0:
                    logger.warning("⚠️ MMR requested but embeddings list is empty.")
                else:
                    doc_embeddings = results["embeddings"][0]
                    # Check if the specific query result has embeddings
                    if doc_embeddings is None:
                         logger.warning("⚠️ MMR requested but first query embedding result is None.")
                    elif len(doc_embeddings) == 0:
                         # This is valid (no docs found), but MMR can't run
                         pass 
                    else:
                        # Run MMR selection
                        try:
                            selected_indices = self._mmr(query_embedding, doc_embeddings, k, lam)
                            
                            # Reorder and slice based on MMR selection
                            documents = [documents[i] for i in selected_indices]
                            metadatas = [metadatas[i] for i in selected_indices]
                            distances = [distances[i] for i in selected_indices]
                            
                            logger.info(f"🧬 MMR reranked {initial_k} -> {len(documents)} docs (lambda={lam})")
                        except Exception as mmr_err:
                            logger.error(f"❌ MMR calculation failed: {mmr_err}")
                            # Fallback to standard ranking (already sorted by similarity)
                            documents = documents[:k]
                            metadatas = metadatas[:k]
                            distances = distances[:k]
            
            
            formatted_results = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Generate Korpora URL and find local TXT file
                volume = meta.get("volume", "")
                page = meta.get("page", "")
                
                korpora_url = ""
                local_txt_path = ""
                
                if volume and page:
                    try:
                        # Format page number to 3 digits (e.g., 7 -> "007")
                        page_num = int(page)
                        page_str = f"{page_num:03d}"
                        
                        # Construct Korpora URL
                        korpora_url = f"https://korpora.org/kant/{volume}/{page_str}.html"
                        
                        # Construct local TXT path
                        # Using relative path from project root
                        txt_path_obj = Path("data/txt/kant") / volume / f"{page_str}.txt"
                        if txt_path_obj.exists():
                            local_txt_path = str(txt_path_obj)
                            
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️  Invalid volume/page in metadata: {volume}/{page}")
                
                # Extract original context from TXT file if available
                original_context = ""
                if local_txt_path:
                    line_range = meta.get("line_range", "")
                    original_context = self._get_original_context(local_txt_path, line_range)

                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # Convert distance to similarity score approx
                    "source_url": korpora_url,
                    "local_txt_path": local_txt_path,
                    "original_context": original_context
                })
                
            logger.info(f"✅ Found {len(formatted_results)} documents.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            # If MMR fails, try to fallback or just return empty? 
            # Ideally we log the error and maybe return whatever we have if possible, 
            # but for now empty is safer to avoid confusion.
            return []

    def _get_original_context(self, txt_path: str, line_range: str) -> str:
        """Extract specific lines from the TXT file.
        
        Args:
            txt_path: Path to the .txt file.
            line_range: String indicating lines (e.g., "1-15" or "5").
            
        Returns:
            The extracted text lines joined by newlines.
        """
        if not txt_path or not line_range:
            return ""
            
        try:
            # Parse line range "start-end" or "start"
            if "-" in line_range:
                parts = line_range.split("-")
                start = int(parts[0])
                end = int(parts[1])
            else:
                start = int(line_range)
                end = start
                
            extracted_lines = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Line format: "01: Content..." or "1: Content..."
                    # Split only on first colon
                    parts = line.split(":", 1)
                    if len(parts) < 2: 
                        continue
                    
                    try:
                        line_num = int(parts[0])
                        if start <= line_num <= end:
                            # Keep the line number prefix for reference? 
                            # User asked for "corresponding paragraph", maybe raw text is better?
                            # Let's keep the full line "01: Content" as it gives context
                            extracted_lines.append(line)
                    except ValueError:
                        continue
                        
            return "\n".join(extracted_lines)
        except Exception as e:
            logger.warning(f"⚠️  Failed to read original context from {txt_path}: {e}")
            return ""

def import_torch():
    """Lazy import torch to avoid heavy load if not needed immediately."""
    import torch
    return torch
