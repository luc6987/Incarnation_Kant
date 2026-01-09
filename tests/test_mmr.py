import sys
import os
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock settings before importing retriever to avoid config loading issues if any
with patch("src.config.settings") as mock_settings:
    mock_settings.database.persist_directory = "./data/chromadb"
    mock_settings.database.collection_name = "test_coll"
    mock_settings.database.embedding_model = "test_model"
    mock_settings.database.top_k = 3
    mock_settings.database.search_type = "similarity"
    mock_settings.database.mmr_lambda = 0.5
    mock_settings.database.fetch_k = 10

    from src.database.retriever import KantRetriever

def test_mmr_logic():
    print("🚀 Testing MMR Logic with Mocks...")
    
    # Mock dependencies
    with patch("src.database.retriever.chromadb") as mock_chroma, \
         patch("src.database.retriever.HuggingFaceEmbeddings") as mock_hf:
        
        # Setup Retriever
        retriever = KantRetriever()
        
        # Mock embedding function
        # Dimension = 4 for simple testing
        retriever.embedding_fn = MagicMock()
        retriever.embedding_fn.embed_query.return_value = [1.0, 0.0, 0.0, 0.0] # Query is aligned with dim 0
        
        # Mock Collection
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Create fake data
        # Doc 0: Aligned with query (Sim=1.0)
        # Doc 1: Aligned with query (Sim=1.0) - Redundant with Doc 0
        # Doc 2: Orthogonal to query (Sim=0.0)
        # Doc 3: Partially aligned with query (Sim=0.5), Orthogonal to Doc 0
        
        # Embeddings
        embeddings = [
            [1.0, 0.0, 0.0, 0.0], # Doc 0
            [0.99, 0.01, 0.0, 0.0], # Doc 1 (very similar to 0)
            [0.0, 1.0, 0.0, 0.0], # Doc 2
            [0.5, 0.0, 0.866, 0.0], # Doc 3 (some relevance, different direction)
        ]
        
        documents = ["Doc 0 (Best)", "Doc 1 (Redundant)", "Doc 2 (Irrelevant)", "Doc 3 (Diverse)"]
        metadatas = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
        distances = [0.0, 0.01, 1.0, 0.5] # Approx 1-Sim
        
        # Setup return for query
        # Format: {"documents": [[...]], ...}
        mock_collection.query.return_value = {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
            "embeddings": [embeddings]
        }
        
        # 1. Test _mmr calculation directly
        print("\n--- Testing _mmr function ---")
        query_emb = [1.0, 0.0, 0.0, 0.0]
        # k=2, lambda=0.5
        # Expected: Doc 0 first. 
        # Then Doc 1 is penalized (sim to Doc 0 is high).
        # Doc 3 is moderate relevance, low sim to Doc 0.
        selected = retriever._mmr(query_emb, embeddings, k=2, lambda_mult=0.5)
        print(f"Selected indices (k=2, lambda=0.5): {selected}")
        
        # Doc 0 should be first (index 0)
        assert selected[0] == 0, "First selection should be Doc 0 (most relevant)"
        # Second selection should likely be Doc 3 (more diverse than Doc 1)
        # Sim(1, 0) ~= 1. Sim(3, 0) = 0.5. 
        # MMR(1) = 0.5*1 - 0.5*1 = 0
        # MMR(3) = 0.5*0.5 - 0.5*0.5 (sim(3,0)=0.5) = 0
        # Wait, let's check calculations.
        # Sim(Doc3, Doc0) = 0.5*1 + 0 + 0 = 0.5.
        # MMR(3) = 0.5*0.5 - 0.5*0.5 = 0.
        # MMR(1) = 0.5*0.99 - 0.5*0.99 = 0.
        
        # Let's try lambda = 0.8 (favor relevance)
        selected_rel = retriever._mmr(query_emb, embeddings, k=2, lambda_mult=0.8)
        print(f"Selected indices (k=2, lambda=0.8): {selected_rel}")
        # Expect Doc 0, then Doc 1
        
        # Let's try lambda = 0.2 (favor diversity)
        selected_div = retriever._mmr(query_emb, embeddings, k=2, lambda_mult=0.2)
        print(f"Selected indices (k=2, lambda=0.2): {selected_div}")
        # Expect Doc 0, then Doc 3 (since Doc 1 is too similar to Doc 0)
        
        # 2. Test retrieve method integration
        print("\n--- Testing retrieve() with search_type='mmr' ---")
        results = retriever.retrieve("test query", top_k=2, search_type="mmr", mmr_lambda=0.5)
        
        print(f"Retrieve returned {len(results)} results")
        for res in results:
            print(f" - {res['content']}")
            
        # Verify collection.query was called with embeddings
        call_args = mock_collection.query.call_args
        assert "embeddings" in call_args.kwargs["include"], "Should request embeddings from Chroma"
        assert len(results) == 2, "Should return top_k results"

        print("\n✅ Mock Test Passed.")

if __name__ == "__main__":
    test_mmr_logic()
