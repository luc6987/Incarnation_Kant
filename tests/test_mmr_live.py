import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.database.retriever import KantRetriever

def test_mmr():
    print("🚀 Initializing Retriever...")
    try:
        retriever = KantRetriever()
    except Exception as e:
        print(f"Failed to init retriever: {e}")
        return

    query = "Freiheit und moralisches Gesetz" # Freedom and moral law
    
    print(f"\n🔎 Query: {query}")
    
    # 1. Similarity Search
    print("\n--- 1. Similarity Search (Top 3) ---")
    results_sim = retriever.retrieve(query, top_k=3, search_type="similarity")
    for i, res in enumerate(results_sim):
        print(f"[{i+1}] Score: {res['score']:.4f}")
        print(f"    Source: {res['metadata'].get('volume', '?')}:{res['metadata'].get('page', '?')}")
        print(f"    Content: {res['content'][:100]}...")

    # 2. MMR Search
    print("\n--- 2. MMR Search (Top 3, lambda=0.3) ---")
    # Using a lower lambda to force more diversity
    results_mmr = retriever.retrieve(query, top_k=3, search_type="mmr", mmr_lambda=0.3, fetch_k=20)
    for i, res in enumerate(results_mmr):
        print(f"[{i+1}] Score: {res['score']:.4f}") 
        print(f"    Source: {res['metadata'].get('volume', '?')}:{res['metadata'].get('page', '?')}")
        print(f"    Content: {res['content'][:100]}...")
        
    print("\n✅ Test Complete.")

if __name__ == "__main__":
    test_mmr()
