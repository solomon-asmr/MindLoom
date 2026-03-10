import chromadb
from rank_bm25 import BM25Okapi
from config import CHROMA_DB_PATH
# Create ONE client that all functions share
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_user_collection(user_id):
    """Get or create a ChromaDB collection for a specific user."""
    collection_name = f"user_{user_id}"
    collection = chroma_client.get_or_create_collection(collection_name)
    return collection

def add_to_collection(user_id, chunks, source_name, source_type, category="general"):
    """Add text chunks to a user's collection with category metadata.
    
    Args:
        user_id: Telegram user ID
        chunks: list of text strings
        source_name: name of the source
        source_type: "document", "webpage", or "image"
        category: category tag for filtering
    
    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    collection = get_user_collection(user_id)

    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": source_name,
            "type": source_type,
            "category": category,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)

def search_collection(user_id, query, top_k=3, categories=None):
    """Hybrid search with optional category filtering.
    
    Args:
        user_id: Telegram user ID
        query: the user's question
        top_k: how many results to return
        categories: list of category keys to filter by, or None for all
    
    Returns:
        dict with "documents" and "sources" keys
    """
    collection = get_user_collection(user_id)

    if collection.count() == 0:
        return {"documents": [], "sources": []}

    # Build the where filter for categories
    where_filter = None
    if categories and "all" not in categories:
        if len(categories) == 1:
            where_filter = {"category": categories[0]}
        else:
            where_filter = {"category": {"$in": categories}}

    # ===== STEP 1: Vector search =====
    total_docs = collection.count()
    fetch_k = min(top_k * 3, total_docs)

    try:
        if where_filter:
            vector_results = collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=where_filter
            )
        else:
            vector_results = collection.query(
                query_texts=[query],
                n_results=fetch_k
            )
    except Exception:
        # If filtered search fails (no docs match filter), search all
        vector_results = collection.query(
            query_texts=[query],
            n_results=fetch_k
        )

    vector_docs = vector_results["documents"][0] if vector_results["documents"] else []
    vector_ids = vector_results["ids"][0] if vector_results["ids"] else []
    vector_metadatas = vector_results["metadatas"][0] if vector_results["metadatas"] else []

    if not vector_docs:
        return {"documents": [], "sources": []}

    # ===== STEP 2: BM25 keyword search =====
    if where_filter:
        try:
            all_data = collection.get(include=["documents", "metadatas"], where=where_filter)
        except Exception:
            all_data = collection.get(include=["documents", "metadatas"])
    else:
        all_data = collection.get(include=["documents", "metadatas"])

    all_docs = all_data["documents"]
    all_ids = all_data["ids"]
    all_metadatas = all_data["metadatas"]

    if not all_docs:
        # No docs match filter, return vector results only
        documents = vector_docs[:top_k]
        sources = []
        for meta in vector_metadatas[:top_k]:
            source = meta.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
        return {"documents": documents, "sources": sources}

    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:fetch_k]

    bm25_docs = [all_docs[i] for i in bm25_top_indices]
    bm25_ids = [all_ids[i] for i in bm25_top_indices]
    bm25_metadatas = [all_metadatas[i] for i in bm25_top_indices]

    # ===== STEP 3: Reciprocal Rank Fusion =====
    rrf_scores = {}
    doc_map = {}
    k = 60

    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)
        doc_map[doc_id] = {
            "document": vector_docs[rank],
            "metadata": vector_metadatas[rank]
        }

    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "document": bm25_docs[rank],
                "metadata": bm25_metadatas[rank]
            }

    # ===== STEP 4: Sort and return =====
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]

    documents = [doc_map[doc_id]["document"] for doc_id in top_ids]

    sources = []
    for doc_id in top_ids:
        source = doc_map[doc_id]["metadata"].get("source", "Unknown")
        if source not in sources:
            sources.append(source)

    return {
        "documents": documents,
        "sources": sources
    }

def get_user_sources(user_id):
    """Get a list of all sources a user has added.
    
    Returns:
        list of dicts: [{"name": "file.pdf", "type": "document", "chunks": 45}, ...]
    """
    collection = get_user_collection(user_id)

    if collection.count() == 0:
        return []

    # Get all metadata from the collection
    all_data = collection.get(include=["metadatas"])

    # Count chunks per source
    sources = {}
    for metadata in all_data["metadatas"]:
        source_name = metadata.get("source", "Unknown")
        source_type = metadata.get("type", "unknown")

        if source_name not in sources:
            sources[source_name] = {
                "name": source_name,
                "type": source_type,
                "chunks": 0
            }
        sources[source_name]["chunks"] += 1

    return list(sources.values())

def delete_source(user_id, source_name):
    """Delete all chunks from a specific source.
    
    Args:
        user_id: Telegram user ID
        source_name: name of the source to delete
    
    Returns:
        True if deleted, False if source not found
    """
    collection = get_user_collection(user_id)

    # Find all chunk IDs that belong to this source
    all_data = collection.get(include=["metadatas"])

    ids_to_delete = []
    for i, metadata in enumerate(all_data["metadatas"]):
        if metadata.get("source") == source_name:
            ids_to_delete.append(all_data["ids"][i])

    if not ids_to_delete:
        return False

    collection.delete(ids=ids_to_delete)
    return True

def clear_user_data(user_id):
    """Delete all data for a user.
    
    Returns:
        True if cleared successfully
    """
    collection_name = f"user_{user_id}"

    try:
        chroma_client.delete_collection(collection_name)
        return True
    except Exception:
        return False

def get_user_stats(user_id):
    """Get statistics about a user's knowledge base.
    
    Returns:
        dict with total_chunks and total_sources
    """
    sources = get_user_sources(user_id)

    total_chunks = sum(s["chunks"] for s in sources)

    return {
        "total_chunks": total_chunks,
        "total_sources": len(sources)
    }


if __name__ == "__main__":
    # --- INTERNAL TEST SUITE ---
    # This code only runs when you execute this file directly.
    
    test_user = 12345
    print("\n" + "="*40)
    print(f"STARTING TESTS FOR USER: {test_user}")
    print("="*40)

    # 1. Clean start (optional but recommended for testing)
    print("\n[1/5] Preparing fresh collection...")
    clear_user_data(test_user)

    # 2. Add sample data
    print("[2/5] Adding sample documents...")
    ai_chunks = [
        'Machine learning is a type of artificial intelligence.',
        'Neural networks are inspired by the human brain.',
        'Deep learning uses multiple layers of neural networks.'
    ]
    python_chunks = [
        'Python is a popular programming language.',
        'Python is used for web development and data science.'
    ]
    
    add_to_collection(test_user, ai_chunks, 'ai_notes.pdf', 'document')
    add_to_collection(test_user, python_chunks, 'python_guide.txt', 'document')
    print("      ✓ Added ai_notes.pdf and python_guide.txt")

    # 3. Verify Stats and Sources
    print("\n[3/5] Verifying database content...")
    stats = get_user_stats(test_user)
    print(f"      Stats: {stats['total_chunks']} chunks from {stats['total_sources']} sources")
    
    sources = get_user_sources(test_user)
    for s in sources:
        print(f"      - {s['name']} ({s['chunks']} chunks)")

    # 4. Test Search
    query = "What is deep learning?"
    print(f"\n[4/5] Testing search query: '{query}'")
    results = search_collection(test_user, query, top_k=2)
    
    for i, doc in enumerate(results['documents'], 1):
        print(f"      Result {i}: {doc}")
    print(f"      Sources found: {results['sources']}")

    # 5. Test Deletion
    print("\n[5/5] Testing partial deletion (python_guide.txt)...")
    delete_source(test_user, 'python_guide.txt')
    new_stats = get_user_stats(test_user)
    print(f"      Remaining chunks: {new_stats['total_chunks']}")
    
    print("\n" + "="*40)
    print("TESTS COMPLETED SUCCESSFULLY")
    print("="*40 + "\n")