import chromadb
from config import CHROMA_DB_PATH
# Create ONE client that all functions share
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_user_collection(user_id):
    """Get or create a ChromaDB collection for a specific user."""
    collection_name = f"user_{user_id}"
    collection = chroma_client.get_or_create_collection(collection_name)
    return collection

def add_to_collection(user_id, chunks, source_name, source_type):
    """Add text chunks to a user's collection.
    
    Args:
        user_id: Telegram user ID
        chunks: list of text strings
        source_name: name of the source (e.g., "notes.pdf")
        source_type: "document" or "webpage"
    
    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    collection = get_user_collection(user_id)

    # Create unique IDs for each chunk
    # We use source_name + index so they don't clash with other sources
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]

    # Create metadata for each chunk (so we know where it came from)
    metadatas = [
        {
            "source": source_name,
            "type": source_type,
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

def search_collection(user_id, query, top_k=3):
    """Search a user's collection for relevant chunks.
    
    Args:
        user_id: Telegram user ID
        query: the user's question
        top_k: how many results to return
    
    Returns:
        dict with "documents" and "sources" keys
    """
    collection = get_user_collection(user_id)

    # Check if collection has any data
    if collection.count() == 0:
        return {"documents": [], "sources": []}

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())
    )

    # Extract the documents and their sources
    documents = results["documents"][0] if results["documents"] else []

    # Get unique source names from metadata
    sources = []
    if results["metadatas"] and results["metadatas"][0]:
        for metadata in results["metadatas"][0]:
            source = metadata.get("source", "Unknown")
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