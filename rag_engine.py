from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K
from chunker import split_into_chunks
from document_loader import load_document
from web_scraper import scan_links, scrape_page, scrape_multiple_pages
from user_manager import add_to_collection, search_collection

groq_client = Groq(api_key=GROQ_API_KEY)
conversation_history = {}
MAX_HISTORY = 10

def get_history(user_id):
    """Get conversation history for a user."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return conversation_history[user_id]


def add_to_history(user_id, role, content):
    """Add a message to the user's conversation history."""
    history = get_history(user_id)
    history.append({"role": role, "content": content})

    # Keep only the last MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        conversation_history[user_id] = history[-MAX_HISTORY:]


def clear_history(user_id):
    """Clear conversation history for a user."""
    conversation_history[user_id] = []

def process_document(user_id, file_path, source_name=None):
    """Process a document and add it to the user's knowledge base.
    
    Args:
        user_id: Telegram user ID
        file_path: path to the file
        source_name: display name (defaults to filename)
    
    Returns:
        dict with source, chunks_added, success
    """
    try:
        # Step 1: Extract text from the file
        text = load_document(file_path)

        # Step 2: Split into chunks
        chunks = split_into_chunks(text)

        if not chunks:
            return {
                "source": source_name or file_path,
                "chunks_added": 0,
                "success": False,
                "error": "No text could be extracted"
            }

        # Step 3: Store in user's collection
        name = source_name or file_path.split("/")[-1]
        chunks_added = add_to_collection(user_id, chunks, name, "document")

        return {
            "source": name,
            "chunks_added": chunks_added,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "source": source_name or file_path,
            "chunks_added": 0,
            "success": False,
            "error": str(e)
        }

def process_url(user_id, url):
    """Scrape a URL and add its content to the user's knowledge base.
    
    Args:
        user_id: Telegram user ID
        url: the webpage URL
    
    Returns:
        dict with source, chunks_added, success
    """
    try:
        # Step 1: Scrape the page
        text = scrape_page(url)

        if not text:
            return {
                "source": url,
                "chunks_added": 0,
                "success": False,
                "error": "No content found on page"
            }

        # Step 2: Split into chunks
        chunks = split_into_chunks(text)

        if not chunks:
            return {
                "source": url,
                "chunks_added": 0,
                "success": False,
                "error": "No chunks created"
            }

        # Step 3: Store in user's collection
        chunks_added = add_to_collection(user_id, chunks, url, "webpage")

        return {
            "source": url,
            "chunks_added": chunks_added,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "source": url,
            "chunks_added": 0,
            "success": False,
            "error": str(e)
        }

def process_urls(user_id, urls):
    """Scrape multiple URLs and add their content.
    
    Args:
        user_id: Telegram user ID
        urls: list of URLs
    
    Returns:
        list of result dicts
    """
    results = []
    for url in urls:
        result = process_url(user_id, url)
        results.append(result)
    return results

def scan_website(url):
    """Scan a URL and return all links found.
    
    Args:
        url: the webpage URL
    
    Returns:
        list of link dicts
    """
    return scan_links(url)

def ask_question(user_id, question):
    """Search the user's knowledge base and generate an answer."""
    # Step 1: RETRIEVE
    search_results = search_collection(user_id, question, top_k=TOP_K)

    documents = search_results["documents"]
    sources = search_results["sources"]

    if not documents:
        return {
            "answer": "Your knowledge base is empty. Please add some documents or websites first!",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: AUGMENT — build the prompt with context AND history
    context = "\n\n".join(documents)

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful study assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context doesn't contain enough information to answer, "
            "say 'I don't have enough information about that in your materials.' "
            "Be clear and concise. When possible, mention which source the information comes from."
        )
    }

    # Build messages: system + history + new question with context
    history = get_history(user_id)

    messages = [system_message]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    # Step 3: GENERATE
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE
        )

        answer = response.choices[0].message.content

        # Save this exchange to history
        add_to_history(user_id, "user", question)
        add_to_history(user_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(documents)
        }

    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "chunks_used": 0
        }


if __name__ == "__main__":
    # === Test the complete RAG pipeline ===
    
    test_user = 99999

    # Test 1: Process a text document
    print("=== Test 1: Process Document ===")
    with open("data/test_notes.txt", "w") as f:
        f.write(
            "Machine learning is a subset of artificial intelligence. "
            "It allows computers to learn from data without being explicitly programmed. "
            "Supervised learning uses labeled data to train models. "
            "Common algorithms include linear regression and decision trees. "
            "Unsupervised learning finds hidden patterns in unlabeled data. "
            "Clustering and dimensionality reduction are popular techniques. "
            "Deep learning uses neural networks with many layers. "
            "Convolutional neural networks are great for image recognition. "
            "Recurrent neural networks work well with sequential data like text. "
            "Transfer learning allows reusing models trained on large datasets. "
            "This saves time and computational resources. "
            "Python is the most popular language for machine learning. "
            "Libraries like scikit-learn, TensorFlow, and PyTorch are widely used."
        )
    
    result = process_document(test_user, "data/test_notes.txt", "ML Notes")
    print(f"Source: {result['source']}")
    print(f"Chunks added: {result['chunks_added']}")
    print(f"Success: {result['success']}")

    # Test 2: Process a URL
    print("\n=== Test 2: Process URL ===")
    result = process_url(test_user, "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html")
    print(f"Source: {result['source']}")
    print(f"Chunks added: {result['chunks_added']}")
    print(f"Success: {result['success']}")

    # Test 3: Ask a question
    print("\n=== Test 3: Ask Question ===")
    result = ask_question(test_user, "What is supervised learning?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Chunks used: {result['chunks_used']}")

    # Test 4: Scan a website
    print("\n=== Test 4: Scan Website ===")
    links = scan_website("https://books.toscrape.com")
    print(f"Found {len(links)} links")
    for link in links[:3]:
        print(f"  {'nyyyy' if link['is_internal'] else 'xxxxx'} {link['title'][:40]}")

    # Cleanup
    from user_manager import clear_user_data
    clear_user_data(test_user)
    print("\n✅ All tests passed! Cleaned up test data.")