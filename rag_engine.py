from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K, CATEGORIES
from chunker import split_into_chunks
from document_loader import load_document
from web_scraper import scan_links, scrape_page, scrape_multiple_pages
from user_manager import add_to_collection, search_collection
import base64
groq_client = Groq(api_key=GROQ_API_KEY)
conversation_history = {}
MAX_HISTORY = 10

def transcribe_audio(file_path):
    """Convert a voice message to text using Groq Whisper.
    
    Args:
        file_path: path to the audio file
    
    Returns:
        dict with text and success
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
            )

        text = transcription.text.strip()

        if not text:
            return {
                "text": "",
                "success": False,
                "error": "Could not understand the audio"
            }

        return {
            "text": text,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }


def text_to_speech(text, file_path):
    """Convert text to speech using Groq Orpheus.
    Splits long text into chunks and combines the audio.
    
    Args:
        text: the text to convert to audio
        file_path: where to save the audio file
    
    Returns:
        dict with success and error
    """
    import wave
    import os

    try:
        # Split text into chunks under 200 characters at sentence boundaries
        chunks = _split_text_for_tts(text)

        if not chunks:
            return {"success": False, "error": "No text to convert"}

        # Generate audio for each chunk
        temp_files = []
        for i, chunk in enumerate(chunks):
            temp_path = file_path.replace(".wav", f"_part{i}.wav")
            temp_files.append(temp_path)

            response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="hannah",
                input=chunk,
                response_format="wav"
            )
            response.write_to_file(temp_path)

        # If only one chunk, just rename it
        if len(temp_files) == 1:
            os.rename(temp_files[0], file_path)
            return {"success": True, "error": None}

        # Combine all audio files into one
        _combine_wav_files(temp_files, file_path)

        # Clean up temp files
        for temp in temp_files:
            if os.path.exists(temp):
                os.remove(temp)

        return {"success": True, "error": None}

    except Exception as e:
        print(f"TTS Error: {e}")
        # Clean up any temp files on error
        for temp in temp_files if 'temp_files' in dir() else []:
            if os.path.exists(temp):
                os.remove(temp)
        return {"success": False, "error": str(e)}

def analyze_image(file_path):
    """Analyze an image and extract all information from it.
    
    Args:
        file_path: path to the image file
    
    Returns:
        dict with text, success, error
    """
    try:
        # Step 1: Read the image and convert to base64
        with open(file_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Step 2: Detect the image type
        if file_path.lower().endswith(".png"):
            media_type = "image/png"
        elif file_path.lower().endswith(".gif"):
            media_type = "image/gif"
        elif file_path.lower().endswith(".webp"):
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

        # Step 3: Send to the vision model
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image deeply and extract ALL information. "
                                "Include:\n"
                                "1. Any text visible in the image (transcribe it exactly)\n"
                                "2. Any numbers, equations, or formulas\n"
                                "3. Any diagrams, charts, or tables (describe them in detail)\n"
                                "4. Any objects, people, or scenes\n"
                                "5. The overall context and purpose of the image\n\n"
                                "Be as thorough and detailed as possible."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_completion_tokens=2048
        )

        text = response.choices[0].message.content

        if not text:
            return {
                "text": "",
                "success": False,
                "error": "Could not analyze the image"
            }

        return {
            "text": text,
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"Vision Error: {e}")
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }

def process_image(user_id, file_path, source_name=None, add_to_kb=True, category=None):
    """Analyze an image and optionally add to knowledge base."""
    result = analyze_image(file_path)

    if not result["success"]:
        return {
            "analysis": "",
            "chunks_added": 0,
            "success": False,
            "error": result["error"],
            "category": "general"
        }

    analysis = result["text"]

    chunks_added = 0
    if add_to_kb and analysis:
        if not category:
            category = detect_category(analysis)

        name = source_name or "image"
        chunks = split_into_chunks(analysis)
        if chunks:
            chunks_added = add_to_collection(user_id, chunks, name, "image", category)

    return {
        "analysis": analysis,
        "chunks_added": chunks_added,
        "success": True,
        "error": None,
        "category": category or "general"
    }


def _split_text_for_tts(text, max_length=190):
    """Split text into chunks under max_length, breaking at sentence boundaries."""
    # Clean up the text
    text = text.strip()
    if not text:
        return []

    # If short enough, return as is
    if len(text) <= max_length:
        return [text]

    # Split into sentences
    sentences = []
    for part in text.split(". "):
        part = part.strip()
        if part:
            if not part.endswith("."):
                part += "."
            sentences.append(part)

    # Group sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence is too long, split by commas
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by commas
            parts = sentence.split(", ")
            for part in parts:
                part = part.strip()
                if len(current_chunk) + len(part) + 2 < max_length:
                    current_chunk += ", " + part if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
        
        elif len(current_chunk) + len(sentence) + 1 < max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _combine_wav_files(input_files, output_file):
    """Combine multiple WAV files into one."""
    import wave

    # Read the first file to get audio parameters
    with wave.open(input_files[0], 'rb') as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    # Read remaining files
    for file_path in input_files[1:]:
        with wave.open(file_path, 'rb') as wf:
            frames.append(wf.readframes(wf.getnframes()))

    # Write combined file
    with wave.open(output_file, 'wb') as output:
        output.setnchannels(params.nchannels)
        output.setsampwidth(params.sampwidth)
        output.setframerate(params.framerate)
        # Don't set nframes — let Python calculate it from the data
        for frame_data in frames:
            output.writeframes(frame_data)

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

def process_document(user_id, file_path, source_name=None, category=None):
    """Process a document and add it to the user's knowledge base."""
    try:
        text = load_document(file_path, analyze_images_fn=analyze_image)

        chunks = split_into_chunks(text)

        if not chunks:
            return {
                "source": source_name or file_path,
                "chunks_added": 0,
                "success": False,
                "error": "No text could be extracted",
                "category": "general"
            }

        # Auto-detect category if not provided
        if not category:
            category = detect_category(text)

        name = source_name or file_path.split("/")[-1]
        chunks_added = add_to_collection(user_id, chunks, name, "document", category)

        return {
            "source": name,
            "chunks_added": chunks_added,
            "success": True,
            "error": None,
            "category": category
        }

    except Exception as e:
        return {
            "source": source_name or file_path,
            "chunks_added": 0,
            "success": False,
            "error": str(e),
            "category": "general"
        }

def process_url(user_id, url, category=None):
    """Scrape a URL and add its content to the user's knowledge base."""
    try:
        text = scrape_page(url)

        if not text:
            return {
                "source": url,
                "chunks_added": 0,
                "success": False,
                "error": "No content found on page",
                "category": "general"
            }

        chunks = split_into_chunks(text)

        if not chunks:
            return {
                "source": url,
                "chunks_added": 0,
                "success": False,
                "error": "No chunks created",
                "category": "general"
            }

        if not category:
            category = detect_category(text)

        chunks_added = add_to_collection(user_id, chunks, url, "webpage", category)

        return {
            "source": url,
            "chunks_added": chunks_added,
            "success": True,
            "error": None,
            "category": category
        }

    except Exception as e:
        return {
            "source": url,
            "chunks_added": 0,
            "success": False,
            "error": str(e),
            "category": "general"
        }


def process_urls(user_id, urls, category=None):
    """Scrape multiple URLs and add their content.
    
    Args:
        user_id: Telegram user ID
        urls: list of URLs
    
    Returns:
        list of result dicts
    """
    results = []
    for url in urls:
        result = process_url(user_id, url, category=category)
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
    # Step 0: ROUTE — detect which categories to search
    categories = detect_question_categories(question)

    # Step 1: RETRIEVE with category filtering
    search_results = search_collection(user_id, question, top_k=TOP_K, categories=categories)

    documents = search_results["documents"]
    sources = search_results["sources"]

    if not documents:
        return {
            "answer": "Your knowledge base is empty. Please add some documents or websites first!",
            "sources": [],
            "chunks_used": 0,
            "categories_searched": categories
        }

    # Step 2: AUGMENT
    context = "\n\n".join(documents)

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful personal assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context doesn't contain enough information to answer, "
            "say you don't have enough information in the user's language. "
            "IMPORTANT: Always answer in the SAME LANGUAGE the user asked in. "
            "Be clear and concise."
        )
    }

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

        add_to_history(user_id, "user", question)
        add_to_history(user_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(documents),
            "categories_searched": categories
        }

    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "chunks_used": 0,
            "categories_searched": categories
        }


def detect_category(text):
    """Use the LLM to detect which category a document belongs to.
    
    Args:
        text: sample text from the document (first 1000 chars)
    
    Returns:
        category key (e.g., "work", "health", "general")
    """
    category_list = "\n".join([
        f"- {key}: {info['label']}"
        for key, info in CATEGORIES.items()
    ])

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document classifier. "
                        "Based on the text provided, determine which single category "
                        "best fits this document. Respond with ONLY the category key, "
                        "nothing else. No explanation, no punctuation, just the key."
                    )
                },
                {
                    "role": "user",
                    "content": f"Categories:\n{category_list}\n\nDocument text:\n{text[:1000]}"
                }
            ],
            temperature=0,
            max_completion_tokens=20
        )

        detected = response.choices[0].message.content.strip().lower()

        # Make sure the response is a valid category
        if detected in CATEGORIES:
            return detected
        
        # If LLM returned the label instead of the key, find the key
        for key, info in CATEGORIES.items():
            if key in detected:
                return key

        return "general"

    except Exception:
        return "general"


def detect_question_categories(question):
    """Detect which categories a question might relate to.
    
    Args:
        question: the user's question
    
    Returns:
        list of category keys, or ["all"] if unclear
    """
    category_list = "\n".join([
        f"- {key}: {info['label']}"
        for key, info in CATEGORIES.items()
    ])

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query router. Based on the question, determine which "
                        "categories of documents would most likely contain the answer. "
                        "Respond with ONLY the category keys separated by commas. "
                        "If the question is general or could span multiple categories, "
                        "respond with 'all'. No explanation, just the keys."
                    )
                },
                {
                    "role": "user",
                    "content": f"Categories:\n{category_list}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0,
            max_completion_tokens=50
        )

        result = response.choices[0].message.content.strip().lower()

        if "all" in result:
            return ["all"]

        # Parse the comma-separated categories
        categories = [c.strip() for c in result.split(",")]
        valid = [c for c in categories if c in CATEGORIES]

        return valid if valid else ["all"]

    except Exception:
        return ["all"]


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