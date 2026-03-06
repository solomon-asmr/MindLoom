from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks, breaking at sentence boundaries with overlap."""

    text = text.strip()

    if not text:
        return []

    sentences = []
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence + ".")

    chunks = []
    current_chunk = ""
    current_sentences = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
            current_sentences.append(sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            overlap_text = ""
            overlap_sentences = []
            for s in reversed(current_sentences):
                if len(overlap_text) + len(s) < overlap:
                    overlap_text = s + " " + overlap_text if overlap_text else s
                    overlap_sentences.insert(0, s)
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
