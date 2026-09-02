import os
import sys
import torch
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_PDF_FILE = "sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Fast, lightweight local LLM (you can also switch to Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-3B-Instruct)
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Automatically choose best device (Mac Metal/MPS, Nvidia CUDA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ==============================================================================
# 1. PDF LOADING
# ==============================================================================
def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using PyPDF."""
    # Clean up quotes and escaped spaces (common when dragging files into terminal)
    clean_path = os.path.expanduser(file_path.strip("'\"").strip().replace("\\ ", " "))

    # If exact path not found, try fuzzy match in current directory
    if not os.path.exists(clean_path):
        current_dir_files = os.listdir(".")
        matched = [f for f in current_dir_files if clean_path.lower() in f.lower() or f.lower() in clean_path.lower()]
        if matched and os.path.isfile(matched[0]):
            clean_path = matched[0]

    if not os.path.exists(clean_path):
        print(f"Error: '{clean_path}' not found.")
        print(f"Please check the path and try again.")
        sys.exit(1)

    try:
        reader = PdfReader(clean_path)
    except Exception as e:
        print(f"Error reading PDF file '{clean_path}': {e}")
        sys.exit(1)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    if not full_text.strip():
        print(f"Warning: No text could be extracted from '{clean_path}'.")
        sys.exit(1)

    return full_text


# ==============================================================================
# 2. CHUNKING
# ==============================================================================
def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split full text into smaller overlapping chunks for retrieval."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance pointer with overlap
        start += chunk_size - chunk_overlap

    return chunks


# ==============================================================================
# 3. EMBEDDINGS & FAISS INDEXING
# ==============================================================================
def create_embeddings(chunks: list[str], model_name: str = EMBEDDING_MODEL_NAME):
    """
    Generate vector embeddings for each chunk using SentenceTransformers
    and store them in a FAISS vector index.
    """
    embed_model = SentenceTransformer(model_name)

    # Generate normalized embeddings for cosine similarity
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Dimension of the vector space
    dimension = embeddings.shape[1]

    # Create Inner Product (Cosine similarity for normalized vectors) FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embed_model, index


# ==============================================================================
# 4. FAISS SIMILARITY SEARCH
# ==============================================================================
def search(query: str, embed_model, index, chunks: list[str], top_k: int = TOP_K) -> list[str]:
    """Search FAISS index for the most relevant text chunks given a query."""
    query_vector = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Search top_k closest vectors
    distances, indices = index.search(query_vector, min(top_k, len(chunks)))

    retrieved_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    return retrieved_chunks


# ==============================================================================
# 5. LLM ANSWER GENERATION
# ==============================================================================
def generate_answer(question: str, context: str, model, tokenizer) -> str:
    """Send retrieved context and user question to the local LLM to generate an answer."""
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question:
{question}

Instructions:
- Answer only using the context.
- Do not make up information.
- If the answer is not in the context, say:
  "I could not find the answer in the PDF."
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You must answer questions ONLY using the provided context. If the answer is not explicitly in the context, you must strictly reply: 'I could not find the answer in the PDF.'"
        },
        {"role": "user", "content": prompt}
    ]

    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_chat, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens (ignoring the prompt tokens)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
    return response


# ==============================================================================
# 6. MAIN APPLICATION LOOP
# ==============================================================================
def main():
    print("=================================")
    print("       LOCAL PDF RAG")
    print("=================================")
    print()

    # Step 0: Get PDF file path (from CLI argument or interactive prompt)
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1].strip()
    else:
        user_input = input(f"Enter PDF file path (press Enter for '{DEFAULT_PDF_FILE}'): ").strip()
        pdf_file = user_input if user_input else DEFAULT_PDF_FILE

    # Step 1: Load PDF
    print(f"\nLoading {pdf_file}...")
    pdf_text = load_pdf(pdf_file)
    print(f"Document loaded successfully ({len(pdf_text)} characters).")

    # Step 2: Split text into chunks
    print("Creating chunks...")
    chunks = split_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print("Error: No chunks created from document.")
        sys.exit(1)
    print(f"Total chunks created: {len(chunks)} (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

    # Step 3: Embed chunks and build vector index
    print("Creating embeddings...")
    embed_model, index = create_embeddings(chunks, EMBEDDING_MODEL_NAME)

    # Step 4: Load local Hugging Face LLM
    print("Loading Hugging Face model...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    print()
    print("RAG is ready!")
    print()
    print("Ask questions about your PDF.")
    print("Type 'exit' to quit.")
    print()

    # Interactive Q&A loop
    while True:
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve top relevant context chunks
        relevant_chunks = search(query, embed_model, index, chunks, TOP_K)
        context = "\n\n".join(relevant_chunks)

        # Generate answer using context
        answer = generate_answer(query, context, model, tokenizer)

        print()
        print("Answer:")
        print(answer)
        print()
        print("Sources:")
        print(os.path.basename(pdf_file))
        print("-" * 40)
        print()


if __name__ == "__main__":
    main()
