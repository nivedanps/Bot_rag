# Simple Local PDF RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) system that runs entirely on your local machine using Hugging Face models and FAISS vector search.

---

## Architecture & Pipeline

```text
sample.pdf
    ↓
Extract text using PyPDF (load_pdf)
    ↓
Split text into chunks (split_text)
    ↓
Create embeddings using BAAI/bge-small-en-v1.5 (create_embeddings)
    ↓
Store embeddings in FAISS (faiss.IndexFlatIP)
    ↓
User enters question in terminal
    ↓
Find top relevant chunks (search)
    ↓
Send chunks + question to Qwen LLM (generate_answer)
    ↓
Display answer in terminal
```

---

## Quickstart Guide

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

You can specify any PDF path directly in two ways:

#### Option A: Pass PDF path via command line
```bash
python app.py /path/to/your/custom_document.pdf
```

#### Option B: Interactive prompt or default `sample.pdf`
```bash
python app.py
```
When prompted, type or paste the path to your PDF (or press **Enter** to use `sample.pdf`).

---

### 4. Ask Questions

```text
=================================
       LOCAL PDF RAG
=================================

Enter PDF file path (press Enter for 'sample.pdf'): /path/to/my_doc.pdf

Loading /path/to/my_doc.pdf...
Creating chunks...
Creating embeddings...
Loading Hugging Face model...

RAG is ready!

Ask questions about your PDF.
Type 'exit' to quit.

Question: What is the main topic of the PDF?

Answer:
This document explains ...

Sources:
my_doc.pdf
----------------------------------------
```

Type `exit` to quit the program.

---

## Project Structure

```text
rag-project/
├── app.py           # Complete beginner-friendly RAG implementation
├── sample.pdf       # Input document for question answering
├── requirements.txt # Minimal dependencies
└── README.md        # Documentation
```
