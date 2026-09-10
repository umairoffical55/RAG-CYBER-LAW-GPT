# ⚖️ Cyber Law GPT

**Cyber Law GPT** is a free-to-run Retrieval-Augmented Generation (RAG) application for answering questions about cyber law in Pakistan. It uses a Pakistan cyber-law PDF as the knowledge base, FAISS for semantic retrieval, Sentence Transformers for embeddings, and Groq for response generation.

> **Legal disclaimer:** This application provides legal information for education/research. It is not a substitute for advice from a qualified Pakistani lawyer. Laws and amendments can change, so verify important matters against the latest official Gazette/law source.

## Features

- Automatic Google Drive PDF download on startup
- Automatic PDF text extraction with PyMuPDF
- Automatic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector search
- Groq LLM generation
- Technical-level selector: Beginner, Intermediate, Advanced, Legal/Technical
- Response-size selector: Short, Medium, Detailed
- English, Urdu, Roman Urdu, or bilingual answers
- Adjustable number of retrieved law chunks
- Adjustable creativity/temperature
- Strict PDF-only legal mode
- Retrieved page/source snippets shown under every answer
- Startup caching: embeddings are reused unless the source PDF changes
- Works locally, in Google Colab, and on Streamlit Community Cloud

## Project files

```text
cyber-law-gpt/
├── app.py
├── requirements.txt
└── readme.md
```

## 1. Get a Groq API key

Create a Groq API key from the Groq console and keep it private.

### Google Colab / local

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Or in a terminal:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

## 2. Run in Google Colab

Upload `app.py` and `requirements.txt` to Colab, then run:

```bash
!pip install -r requirements.txt
```

Set the API key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Start Streamlit:

```bash
!streamlit run app.py &>/content/logs.txt &
```

To expose it from Colab, use a tunnel such as Cloudflare Tunnel or ngrok. The application itself does not require paid infrastructure.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the configured Google Drive PDF automatically and creates the FAISS index/embeddings on first startup.

## 4. Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `readme.md`.
3. Open Streamlit Community Cloud.
4. Select the repository and `app.py` as the main file.
5. Add a secret named `GROQ_API_KEY` in the app's Secrets settings:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

6. Deploy.

The first startup downloads the PDF and builds embeddings. Later restarts reuse the generated index while the application environment persists it. If the deployment is rebuilt from scratch, embeddings may be generated again.

## RAG pipeline

```text
Pakistan Cyber-Law PDF
        ↓
Google Drive download
        ↓
PyMuPDF text extraction
        ↓
Text cleaning + overlapping chunks
        ↓
Sentence Transformer embeddings
        ↓
FAISS IndexFlatIP
        ↓
User question
        ↓
Question embedding
        ↓
Top-K relevant legal passages
        ↓
Groq LLM
        ↓
Grounded legal-information answer + page sources
```

## Why this design?

The model is instructed to ground legal claims in retrieved PDF passages and avoid inventing section numbers, penalties, procedures, or legal conclusions. This is especially important for legal applications. The UI also exposes the retrieved pages so the user can inspect the evidence used by the RAG pipeline.

## Updating the law PDF

Replace the Google Drive file referenced by `PDF_ID` in `app.py` or update the ID. The application computes a SHA-256 fingerprint of the downloaded PDF. When the source changes, the FAISS index and embeddings are rebuilt automatically.

## Important limitation

The application answers from the configured PDF. It should not be marketed as an official Pakistani legal authority. For a production legal product, add versioned legislation, Gazette verification, amendment tracking, jurisdiction/date awareness, stronger citation extraction, and professional legal review.
