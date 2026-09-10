import os
import re
import hashlib
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import gdown
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

APP_NAME = "Cyber Law GPT"
PDF_ID = "1jCgiDz_A5dzAOASNv9Xg7c62IIg1FaSx"
PDF_URL = f"https://drive.google.com/uc?id={PDF_ID}"
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "pakistan_cyber_law.pdf"
INDEX_PATH = DATA_DIR / "cyber_law.index"
CHUNKS_PATH = DATA_DIR / "chunks.npy"

st.set_page_config(page_title=APP_NAME, page_icon="⚖️", layout="wide")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_pdf():
    DATA_DIR.mkdir(exist_ok=True)
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size < 10000:
        try:
            gdown.download(PDF_URL, str(PDF_PATH), quiet=True, fuzzy=True)
        except Exception as e:
            raise RuntimeError(f"Could not download the cyber-law PDF from Google Drive: {e}")
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size < 10000:
        raise RuntimeError("The downloaded file is missing or invalid. Check the Google Drive link and sharing permission.")


def extract_pages():
    doc = fitz.open(PDF_PATH)
    pages = []
    for page_no, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append({"page": page_no, "text": text})
    doc.close()
    if not pages:
        raise RuntimeError("No selectable text was found in the PDF. A scanned/OCR PDF needs OCR before indexing.")
    return pages


def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end]
            if len(piece) >= 80:
                chunks.append({"page": item["page"], "text": piece})
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def source_fingerprint():
    h = hashlib.sha256()
    h.update(PDF_PATH.read_bytes())
    h.update(b"all-MiniLM-L6-v2|900|150")
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=True)
def build_or_load_index():
    download_pdf()
    meta_path = DATA_DIR / "fingerprint.txt"
    fingerprint = source_fingerprint()

    if INDEX_PATH.exists() and CHUNKS_PATH.exists() and meta_path.exists() and meta_path.read_text() == fingerprint:
        index = faiss.read_index(str(INDEX_PATH))
        chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
        return index, chunks

    pages = extract_pages()
    chunks = make_chunks(pages)
    texts = [c["text"] for c in chunks]
    model = load_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object), allow_pickle=True)
    meta_path.write_text(fingerprint)
    return index, chunks


def retrieve(query, index, chunks, model, top_k=5):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({"score": float(score), **chunks[int(idx)]})
    return results


def build_prompt(question, results, level, response_size, language, strict_mode):
    context = "\n\n".join(
        f"[Source page {r['page']} | similarity {r['score']:.3f}]\n{r['text']}" for r in results
    )
    strict = (
        "Use ONLY the supplied PDF context for legal claims. If the answer is not supported by the context, say that the PDF does not provide enough information."
        if strict_mode else
        "Prefer the supplied PDF context. If a narrow clarification is needed, clearly label it as general context and do not present unsupported claims as provisions of the Act."
    )
    return f"""You are Cyber Law GPT, a Pakistan cyber-law research assistant.

{strict}
The source document is the Pakistan cyber-law PDF supplied to this application. Treat it as the primary authority for this answer. Do not invent section numbers, penalties, procedures, authorities, dates, or legal conclusions.

User technical level: {level}
Requested response size: {response_size}
Answer language: {language}

For each legal answer:
1. Identify the relevant provision/section when the context supports it.
2. Explain it in plain language appropriate to the selected level.
3. Apply it to the user's scenario carefully, distinguishing facts from assumptions.
4. Mention penalties/procedure only when supported by the retrieved text.
5. End with a short disclaimer: this is legal information, not legal advice.

RETRIEVED PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""


def ask_groq(prompt, model_name, temperature):
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Colab environment variables or Streamlit Cloud Secrets.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful legal information assistant focused on Pakistan cyber law."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1800,
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.title("⚖️ Cyber Law GPT")
st.caption("RAG-powered Pakistan cyber-law assistant • PDF-grounded answers • Groq LLM")

with st.sidebar:
    st.header("⚙️ Response Settings")
    level = st.selectbox("Technical level", ["Beginner", "Intermediate", "Advanced", "Legal / Technical"], index=0)
    response_size = st.select_slider("Response size", options=["Short", "Medium", "Detailed"], value="Medium")
    language = st.selectbox("Answer language", ["English", "Urdu", "Roman Urdu", "English + Urdu"], index=0)
    top_k = st.slider("Retrieved law sections", 3, 10, 5)
    temperature = st.slider("Creativity", 0.0, 0.5, 0.1, 0.05)
    strict_mode = st.toggle("Strict PDF-only legal mode", value=True)
    model_name = st.selectbox("Groq model", ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"], index=0)

    st.divider()
    st.subheader("📚 Knowledge base")
    st.write("Pakistan cyber-law PDF from the configured Google Drive source.")
    st.info("The PDF is downloaded and embedded automatically when the app starts. Cached files are reused until the source PDF changes.")

try:
    with st.spinner("Loading cyber-law PDF and building/loading embeddings..."):
        index, chunks = build_or_load_index()
        embedder = load_embedder()
    st.success(f"Knowledge base ready • {len(chunks)} searchable chunks")
except Exception as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Retrieved law sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}** — similarity `{s['score']:.3f}`")
                    st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

question = st.chat_input("Ask a question about cyber law in Pakistan...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cyber-law knowledge base..."):
            results = retrieve(question, index, chunks, embedder, top_k)
            prompt = build_prompt(question, results, level, response_size, language, strict_mode)
        with st.spinner("Generating answer with Groq..."):
            try:
                answer = ask_groq(prompt, model_name, temperature)
            except Exception as e:
                answer = f"⚠️ Could not generate the answer: {e}"
        st.markdown(answer)
        with st.expander("📌 Retrieved law sources"):
            for s in results:
                st.markdown(f"**Page {s['page']}** — similarity `{s['score']:.3f}`")
                st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
