import os
import re
import hashlib
from pathlib import Path

import streamlit as st

APP_NAME = "Cyber Law GPT"
PDF_ID = "1jCgiDz_A5dzAOASNv9Xg7c62IIg1FaSx"
PDF_URL = f"https://drive.google.com/uc?id={PDF_ID}"
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "pakistan_cyber_law.pdf"
INDEX_PATH = DATA_DIR / "cyber_law.index"
CHUNKS_PATH = DATA_DIR / "chunks.npy"
FINGERPRINT_PATH = DATA_DIR / "fingerprint.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title=APP_NAME, page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;margin-bottom:0}
.subtitle{color:#777;font-size:17px;margin-bottom:20px}
.notice{padding:14px;border-radius:10px;background:rgba(255,193,7,.10);border:1px solid rgba(255,193,7,.25)}
</style>
""", unsafe_allow_html=True)


def import_dependencies():
    """Import heavy dependencies only after Streamlit has rendered the page.
    This makes deployment errors visible instead of showing only 'Oh no'."""
    try:
        import numpy as np
        import fitz
        import gdown
        import faiss
        from sentence_transformers import SentenceTransformer
        from groq import Groq
        return np, fitz, gdown, faiss, SentenceTransformer, Groq
    except Exception as exc:
        raise RuntimeError(
            "A required Python package could not be imported. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


@st.cache_resource(show_spinner=False)
def load_embedder():
    _, _, _, _, SentenceTransformer, _ = import_dependencies()
    return SentenceTransformer(EMBEDDING_MODEL)


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_pdf(gdown):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10000:
        try:
            import fitz
            with fitz.open(PDF_PATH) as doc:
                if len(doc) > 0:
                    return
        except Exception:
            PDF_PATH.unlink(missing_ok=True)

    try:
        result = gdown.download(PDF_URL, str(PDF_PATH), quiet=True, fuzzy=True)
        if not result:
            raise RuntimeError("gdown returned no file")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the cyber-law PDF from Google Drive. "
            "Make sure the Drive file is shared as 'Anyone with the link → Viewer'. "
            f"Details: {exc}"
        ) from exc

    try:
        import fitz
        with fitz.open(PDF_PATH) as doc:
            if len(doc) == 0:
                raise RuntimeError("PDF has zero pages")
    except Exception as exc:
        raise RuntimeError(f"Downloaded file is not a valid PDF: {exc}") from exc


def extract_pages(fitz):
    pages = []
    with fitz.open(PDF_PATH) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"page": page_no, "text": text})

    if not pages:
        raise RuntimeError(
            "No selectable text was found in the PDF. The PDF may be scanned and need OCR."
        )
    return pages


def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if len(piece) >= 80:
                chunks.append({"page": item["page"], "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def fingerprint():
    h = hashlib.sha256()
    h.update(PDF_PATH.read_bytes())
    h.update(f"{EMBEDDING_MODEL}|900|150".encode())
    return h.hexdigest()


@st.cache_resource(show_spinner=True)
def build_knowledge_base():
    np, fitz, gdown, faiss, _, _ = import_dependencies()
    download_pdf(gdown)
    fp = fingerprint()

    if INDEX_PATH.exists() and CHUNKS_PATH.exists() and FINGERPRINT_PATH.exists():
        try:
            if FINGERPRINT_PATH.read_text().strip() == fp:
                index = faiss.read_index(str(INDEX_PATH))
                chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
                if len(chunks) == index.ntotal and index.ntotal > 0:
                    return index, chunks
        except Exception:
            pass

    pages = extract_pages(fitz)
    chunks = make_chunks(pages)
    if not chunks:
        raise RuntimeError("The PDF produced no searchable chunks.")

    model = load_embedder()
    texts = [x["text"] for x in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    try:
        faiss.write_index(index, str(INDEX_PATH))
        np.save(CHUNKS_PATH, np.array(chunks, dtype=object), allow_pickle=True)
        FINGERPRINT_PATH.write_text(fp)
    except Exception:
        # The app can still work for the current process if cloud storage is read-only.
        pass

    return index, chunks


def retrieve(question, index, chunks, model, top_k, np):
    q = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({"page": chunks[int(idx)]["page"], "text": chunks[int(idx)]["text"], "score": float(score)})
    return results


def get_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


def make_prompt(question, results, level, size, language, strict):
    context = "\n\n".join(
        f"[PDF PAGE {r['page']}]\n{r['text']}" for r in results
    )

    strict_rule = (
        "Use only the retrieved PDF context for legal claims. If the context is insufficient, explicitly say so and do not guess."
        if strict else
        "Prefer the PDF context. If general context is added, label it clearly as general information and never present it as a quoted provision."
    )

    size_rules = {
        "Short": "Keep the answer around 100-180 words.",
        "Medium": "Give a balanced answer around 200-350 words.",
        "Detailed": "Give a detailed answer with useful headings and careful explanation."
    }

    level_rules = {
        "Beginner": "Use simple language and explain legal terms.",
        "Intermediate": "Use moderate legal terminology and explain important terms.",
        "Advanced": "Use detailed legal and technical terminology.",
        "Legal / Technical": "Use professional legal terminology and identify provisions only when supported by the source."
    }

    language_rules = {
        "English": "Answer in English.",
        "Urdu": "Answer in Urdu script.",
        "Roman Urdu": "Answer in Roman Urdu.",
        "English + Urdu": "Use a clear mix of English and Urdu."
    }

    return f"""You are Cyber Law GPT, a Pakistan cyber-law RAG assistant.

GROUNDING RULE:
{strict_rule}

Never invent section numbers, penalties, fines, procedures, authorities, dates, or legal conclusions.
If a question asks about a situation, explain possible relevance of the retrieved law without pretending to issue a court decision.

TECHNICAL LEVEL: {level}
{level_rules[level]}

RESPONSE SIZE: {size}
{size_rules[size]}

LANGUAGE:
{language_rules[language]}

When possible structure the answer as:
1. Relevant Law
2. Explanation
3. Application to the question
4. Penalty/Procedure (only if supported)
5. Source page(s)
6. Short disclaimer: This is general legal information, not legal advice.

RETRIEVED PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""


def ask_groq(prompt, model_name, temperature, max_tokens):
    _, _, _, _, _, Groq = import_dependencies()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets."
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful Pakistan cyber-law information assistant. Accuracy is more important than confidence."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def show_sources(results):
    with st.expander("📚 Retrieved legal sources", expanded=False):
        for i, r in enumerate(results, 1):
            st.markdown(f"**Source {i} — PDF page {r['page']} — similarity {r['score']:.3f}**")
            st.caption(r["text"][:650] + ("..." if len(r["text"]) > 650 else ""))
            if i < len(results):
                st.divider()


# ---------------- UI ----------------
st.markdown('<div class="main-title">⚖️ Cyber Law GPT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RAG-powered Pakistan Cyber Law Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ This application provides general legal information grounded in the supplied cyber-law PDF. It is not a lawyer, court, or government authority.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    level = st.selectbox("Technical level", ["Beginner", "Intermediate", "Advanced", "Legal / Technical"])
    size = st.selectbox("Response size", ["Short", "Medium", "Detailed"], index=1)
    language = st.selectbox("Answer language", ["English", "Urdu", "Roman Urdu", "English + Urdu"])
    top_k = st.slider("Retrieved passages", 3, 10, 5)
    temperature = st.slider("Creativity", 0.0, 0.5, 0.1, 0.05)
    strict = st.toggle("Strict PDF-only mode", value=True)
    model_name = st.selectbox("Groq model", [DEFAULT_MODEL, "llama-3.3-70b-versatile"])
    st.divider()
    st.caption("PDF is downloaded and embedded automatically on first startup. The FAISS index is reused when possible.")

# Diagnostics: imports happen after the page is rendered.
try:
    np, fitz, gdown, faiss, _, _ = import_dependencies()
except Exception as exc:
    st.error("❌ Dependency/import error")
    st.code(str(exc))
    st.info("Commit the requirements.txt supplied with this project, then redeploy using Python 3.12.")
    st.stop()

try:
    with st.spinner("📚 Downloading PDF and loading/building FAISS index..."):
        index, chunks = build_knowledge_base()
        embedder = load_embedder()
    st.success(f"Knowledge base ready • {len(chunks)} searchable chunks • FAISS active")
except Exception as exc:
    st.error("❌ Knowledge-base startup failed")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown("**If this is a Google Drive error:** make the PDF `Anyone with the link → Viewer`.\n\n**If this is a memory/resource error:** reboot the app and use Python 3.12.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask any question about Pakistan cyber law...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Searching relevant law..."):
                results = retrieve(question, index, chunks, embedder, top_k, np)
                prompt = make_prompt(question, results, level, size, language, strict)

            with st.spinner("🤖 Generating grounded answer..."):
                max_tokens = {"Short": 700, "Medium": 1300, "Detailed": 2000}[size]
                answer = ask_groq(prompt, model_name, temperature, max_tokens)

            st.markdown(answer)
            show_sources(results)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
        except Exception as exc:
            error_text = f"⚠️ **Error:** {type(exc).__name__}: {exc}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
