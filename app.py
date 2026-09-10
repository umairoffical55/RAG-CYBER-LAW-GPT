````python
import os
import re
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import gdown

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# APP CONFIGURATION
# ============================================================

APP_NAME = "Cyber Law GPT"

# Google Drive PDF
PDF_ID = "1jCgiDz_A5dzAOASNv9Xg7c62IIg1FaSx"
PDF_URL = f"https://drive.google.com/uc?id={PDF_ID}"

# Local data directory
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "pakistan_cyber_law.pdf"
INDEX_PATH = DATA_DIR / "cyber_law.index"
CHUNKS_PATH = DATA_DIR / "chunks.npy"
FINGERPRINT_PATH = DATA_DIR / "fingerprint.txt"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default Groq model
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SAFE FAISS IMPORT
# ============================================================

FAISS_AVAILABLE = False
faiss = None

try:
    import faiss

    FAISS_AVAILABLE = True

except Exception:
    FAISS_AVAILABLE = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        background-color: rgba(128,128,128,0.08);
        margin-bottom: 10px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.3);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def get_groq_api_key():
    """
    Get Groq API key from:
    1. Streamlit secrets
    2. Environment variable
    """

    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    env_key = os.getenv("GROQ_API_KEY", "")

    return secret_key or env_key


def clean_text(text):
    """Clean extracted PDF text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def normalize_embeddings(embeddings):
    """
    Normalize vectors for cosine similarity.
    """

    embeddings = np.asarray(embeddings, dtype="float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    norms[norms == 0] = 1.0

    return embeddings / norms


# ============================================================
# PDF DOWNLOAD
# ============================================================

def download_pdf():
    """
    Download the Pakistan cyber-law PDF from Google Drive.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Existing valid PDF
    if PDF_PATH.exists():

        file_size = PDF_PATH.stat().st_size

        if file_size > 10_000:

            # Validate that it is actually a PDF
            try:
                with fitz.open(PDF_PATH) as doc:

                    if len(doc) > 0:
                        return

            except Exception:
                pass

    # Remove invalid file
    if PDF_PATH.exists():
        PDF_PATH.unlink()

    try:

        with st.spinner("Downloading Pakistan cyber-law PDF..."):

            downloaded = gdown.download(
                PDF_URL,
                str(PDF_PATH),
                quiet=False,
                fuzzy=True,
            )

        if not downloaded:
            raise RuntimeError(
                "Google Drive download failed."
            )

    except Exception as e:

        raise RuntimeError(
            "Unable to download the cyber-law PDF.\n\n"
            f"Details: {e}\n\n"
            "Make sure the Google Drive file is shared as "
            "'Anyone with the link → Viewer'."
        )


    # Validate PDF
    try:

        with fitz.open(PDF_PATH) as doc:

            if len(doc) == 0:
                raise RuntimeError(
                    "The downloaded PDF contains no pages."
                )

    except Exception as e:

        raise RuntimeError(
            f"The downloaded file is not a valid PDF: {e}"
        )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages():
    """
    Extract text page-by-page.
    """

    pages = []

    try:

        with fitz.open(PDF_PATH) as document:

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text")

                text = clean_text(text)

                if text:

                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                        }
                    )

    except Exception as e:

        raise RuntimeError(
            f"Could not read the PDF: {e}"
        )

    if not pages:

        raise RuntimeError(
            "No readable text was found in the PDF. "
            "The PDF may be scanned and require OCR."
        )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150,
):
    """
    Split PDF pages into overlapping chunks.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        if len(text) <= chunk_size:

            chunks.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            # Try not to cut in the middle of a word
            if end < len(text):

                last_space = chunk.rfind(" ")

                if last_space > chunk_size * 0.7:

                    chunk = chunk[:last_space]
                    end = start + last_space

            chunk = chunk.strip()

            if len(chunk) >= 100:

                chunks.append(
                    {
                        "page": page_number,
                        "text": chunk,
                    }
                )

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


# ============================================================
# PDF FINGERPRINT
# ============================================================

def get_pdf_fingerprint():

    if not PDF_PATH.exists():
        return ""

    sha = hashlib.sha256()

    with open(PDF_PATH, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha.update(data)

    # Include chunk configuration and embedding model
    sha.update(
        f"{EMBEDDING_MODEL}|1000|150".encode()
    )

    return sha.hexdigest()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    if not FAISS_AVAILABLE:

        raise RuntimeError(
            "FAISS is not installed."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


# ============================================================
# BUILD / LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource(show_spinner=True)
def build_knowledge_base():

    download_pdf()

    fingerprint = get_pdf_fingerprint()

    # --------------------------------------------------------
    # LOAD EXISTING INDEX
    # --------------------------------------------------------

    if (
        FAISS_AVAILABLE
        and INDEX_PATH.exists()
        and CHUNKS_PATH.exists()
        and FINGERPRINT_PATH.exists()
    ):

        saved_fingerprint = (
            FINGERPRINT_PATH
            .read_text()
            .strip()
        )

        if saved_fingerprint == fingerprint:

            try:

                index = faiss.read_index(
                    str(INDEX_PATH)
                )

                chunks = np.load(
                    CHUNKS_PATH,
                    allow_pickle=True,
                ).tolist()

                return (
                    index,
                    chunks,
                    "FAISS",
                )

            except Exception:
                # Corrupted index → rebuild
                pass


    # --------------------------------------------------------
    # EXTRACT PDF
    # --------------------------------------------------------

    pages = extract_pdf_pages()

    chunks = create_chunks(pages)

    if not chunks:

        raise RuntimeError(
            "No chunks were created from the PDF."
        )

    texts = [
        item["text"]
        for item in chunks
    ]


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    with st.spinner(
        "Creating embeddings for Pakistan cyber law..."
    ):

        embeddings = embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    embeddings = normalize_embeddings(
        embeddings
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if FAISS_AVAILABLE:

        index = create_faiss_index(
            embeddings
        )

        try:

            faiss.write_index(
                index,
                str(INDEX_PATH),
            )

            np.save(
                CHUNKS_PATH,
                np.array(
                    chunks,
                    dtype=object,
                ),
                allow_pickle=True,
            )

            FINGERPRINT_PATH.write_text(
                fingerprint
            )

        except Exception:
            pass

        return (
            index,
            chunks,
            "FAISS",
        )


    # --------------------------------------------------------
    # NUMPY FALLBACK
    # --------------------------------------------------------

    # This prevents the application from crashing if
    # Streamlit Cloud failed to install faiss-cpu.
    #
    # FAISS remains the primary vector database.
    #

    return (
        embeddings,
        chunks,
        "NUMPY FALLBACK",
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    vector_store,
    chunks,
    embedding_model,
    top_k=5,
    backend="FAISS",
):
    """
    Retrieve most relevant legal chunks.
    """

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
    )

    query_embedding = normalize_embeddings(
        query_embedding
    ).astype("float32")


    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    if backend == "FAISS":

        scores, indices = vector_store.search(
            query_embedding,
            min(top_k, len(chunks)),
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            results.append(
                {
                    "page": chunks[index_id]["page"],
                    "text": chunks[index_id]["text"],
                    "score": float(score),
                }
            )

        return results


    # --------------------------------------------------------
    # NUMPY COSINE SEARCH
    # --------------------------------------------------------

    similarities = np.dot(
        vector_store,
        query_embedding[0],
    )

    best_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index_id in best_indices:

        results.append(
            {
                "page": chunks[index_id]["page"],
                "text": chunks[index_id]["text"],
                "score": float(
                    similarities[index_id]
                ),
            }
        )

    return results


# ============================================================
# BUILD LEGAL PROMPT
# ============================================================

def build_legal_prompt(
    question,
    retrieved_documents,
    technical_level,
    response_size,
    language,
    strict_mode,
):
    """
    Create a grounded legal RAG prompt.
    """

    context_parts = []

    for document in retrieved_documents:

        context_parts.append(
            f"""
[PDF PAGE: {document['page']}]
[RELEVANCE SCORE: {document['score']:.4f}]

{document['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )


    if strict_mode:

        grounding_instruction = """
STRICT MODE IS ENABLED.

You MUST use the supplied PDF context as the
primary and controlling source.

Do NOT invent:
- section numbers
- subsections
- penalties
- fines
- imprisonment periods
- authorities
- procedures
- definitions
- legal rights
- legal obligations
- dates
- amendments

If the retrieved context does not contain enough
information to answer the question, explicitly say:

"The provided cyber-law document does not contain
enough information to answer this question reliably."

Do not guess.
"""

    else:

        grounding_instruction = """
Use the supplied PDF as the primary source.

If general legal context is useful, clearly separate
it from information found in the PDF.

Never pretend general knowledge came from the PDF.
"""


    # Response size
    size_instruction = {
        "Short": "Answer concisely in approximately 100-180 words.",
        "Medium": "Give a balanced explanation in approximately 200-350 words.",
        "Detailed": "Give a detailed explanation with headings, relevant sections, application, and important caveats.",
    }

    # Technical level
    level_instruction = {
        "Beginner": """
Explain legal concepts in simple language.
Avoid unnecessary legal terminology.
Use a practical example when useful.
""",

        "Intermediate": """
Use moderate legal and technical terminology.
Explain important terms briefly.
""",

        "Advanced": """
Use detailed legal and technical terminology.
Discuss relevant provisions and distinctions carefully.
""",

        "Legal / Technical": """
Use professional legal and technical terminology.
Identify provisions, subsections, conditions, exceptions,
and penalties only when supported by the retrieved PDF.
""",
    }


    language_instruction = {
        "English": "Answer entirely in English.",
        "Urdu": "Answer primarily in Urdu script.",
        "Roman Urdu": "Answer primarily in Roman Urdu.",
        "English + Urdu": "Explain important concepts in English and Urdu together.",
    }


    prompt = f"""
You are Cyber Law GPT, a retrieval-augmented
Pakistan cyber-law information assistant.

Your job is to answer questions about cyber law
using the supplied Pakistan cyber-law PDF.

{grounding_instruction}

USER TECHNICAL LEVEL:
{technical_level}

{level_instruction[technical_level]}

RESPONSE SIZE:
{response_size}

{size_instruction[response_size]}

LANGUAGE:
{language}

{language_instruction[language]}

IMPORTANT LEGAL RULES:

1. Never fabricate a legal provision.
2. Never fabricate a section number.
3. Never invent a punishment.
4. Never invent a fine.
5. Never claim something is illegal unless the retrieved context supports it.
6. Distinguish between:
   - what the law says
   - what the user's situation appears to be
   - general explanation
7. If multiple provisions appear relevant, explain them separately.
8. If the question asks about a hypothetical situation, explain how the retrieved law may apply without pretending to make a court determination.
9. If the retrieved context is insufficient, say so.
10. Always mention the relevant PDF page when possible.
11. Do not claim to be a lawyer or government authority.
12. End with a short legal-information disclaimer.

ANSWER STRUCTURE WHEN APPROPRIATE:

### Relevant Law
Mention the relevant section/provision if supported.

### Explanation
Explain what it means.

### Application
Explain how it may relate to the user's situation.

### Penalty / Procedure
Only include this if supported by the retrieved PDF.

### Source
Mention the relevant PDF page(s).

### Disclaimer
This response provides general legal information based on the
provided document and is not a substitute for advice from a
qualified Pakistani lawyer.

==================================================
RETRIEVED LEGAL DOCUMENT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    return prompt


# ============================================================
# GROQ
# ============================================================

def ask_groq(
    prompt,
    model_name,
    temperature,
    response_size,
):
    """
    Send prompt to Groq.
    """

    api_key = get_groq_api_key()

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing.\n\n"
            "For Streamlit Cloud:\n"
            "Go to App → Settings → Secrets and add:\n\n"
            "GROQ_API_KEY = \"your_api_key_here\""
        )


    client = Groq(
        api_key=api_key
    )


    max_tokens = {
        "Short": 800,
        "Medium": 1400,
        "Detailed": 2200,
    }.get(
        response_size,
        1400,
    )


    try:

        response = client.chat.completions.create(

            model=model_name,

            messages=[
                {
                    "role": "system",
                    "content": """
You are a careful Pakistan cyber-law
information assistant.

Accuracy is more important than sounding confident.

Never hallucinate legal sections, penalties,
procedures, or legal authorities.
Use retrieved evidence.
""",
                },

                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=temperature,

            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:

        error_message = str(e)

        if "401" in error_message:
            raise RuntimeError(
                "Groq API authentication failed. "
                "Check your GROQ_API_KEY."
            )

        if "429" in error_message:
            raise RuntimeError(
                "Groq rate limit reached. "
                "Please wait and try again."
            )

        raise RuntimeError(
            f"Groq API error: {error_message}"
        )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:
        return

    with st.expander(
        "📚 Retrieved legal sources",
        expanded=False,
    ):

        for number, result in enumerate(
            results,
            start=1,
        ):

            st.markdown(
                f"""
                **Source {number} — PDF Page {result['page']}**

                Similarity: `{result['score']:.4f}`
                """
            )

            st.caption(
                result["text"][:700]
                + (
                    "..."
                    if len(result["text"]) > 700
                    else ""
                )
            )

            if number < len(results):
                st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Cyber Law GPT")

    st.markdown(
        "Configure how the legal answer should be generated."
    )

    st.divider()

    # Technical level
    technical_level = st.selectbox(
        "🎓 Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal / Technical",
        ],
        index=0,
    )


    # Response size
    response_size = st.selectbox(
        "📏 Response size",
        [
            "Short",
            "Medium",
            "Detailed",
        ],
        index=1,
    )


    # Language
    language = st.selectbox(
        "🌐 Answer language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
            "English + Urdu",
        ],
        index=0,
    )


    # Retrieval
    top_k = st.slider(
        "🔎 Retrieved legal passages",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )


    # Temperature
    temperature = st.slider(
        "🎯 Response creativity",
        min_value=0.0,
        max_value=0.7,
        value=0.1,
        step=0.05,
    )


    # Strict mode
    strict_mode = st.toggle(
        "🔒 Strict PDF-only mode",
        value=True,
    )


    # Model
    model_name = st.selectbox(
        "🤖 Groq model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ],
        index=0,
    )


    st.divider()

    st.subheader("📚 Knowledge Base")

    st.caption(
        "Pakistan cyber-law PDF downloaded from "
        "the configured Google Drive source."
    )

    if FAISS_AVAILABLE:

        st.success(
            "FAISS vector search available"
        )

    else:

        st.warning(
            "FAISS is not currently available. "
            "The app will use a temporary NumPy "
            "fallback until faiss-cpu is installed."
        )


    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ Cyber Law GPT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    RAG-powered Pakistan Cyber Law Assistant
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LEGAL NOTICE
# ============================================================

st.markdown(
    """
    <div class="warning-box">

    ⚠️ <b>Legal Information Notice</b><br><br>

    Cyber Law GPT provides information based on the
    configured Pakistan cyber-law document. It is an
    AI research assistant, not a lawyer, court, government
    authority, or substitute for professional legal advice.

    For serious legal matters, verify the current law and
    consult a qualified Pakistani lawyer.

    </div>
    """,
    unsafe_allow_html=True,
)


st.write("")


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

try:

    with st.spinner(
        "📚 Loading cyber-law knowledge base..."
    ):

        vector_store, chunks, backend = (
            build_knowledge_base()
        )

        embedding_model = (
            load_embedding_model()
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Indexed chunks",
            len(chunks),
        )

    with col2:

        st.metric(
            "Search engine",
            backend,
        )

    with col3:

        st.metric(
            "Embedding model",
            "MiniLM",
        )


except Exception as error:

    st.error(
        "❌ Knowledge base initialization failed."
    )

    st.code(
        str(error)
    )

    st.markdown(
        """
        ### Common solution

        Make sure your `requirements.txt` contains:

        ```text
        streamlit
        faiss-cpu
        sentence-transformers
        PyMuPDF
        numpy
        groq
        gdown
        ```

        If deploying on Streamlit Cloud, commit the updated
        `requirements.txt` and reboot/redeploy the application.
        """
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            display_sources(
                message["sources"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask any question about Pakistan cyber law..."
)


if question:

    question = question.strip()

    if not question:
        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        # Retrieval
        with st.spinner(
            "🔎 Searching relevant cyber-law provisions..."
        ):

            try:

                results = retrieve_documents(
                    question=question,
                    vector_store=vector_store,
                    chunks=chunks,
                    embedding_model=embedding_model,
                    top_k=top_k,
                    backend=backend,
                )

            except Exception as e:

                st.error(
                    f"Retrieval error: {e}"
                )

                st.stop()


        # Build prompt
        prompt = build_legal_prompt(
            question=question,
            retrieved_documents=results,
            technical_level=technical_level,
            response_size=response_size,
            language=language,
            strict_mode=strict_mode,
        )


        # Generate answer
        with st.spinner(
            "🤖 Generating grounded legal response..."
        ):

            try:

                answer = ask_groq(
                    prompt=prompt,
                    model_name=model_name,
                    temperature=temperature,
                    response_size=response_size,
                )

            except Exception as e:

                answer = (
                    "⚠️ **Unable to generate the response.**\n\n"
                    f"{e}"
                )


        st.markdown(
            answer
        )


        # Sources
        display_sources(
            results
        )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )
````
