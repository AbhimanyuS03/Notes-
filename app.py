import os
import json
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Flashcard Generator", layout="wide")
st.title("📚 Flashcard Generator from Notes (RAG)")

HERO_BANNER = """
<div style="display:flex; justify-content:center; margin: 10px 0 30px 0;">
  <svg width="620" height="150" viewBox="0 0 620 150" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(40,25) rotate(-8)">
      <rect width="140" height="95" rx="14" fill="#FF4B6E" opacity="0.9"/>
    </g>
    <g transform="translate(70,15) rotate(-2)">
      <rect width="140" height="95" rx="14" fill="#4B7BFF" opacity="0.9"/>
    </g>
    <g transform="translate(100,10)">
      <rect width="150" height="100" rx="14" fill="#FFFFFF" stroke="#22C55E" stroke-width="4"/>
      <line x1="20" y1="35" x2="130" y2="35" stroke="#22C55E" stroke-width="4" stroke-linecap="round"/>
      <line x1="20" y1="55" x2="110" y2="55" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
      <line x1="20" y1="75" x2="90" y2="75" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
    </g>
    <g transform="translate(300,45)">
      <circle cx="0" cy="0" r="6" fill="#F59E0B"/>
      <circle cx="25" cy="-15" r="4" fill="#F59E0B"/>
      <circle cx="30" cy="15" r="3" fill="#F59E0B"/>
    </g>
    <text x="350" y="60" font-family="Arial, sans-serif" font-size="26" font-weight="bold" fill="#1E293B">
      Turn your notes into
    </text>
    <text x="350" y="95" font-family="Arial, sans-serif" font-size="26" font-weight="bold" fill="#4B7BFF">
      instant flashcards
    </text>
  </svg>
</div>
"""
st.markdown(HERO_BANNER, unsafe_allow_html=True)

# ============ SIDEBAR: API KEY ============
st.sidebar.title("Setup")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.sidebar.success("API key loaded ✅")
else:
    st.sidebar.info("Enter your Google API key to continue")

# ============ SIDEBAR: PDF UPLOAD ============
uploaded_file = st.sidebar.file_uploader("Upload your notes PDF", type=["pdf"])

if not uploaded_file:
    st.info("👈 Upload a PDF of your notes from the sidebar to get started.")
    st.stop()

if not GOOGLE_API_KEY:
    st.warning("👈 Enter your Google API key in the sidebar to continue.")
    st.stop()

# Save uploaded file to disk
SAVE_DIR = "pdf_files"
os.makedirs(SAVE_DIR, exist_ok=True)
file_path = os.path.join(SAVE_DIR, uploaded_file.name)
with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# A key that changes whenever a *new* file is uploaded, used to invalidate caches correctly
file_key = f"{uploaded_file.name}_{uploaded_file.size}"

# ============ LOAD + SPLIT (cached per file) ============
@st.cache_data(show_spinner="Reading and splitting PDF...")
def load_and_split(path: str, _cache_key: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(docs)

chunks = load_and_split(file_path, file_key)
st.sidebar.write(f"📄 {len(chunks)} chunks extracted from your PDF")

# ============ EMBEDDINGS + VECTORSTORE (cached resources) ============
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Building vector index...")
def build_vectorstore(_chunks, _cache_key: str):
    embeddings = load_embeddings()
    return FAISS.from_documents(_chunks, embeddings)

vectorstore = build_vectorstore(chunks, file_key)

# ============ LLM + PROMPT ============
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

FLASHCARD_PROMPT = ChatPromptTemplate.from_template("""
You are an expert study-guide creator. Based ONLY on the context below, generate {num_cards} flashcards
in simple question-and-answer format to help a student revise this material.

Rules:
- Each flashcard must have a clear, specific "question" and a concise "answer".
- Do not invent facts that are not present in the context.
- Return ONLY a valid JSON array, with no markdown code fences and no extra commentary. Format:
[{{"question": "...", "answer": "..."}}, ...]

Context:
{context}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def parse_flashcards(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        cards = json.loads(text)
        return [c for c in cards if isinstance(c, dict) and "question" in c and "answer" in c]
    except json.JSONDecodeError:
        return []

def generate_flashcards(context_text: str, num_cards: int):
    chain = FLASHCARD_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke({"context": context_text, "num_cards": num_cards})
    except Exception as e:
        # Surface the real error instead of Streamlit Cloud's redacted message
        st.error(f"Gemini API call failed: {e}")
        return []
    return parse_flashcards(raw)

# ============ SESSION STATE ============
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# ============ MODE SELECTION ============
st.subheader("Generate Flashcards")
mode = st.radio(
    "How do you want to generate flashcards?",
    ["From the whole PDF", "From a specific topic"],
    horizontal=True,
)
num_cards = st.slider("Number of flashcards", min_value=3, max_value=20, value=8)

if mode == "From the whole PDF":
    if st.button("Generate Flashcards", type="primary"):
        with st.spinner("Generating flashcards from your notes..."):
            # Sample chunks spread across the document so cards aren't all from page 1
            step = max(1, len(chunks) // num_cards)
            sampled = chunks[::step][: max(5, num_cards)]
            context_text = format_docs(sampled)
            cards = generate_flashcards(context_text, num_cards)
            if cards:
                st.session_state.flashcards = cards
            else:
                st.error("Couldn't parse flashcards from the model's response. Please try again.")

else:
    topic = st.text_input("Enter a topic from your notes (e.g. 'Newton's laws')")
    k_value = st.slider("How many chunks to retrieve as context", 1, 10, 4)
    if st.button("Generate Flashcards", type="primary") and topic:
        with st.spinner("Retrieving relevant content and generating flashcards..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
            docs = retriever.invoke(topic)
            context_text = format_docs(docs)
            cards = generate_flashcards(context_text, num_cards)
            if cards:
                st.session_state.flashcards = cards
            else:
                st.error("Couldn't parse flashcards from the model's response. Please try again.")

# ============ DISPLAY FLASHCARDS ============
if st.session_state.flashcards:
    st.subheader(f"Your Flashcards ({len(st.session_state.flashcards)})")
    for i, card in enumerate(st.session_state.flashcards, start=1):
        with st.expander(f"Card {i}: {card['question']}"):
            st.write(card["answer"])

    # ============ CSV EXPORT ============
    df = pd.DataFrame(st.session_state.flashcards)[["question", "answer"]]
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Flashcards as CSV",
        data=csv_data,
        file_name="flashcards.csv",
        mime="text/csv",
    )
