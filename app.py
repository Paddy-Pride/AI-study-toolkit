import os
import json
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq

# ---------- Secrets / Env ----------
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------- Page Config ----------
st.set_page_config(
    page_title="Study Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- MTN Pulse Inspired CSS ----------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, "Segoe UI", Roboto, sans-serif;
    }
    .stApp { background-color: #FFFFFF; }

    h1, h2, h3 {
        color: #000000;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 2.2rem;
        border-bottom: 4px solid #FFCC00;
        padding-bottom: 0.4rem;
        display: inline-block;
    }

    .stButton > button {
        background-color: #FFCC00;
        color: #000000;
        border: 2px solid #000000;
        border-radius: 30px;
        font-weight: 700;
        padding: 0.6rem 1.8rem;
        font-size: 0.95rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #000000;
        color: #FFCC00;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        color: #666666;
        padding: 0.8rem 0;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #000000;
        border-bottom: 3px solid #FFCC00;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #CCCCCC;
    }
    .stTextArea textarea:focus {
        border-color: #FFCC00;
        box-shadow: 0 0 0 2px rgba(255, 204, 0, 0.3);
    }

    .streamlit-expanderHeader {
        font-weight: 700;
        background-color: #FFF9E0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    details[open] summary {
        border-left: 5px solid #FFCC00;
    }

    .stAlert { border-radius: 12px; }

    .stCodeBlock pre {
        background-color: #1A1A1A !important;
        color: #FFCC00 !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Prompt Engineer ====================
PE_SYSTEM = """You are a senior prompt engineer. Transform the user's rough request into a structured, high-quality prompt.

Strict rules:
- Output ONLY the optimized prompt. No preamble, no explanation.
- Include these sections: PERSONA, OBJECTIVE, CONTEXT/CONSTRAINTS, OUTPUT FORMAT.
- Do NOT answer the user's original request. Only optimize the prompt."""

def optimize_prompt(user_input: str) -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": PE_SYSTEM},
            {"role": "user", "content": user_input},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content

# ==================== PDF Extraction ====================
def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# ==================== Chunking (no truncation) ====================
def split_into_chunks(text: str, chunk_size: int = 5000, overlap: int = 300) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# ==================== Prompt Builder ====================
def build_prompt(chunk: str, q_type: str, n: int) -> str:
    if q_type == "mcq":
        return f"""Generate {n} multiple-choice questions based on the text below.
Each question must have 4 options, one correct answer index (0-based), and a brief explanation.
Output ONLY a valid JSON array. Each element: {{"question": "...", "options": ["A","B","C","D"], "answer_index": 0, "explanation": "..."}}
No markdown code fences.

Text:
    if q_type == "qa":
        return f"""Generate {n} question-answer pairs based on the text below.
Output ONLY a valid JSON array. Each element: {{"question": "...", "answer": "..."}}
No markdown code fences.

Text:
    return f"""Generate {n} scenario-based questions based on the text below.
Each question should present a real-world scenario and ask how a concept from the text applies.
Output ONLY a valid JSON array. Each element: {{"scenario": "...", "question": "...", "model_answer": "..."}}
No markdown code fences.

Text:

# ==================== Clean JSON ====================
def clean_json(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []

# ==================== Full-document Generation ====================
def generate_questions(text: str, q_type: str, count: int, progress_bar=None) -> list:
    chunks = split_into_chunks(text, chunk_size=5000, overlap=300)
    total_chunks = len(chunks)
    per_chunk = max(1, -(-count // total_chunks))  # ceil division

    all_items = []
    for idx, chunk in enumerate(chunks):
        prompt = build_prompt(chunk, q_type, per_chunk)
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            items = clean_json(resp.choices[0].message.content)
            all_items.extend(items)
        except Exception:
            pass

        if progress_bar is not None:
            progress_bar.progress(min((idx + 1) / total_chunks, 1.0))

        if len(all_items) >= count:
            break

    return all_items[:count]

# ==================== UI ====================
st.title("STUDY TOOLKIT")

tab1, tab2 = st.tabs(["PROMPT ENGINEER", "REVISION COMPANION"])

# ----- Tab 1 -----
with tab1:
    st.subheader("Turn a rough idea into a structured prompt")
    user_input = st.text_area(
        "Describe your task or idea",
        placeholder="e.g., help me write a cover letter for a software internship",
        height=130,
    )
    if st.button("OPTIMIZE PROMPT", key="pe_btn"):
        if not user_input.strip():
            st.warning("Please enter something first.")
        else:
            with st.spinner("Optimizing..."):
                result = optimize_prompt(user_input)
            st.markdown("### Optimized Prompt")
            st.code(result, language="markdown")

# ----- Tab 2 -----
with tab2:
    st.subheader("Upload a PDF and generate study questions")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        @st.cache_data(show_spinner=False)
        def get_text(file_bytes):
            return extract_pdf_text(file_bytes)

        with st.spinner("Extracting text from PDF..."):
            pdf_text = get_text(uploaded_file.getvalue())

        if not pdf_text.strip():
            st.error("No text found in this PDF. It may be a scanned document.")
        else:
            chunks_preview = split_into_chunks(pdf_text, 5000, 300)
            st.success(
                f"Extracted {len(pdf_text)} characters across "
                f"{len(chunks_preview)} chunk(s). All chunks will be processed."
            )

            col1, col2 = st.columns(2)
            with col1:
                q_type = st.selectbox(
                    "Question type",
                    options=["mcq", "qa", "scenario"],
                    format_func=lambda x: {
                        "mcq": "Multiple Choice",
                        "qa": "Question & Answer",
                        "scenario": "Scenario-Based",
                    }[x],
                )
            with col2:
                count = st.slider("Number of questions", 3, 15, 5)

            if st.button("GENERATE QUESTIONS", key="gen_btn"):
                progress = st.progress(0)
                with st.spinner("Generating via Groq..."):
                    items = generate_questions(pdf_text, q_type, count, progress)
                progress.empty()

                if not items:
                    st.error("No questions were generated. Try again or a smaller PDF.")
                else:
                    st.markdown(f"### Generated Questions ({len(items)})")
                    for i, item in enumerate(items, 1):
                        with st.expander(f"Question {i}"):
                            if q_type == "mcq":
                                st.write(f"**{item.get('question', '')}**")
                                for j, opt in enumerate(item.get("options", [])):
                                    marker = "[CORRECT]" if j == item.get("answer_index") else "[ ]"
                                    st.write(f"{marker} {opt}")
                                st.info(f"Explanation: {item.get('explanation', '')}")
                            elif q_type == "qa":
                                st.write(f"**Q:** {item.get('question', '')}")
                                st.write(f"**A:** {item.get('answer', '')}")
                            else:
                                st.write(f"**Scenario:** {item.get('scenario', '')}")
                                st.write(f"**Question:** {item.get('question', '')}")
                                st.info(f"Model Answer: {item.get('model_answer', '')}")
                                
