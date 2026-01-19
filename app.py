import os
import streamlit as st
from dotenv import load_dotenv
from rag import RAGIndex
from utils import extract_text_from_pdf, chunk_text
from agents import planner_agent, answer_agent, critic_agent

load_dotenv()

st.set_page_config(page_title="Agentic RAG + Claude", layout="wide")
st.title("Agentic RAG Assistant (Claude)")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Missing ANTHROPIC_API_KEY. Set it as an environment variable or in .env.")
    st.stop()

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGIndex()
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = 0

st.sidebar.header("Load Documents")
uploaded = st.sidebar.file_uploader("Upload PDFs or TXT", type=["pdf", "txt"], accept_multiple_files=True)
k = st.sidebar.slider("Top-K retrieved chunks", 3, 10, 5)

if uploaded and st.sidebar.button("Ingest"):
    total_chunks = 0
    for f in uploaded:
        if f.type == "application/pdf":
            text = extract_text_from_pdf(f)
        else:
            text = f.read().decode("utf-8", errors="ignore")

        chunks = chunk_text(text)
        meta = [{"doc": f.name, "chunk": i} for i in range(len(chunks))]
        st.session_state.rag.add_chunks(chunks, meta)
        total_chunks += len(chunks)

    st.session_state.loaded_docs += len(uploaded)
    st.sidebar.success(f"Ingested {len(uploaded)} docs, {total_chunks} chunks.")

st.divider()

st.subheader("Ask a question")
q = st.text_input("Question", placeholder="e.g., Summarize key risks and controls described in the uploaded policy docs.")

if st.button("Run") and q:
    if st.session_state.rag.index is None:
        st.warning("Please ingest at least one document first.")
        st.stop()

    with st.spinner("Planner agent thinking..."):
        plan = planner_agent(q)

    with st.spinner("Retrieving context..."):
        retrieved = st.session_state.rag.search(q, k=k)

    with st.spinner("Answer agent generating response..."):
        answer = answer_agent(q, retrieved)

    with st.spinner("Critic agent reviewing..."):
        critique = critic_agent(q, answer)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Plan")
        st.write(plan)

        st.markdown("### Answer")
        st.write(answer)

    with col2:
        st.markdown("### Critic Check")
        st.write(critique)

        st.markdown("### Retrieved Sources")
        for i, r in enumerate(retrieved, start=1):
            st.markdown(f"**S{i}** — score: {r['score']:.3f} | {r['meta']}")
            st.caption(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

