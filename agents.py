import os
from dotenv import load_dotenv
from anthropic import Anthropic
import streamlit as st

load_dotenv()

MODEL = "claude-sonnet-4-5-20250929"

def get_client() -> Anthropic:
   api_key = st.secrets.get(
    "ANTHROPIC_API_KEY",
    os.getenv("ANTHROPIC_API_KEY"))
   if not api_key:
      raise RuntimeError("ANTHROPIC_API_KEY not found. Check your .env or environment variables.")
   return Anthropic(api_key=api_key)

def claude(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    client = get_client()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def planner_agent(user_question: str) -> str:
    system = "You are a programmatic planner. Output a short numbered plan."
    prompt = f"""
Create a concise plan (3-6 steps) to answer this question using retrieved documents.
Question: {user_question}
"""
    return claude(prompt, system=system, max_tokens=250)

def answer_agent(user_question: str, retrieved: list) -> str:
    system = (
        "You answer questions using only the provided context. "
        "Cite sources as [S1], [S2] etc. If not in context, say you don't know."
    )
    sources = []
    for i, r in enumerate(retrieved, start=1):
        tag = f"S{i}"
        sources.append(f"[{tag}] {r['text']}\n(Source: {r['meta']})")

    context = "\n\n".join(sources)

    prompt = f"""
Question: {user_question}

Context snippets:
{context}

Instructions:
- Use only the context snippets.
- Provide a direct answer in 5-10 sentences.
- Add citations inline like [S1].
- End with a short bullet list: "Supporting Evidence" listing each citation and what it supports.
"""
    return claude(prompt, system=system, max_tokens=900)

def critic_agent(user_question: str, answer: str) -> str:
    system = (
        "You are a strict QA reviewer for enterprise settings. "
        "Check for unsupported claims, missing citations, and risky statements."
    )
    prompt = f"""
Review the answer for:
1) Missing citations or claims not supported by sources
2) Overconfident language
3) Compliance/financial risk language (promises, guarantees, advice)

Question: {user_question}

Answer:
{answer}

Output:
- 'Verdict' (Pass / Needs work)
- 3-6 bullet issues (if any)
- Suggested rewrite guidance (2-4 bullets)
"""
    return claude(prompt, system=system, max_tokens=350)

