import streamlit as st
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
from openai import OpenAI
import google.generativeai as genai


st.title("Homework 3: Chatbot")

provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Mistral", "Gemini"], index=0)
model_tier = st.sidebar.selectbox("Model Tier", ["mini", "regular"], index=0)

# map provider + tier to concrete model names
def resolve_model(p: str, tier: str) -> str:
    if p == "OpenAI":
        return "gpt-4o-mini" if tier == "mini" else "gpt-4o"
    if p == "Mistral":
        return "mistral-small-latest" if tier == "mini" else "mistral-medium-latest"
    if p == "Gemini":
        return "gemini-2.5-flash-lite" if tier == "mini" else "gemini-2.5-flash"

model_to_use = resolve_model(provider, model_tier)

# memory policy
memory_mode = st.sidebar.selectbox(
    "Memory Mode",
    ["Buffer: 6 questions", "Buffer: 2,000 tokens", "Conversation summary"],
    index=0,
)

# two optional URLs to prime context
url1 = st.sidebar.text_input("Context URL 1 (optional)")
url2 = st.sidebar.text_input("Context URL 2 (optional)")
fetch_btn = st.sidebar.button("Fetch URLs")

# ---------------- Secrets ----------------
def get_api_key(p: str) -> str | None:
    try:
        return st.secrets[f"{p.upper()}_API_KEY"]
    except Exception:
        return None

api_key = get_api_key(provider)
if not api_key:
    st.error(f"Missing API key for {provider}. Add it to secrets as either [{provider.lower()}].api_key or {provider.upper()}_API_KEY.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "info_stage" not in st.session_state:
    st.session_state.info_stage = 0
if "context_blobs" not in st.session_state:
    st.session_state.context_blobs = []  # fetched URL texts
if "summary" not in st.session_state:
    st.session_state.summary = ""  # running conversation summary, if used

def fetch_url_text(url: str, max_chars: int = 20000) -> str | None:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(separator="\n")
        return text[:max_chars]
    except requests.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")

if fetch_btn:
    blobs = []
    for u in (url1, url2):
        blob = fetch_url_text(u)
        if blob:
            blobs.append(blob)
    st.session_state.context_blobs = blobs
    st.success(f"Loaded {len(st.session_state.context_blobs)} context source(s).")

def approx_token_count(s: str) -> int:
    return max(1, len(s) // 3)

def build_context_messages() -> list[dict]:
    msgs = []
    if st.session_state.context_blobs:
        joined = "\n\n---\n\n".join(st.session_state.context_blobs)
        msgs.append({
            "role": "system",
            "content": "Reference context (use if relevant; do not invent facts):\n" + joined
        })
    if st.session_state.summary and memory_mode == "Conversation summary":
        msgs.append({"role": "system", "content": "Conversation summary so far:\n" + st.session_state.summary})
    return msgs

def trim_messages_by_policy(history: list[dict]) -> list[dict]:
    if memory_mode == "Buffer: 6 questions":
        user_idxs = [i for i, m in enumerate(history) if m["role"] == "user"]
        keep_user_idxs = set(user_idxs[-6:]) if len(user_idxs) > 6 else set(user_idxs)
        keep_idxs = set()
        for i, m in enumerate(history):
            if i in keep_user_idxs:
                keep_idxs.add(i)
                if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                    keep_idxs.add(i + 1)
        keep_idxs.add(0)
        trimmed = [m for i, m in enumerate(history) if i in sorted(keep_idxs)]
        return trimmed

    if memory_mode == "Buffer: 2,000 tokens":
        budget = 2000
        kept = []
        total = 0
        for m in reversed(history):
            t = approx_token_count(m["content"])
            if total + t > budget and kept:
                break
            kept.append(m)
            total += t
        return list(reversed(kept))
    return history[-6:]

def update_running_summary_if_needed():
    if memory_mode != "Conversation summary":
        return
    window = 12
    if len(st.session_state.messages) <= window:
        return
    older = st.session_state.messages[:-6]
    older_text = ""
    for m in older:
        role = m["role"]
        older_text += f"{role.upper()}: {m['content']}\n"
    summary_prompt = (
        "Summarise the prior conversation in 5 bullets capturing user goals, "
        "constraints, decisions, and unresolved items. No fluff."
    )

    summary_content = call_llm(
        provider=provider,
        model=model_to_use,
        api_key=api_key,
        messages=[
            {"role": "system", "content": "You are a precise note-taker."},
            {"role": "user", "content": summary_prompt + "\n\n" + older_text},
        ],
    )
    if summary_content:
        st.session_state.summary = summary_content

# ---------------- Provider call wrappers ----------------
def call_llm(provider: str, model: str, api_key: str, messages: list[dict]) -> str:
    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    if provider == "Mistral":
        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    if provider == "Gemini":
        genai.configure(api_key=api_key)
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}: {m['content']}\n"
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(prompt)
        return getattr(resp, "text", "") or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")

    raise ValueError(f"Unknown provider: {provider}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    if st.session_state.info_stage == 1 and user_input.strip().lower() == "yes":
        prompt = "Please expand with more detail. Build on the last answer, do not repeat the same information. If you don't know the answer, just say so"
        st.session_state.info_stage = 2
    elif st.session_state.info_stage == 2 and user_input.strip().lower() == "yes":
        prompt = "Please expand with even more detail. Build on the last 2 answers, do not repeat the same information"
        st.session_state.info_stage = 0
    elif st.session_state.info_stage in [1, 2] and user_input.strip().lower() != "yes":
        prompt = "What can I help you with?"
        st.session_state.info_stage = 0
    else:
        prompt = user_input
        st.session_state.info_stage = 1

    base_history = st.session_state.messages + [{"role": "user", "content": prompt}]
    trimmed_history = trim_messages_by_policy(base_history)
    final_messages = build_context_messages() + trimmed_history

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {provider} · {model_to_use}..."):
            reply = call_llm(provider, model_to_use, api_key, final_messages)
            st.markdown(reply if reply else "_No response_")


    st.session_state.messages.append({"role": "assistant", "content": reply})

    update_running_summary_if_needed()

    if st.session_state.info_stage in [1, 2]:
        follow_up = "DO YOU WANT MORE INFO?"
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        with st.chat_message("assistant"):
            st.markdown(follow_up)