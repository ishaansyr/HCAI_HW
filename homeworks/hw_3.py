import streamlit as st
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
from openai import OpenAI
import google.generativeai as genai

st.title("Homework 3: Chatbot")
provider = st.sidebar.selectbox("LLM Vendor", ["OpenAI", "Mistral", "Gemini"], index=0)
model_tier = st.sidebar.selectbox("Model Size", ["mini", "regular"], index=0)

def resolve_model(p: str, tier: str) -> str:
    if p == "OpenAI":
        return "gpt-5-nano" if tier == "mini" else "gpt-4o"
    if p == "Mistral":
        return "mistral-small-latest" if tier == "mini" else "mistral-medium-latest"
    if p == "Gemini":
        return "gemini-2.5-flash-lite" if tier == "mini" else "gemini-2.5-flash"
    raise ValueError(f"Unknown provider: {p}")

model_to_use = resolve_model(provider, model_tier)

memory_mode = st.sidebar.selectbox(
    "Memory Mode",
    ["Buffer: 6 questions", "Buffer: 2,000 tokens", "Conversation summary"],
    index=0,
)

url1 = st.sidebar.text_input("Context URL 1")
url2 = st.sidebar.text_input("Context URL 2")
fetch_btn = st.sidebar.button("Fetch URL(s)")

def get_api_key(p: str) -> str | None:
    try:
        return st.secrets[f"{p.upper()}_API_KEY"]
    except Exception:
        return None

api_key = get_api_key(provider)
if not api_key:
    st.error(f"Missing API key for {provider}. Add {provider.upper()}_API_KEY to secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "context_blobs" not in st.session_state:
    st.session_state.context_blobs = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

def fetch_url_text(url: str, max_chars: int = 20000) -> str | None:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=10)  # headers optional
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(separator="\n")
        return text[:max_chars]
    except requests.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
        return None

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
        keep_idxs.add(0)  # keep initial greeting
        return [m for i, m in enumerate(history) if i in sorted(keep_idxs)]

    if memory_mode == "Buffer: 2,000 tokens":
        budget = 2000
        kept, total = [], 0
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
        older_text += f"{m['role'].upper()}: {m['content']}\n"
    summary_prompt = (
        "Summarise the prior conversation in 5 concise bullets capturing user goals, "
        "constraints, decisions, and unresolved items. No fluff."
    )
    summary_content = consume_stream(
        call_llm_stream(
            provider=provider,
            model=model_to_use,
            api_key=api_key,
            messages=[
                {"role": "system", "content": "You are a precise note-taker."},
                {"role": "user", "content": summary_prompt + "\n\n" + older_text},
            ],
        )
    )
    if summary_content:
        st.session_state.summary = summary_content

def call_llm_stream(provider: str, model: str, api_key: str, messages: list[dict], debug: bool = False):
    """
    Yield incremental text chunks from the selected provider.
    Set debug=True to print raw streaming events in the app.
    """

    if provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for event in stream:
            if debug:
                st.write(event)  # raw event from API
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content
        return

    if provider == "Mistral":
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        with client.chat.stream(
            model=model,
            messages=messages,
            temperature=0.7,
        ) as stream:
            for event in stream:
                if debug:
                    st.write(event)
                text = getattr(event, "delta", None)
                if not text and hasattr(event, "data"):
                    text = getattr(event.data, "delta", None) or getattr(event.data, "content", None)
                if text:
                    yield text
        return

    if provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Convert chat-style messages to plain prompt
        prompt = ""
        for m in messages:
            role = m.get("role", "user").upper()
            prompt += f"{role}: {m.get('content','')}\n"
        gmodel = genai.GenerativeModel(model)
        response = gmodel.generate_content(prompt, stream=True)
        for chunk in response:
            if debug:
                st.write(chunk)
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
            else:
                try:
                    yield chunk.candidates[0].content.parts[0].text
                except Exception:
                    pass
        return

    raise ValueError(f"Unknown provider: {provider}")


def consume_stream(generator) -> str:
    out = ""
    for delta in generator:
        out += delta
    return out

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    prompt = user_input  
    base_history = st.session_state.messages + [{"role": "user", "content": prompt}]
    trimmed_history = trim_messages_by_policy(base_history)
    final_messages = build_context_messages() + trimmed_history

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {provider} · {model_to_use}..."):
            placeholder = st.empty()
            reply = ""
            for delta in call_llm_stream(provider, model_to_use, api_key, final_messages):
                reply += delta
                placeholder.markdown(reply if reply else "_No response_")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    update_running_summary_if_needed()
