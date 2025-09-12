import streamlit as st
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
from openai import OpenAI
import google.generativeai as genai

st.title("HW2: Text Summariser")

# --- Sidebar: provider, model tier, summary type, language ---
provider = st.sidebar.selectbox(
    "Model Provider",
    ["OpenAI", "Mistral", "Gemini"],  
    index=0
)

use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)

summary_choice = st.sidebar.radio(
    "Summary Type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ],
    index=0,
)

output_language = st.sidebar.selectbox(
    "Output Language",
    ["English", "Spanish", "German"],
    index=0,
)

# --- Model mapping per provider ---
def resolve_model(provider: str, advanced: bool) -> str:
    if provider == "OpenAI":
        return "gpt-4o" if advanced else "gpt-4o-mini"
    if provider == "Mistral":
        return "mistral-medium-latest" if advanced else "mistral-small-latest"
    if provider == "Gemini":
        return "gemini-2.5-flash" if advanced else "gemini-2.5-flash-lite"
    raise ValueError(f"Unknown provider: {provider}")

model_name = resolve_model(provider, use_advanced)

# --- Secrets loading (vendor-prefixed) ---
def get_api_key(provider: str) -> str | None:
    section = provider.lower()  
    try:
        return st.secrets[section]["api_key"]
    except Exception:
        return None

api_key = get_api_key(provider)
if not api_key:
    st.error(
        f"API key not found for {provider}. "
        "Add vendor-prefixed keys in secrets (see instructions below)."
    )
    st.stop()

# --- URL input only ---
url_input = st.text_input("Enter a URL to summarise")
go = st.button("Summarise")

def read_url_content(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text(separator="\n")
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def build_instruction(choice: str, language: str) -> str:
    lang_sentence = f" Write the summary in {language}. Use only information from the content."
    if "100 words" in choice:
        return (
            "Summarise the content in exactly 100 words as a single paragraph. "
            "No title or bullets." + lang_sentence
        )
    if "2 connecting paragraphs" in choice:
        return (
            "Summarise the content in two connected paragraphs. "
            "Paragraph 1: core thesis and key evidence. "
            "Paragraph 2: implications, limitations, or next steps linking back to paragraph 1. "
            "No headings or bullets." + lang_sentence
        )
    return (
        "Summarise the content in exactly five bullet points. "
        "Each point must be 25 words or fewer, factual, and non-overlapping. "
        "No introduction or conclusion." + lang_sentence
    )

def make_prompt(document_text: str, instruction: str) -> str:
    # Single-string prompt that works across providers
    return (
        "You are a careful, concise summariser. "
        "Follow the requested format exactly, do not invent facts, and obey the output language.\n\n"
        f"{instruction}\n\n--- CONTENT START ---\n{document_text}\n--- CONTENT END ---"
    )

# --- Provider-specific inference wrappers (non-streaming for simplicity) ---
def summarize_with_openai(prompt: str, model: str, key: str) -> str:
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, concise summariser."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content

def summarize_with_mistral(prompt: str, model: str, key: str) -> str:
    client = Mistral(api_key=key)
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content

def summarize_with_gemini(prompt: str, model: str, key: str) -> str:
    genai.configure(api_key=key)
    gmodel = genai.GenerativeModel(model)
    resp = gmodel.generate_content(prompt)
    return getattr(resp, "text", "") or (resp.candidates[0].content.parts[0].text if resp.candidates else "")

def run_summary(provider: str, model: str, key: str, prompt: str) -> str:
    if provider == "OpenAI":
        return summarize_with_openai(prompt, model, key)
    if provider == "Mistral":
        return summarize_with_mistral(prompt, model, key)
    if provider == "Gemini":
        return summarize_with_gemini(prompt, model, key)
    raise ValueError(f"Unknown provider: {provider}")

if go and url_input:
    doc = read_url_content(url_input)
    if doc:
        doc = doc[:100000]  # simple safeguard
        instruction = build_instruction(summary_choice, output_language)
        prompt = make_prompt(doc, instruction)
        with st.spinner(f"Running {provider} · {model_name}..."):
            out = run_summary(provider, model_name, api_key, prompt)
        st.markdown(out if out else "_No text returned._")
