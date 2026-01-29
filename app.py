import streamlit as st
import requests

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 AI PDF RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        files = {"file": f}
        res = requests.post("http://localhost:8000/upload", files=files)
        st.success(res.json()["status"])

st.divider()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something from your PDFs")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[:-1]
    ]

    payload = {
        "question": prompt,
        "history": history
    }

    res = requests.post("http://localhost:8000/ask", json=payload).json()

    answer = res["answer"]
    sources = res["sources"]

    source_text = "\n".join([f"- {s['file']} (page {s['page']})" for s in sources])

    final = answer + "\n\n**Sources:**\n" + source_text

    st.session_state.messages.append({"role": "assistant", "content": final})

    with st.chat_message("assistant"):
        st.markdown(final)


