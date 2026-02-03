# =========================
# IMPORTS
# =========================
import streamlit as st
import re
import os
import time
from datetime import datetime
from dateutil import parser as dateparser

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# =========================
# CONFIG
# =========================
CHAT_FILE = "_chat.txt"
VECTOR_DIR = "vectorstore"
CHUNK_SIZE = 15


# =========================
# REGEX
# =========================
CHAT_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)"
)


# =========================
# PARSER
# =========================
def parse_whatsapp_chat(text):
    messages = []
    buffer = None

    for line in text.split("\n"):
        line = line.strip()
        match = CHAT_PATTERN.match(line)

        if match:
            if buffer:
                messages.append(buffer)

            date, time_, sender, message = match.groups()
            buffer = {
                "datetime": dateparser.parse(f"{date} {time_}",tdayfirst=True),
                "sender": sender,
                "message": message
            }
        else:
            if buffer and line:
                buffer["message"] += " " + line

    if buffer:
        messages.append(buffer)

    return messages


# =========================
# DOCUMENTS
# =========================
def messages_to_documents(messages):
    docs = []
    for m in messages:
        text = f"[{m['datetime']}] {m['sender']}: {m['message']}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "sender": m["sender"],
                    "datetime": m["datetime"].isoformat()
                }
            )
        )
    return docs


def chunk_documents(docs, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(docs), size):
        text = "\n".join(d.page_content for d in docs[i:i+size])
        chunks.append(Document(page_content=text))
    return chunks


# =========================
# VECTORSTORE
# =========================
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTOR_DIR)
    return vs


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="WhatsApp RAG", layout="wide")
st.title("📱 WhatsApp Chat RAG + Replay")

tabs = st.tabs(["Index", "Ask", "Replay"])


# =========================
# TAB 1 — INDEX
# =========================
with tabs[0]:
    st.header("Chat Indexing")

    if not os.path.exists(CHAT_FILE):
        st.error(f"{CHAT_FILE} not found in project directory")
    else:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            chat_text = f.read()

        messages = parse_whatsapp_chat(chat_text)
        st.success(f"Loaded {len(messages)} messages")

        if st.button("Build / Rebuild Index"):
            docs = messages_to_documents(messages)
            chunks = chunk_documents(docs)
            build_vectorstore(chunks)
            st.success("Vector index ready")


# =========================
# TAB 2 — QA
# =========================
with tabs[1]:
    st.header("Ask Questions")

    if not os.path.exists(VECTOR_DIR):
        st.warning("Index not found")
    else:
        vs = load_vectorstore()
        llm = ChatOpenAI(model="gpt-4.1", temperature=0)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vs.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff"
        )

        query = st.text_input("Ask about the chat")

        if query:
            with st.spinner("Thinking..."):
                answer = qa.run(query)
            st.markdown("### Answer")
            st.write(answer)


# =========================
# TAB 3 — REPLAY
# =========================
with tabs[2]:
    st.header("Chat Replay")

    if not os.path.exists(CHAT_FILE):
        st.warning("Chat file not found")
    else:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            chat_text = f.read()

        messages = parse_whatsapp_chat(chat_text)

        start = st.number_input(
            "Start index",
            min_value=0,
            max_value=len(messages)-1,
            value=0
        )

        count = st.slider("Messages", 1, 50, 10)
        speed = st.slider("Speed (seconds)", 0.1, 2.0, 0.5)

        if st.button("Play"):
            placeholder = st.empty()
            for m in messages[start:start+count]:
                placeholder.markdown(
                    f"""
                    **{m['sender']}**  
                    {m['datetime'].strftime('%d %b %Y, %I:%M:%S %p')}  
                    {m['message']}
                    ---
                    """
                )
                time.sleep(speed)
