# =========================
# IMPORTS
# =========================
import streamlit as st
import re
import os
import time
from dateutil import parser as dateparser
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# =========================
# CONFIG
# =========================
CHAT_FILE = "_chat.txt"
VECTOR_DIR = "vectorstore"
CHUNK_SIZE = 15
BATCH_SIZE = 100  # Process embeddings in batches


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
                "datetime": dateparser.parse(f"{date} {time_}", dayfirst=True),
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
# VECTORSTORE WITH BATCHING
# =========================
def build_vectorstore(chunks, progress_bar=None, status_text=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    total_chunks = len(chunks)
    if total_chunks <= BATCH_SIZE:
        if status_text:
            status_text.text(f"Creating embeddings for {total_chunks} chunks...")
        vs = FAISS.from_documents(chunks, embeddings)
    else:
        vs = None
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            
            if status_text:
                status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            if vs is None:
                vs = FAISS.from_documents(batch, embeddings)
            else:
                vs_batch = FAISS.from_documents(batch, embeddings)
                vs.merge_from(vs_batch)
            
            if progress_bar:
                progress_bar.progress(min(1.0, (i + len(batch)) / total_chunks))
    
    if status_text:
        status_text.text("Saving index to disk...")
    
    vs.save_local(VECTOR_DIR)
    return vs


def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# HELPER FUNCTION
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="WhatsApp RAG", layout="wide")
st.title("WhatsApp Chat RAG + Replay")

# Check API Key
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please create a .env file with your Google API key.")
    st.info("Create a .env file in the same directory as this script with:\nGOOGLE_API_KEY=your_api_key_here")
    st.stop()

tabs = st.tabs(["Index", "Ask", "Replay"])


# =========================
# TAB 1 — INDEX
# =========================
with tabs[0]:
    st.header("Chat Indexing")

    if not os.path.exists(CHAT_FILE):
        st.error(f"{CHAT_FILE} not found")
    else:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            chat_text = f.read()

        messages = parse_whatsapp_chat(chat_text)
        st.success(f"Loaded {len(messages):,} messages")
        
        docs = messages_to_documents(messages)
        chunks = chunk_documents(docs)
        st.info(f"Will create {len(chunks):,} chunks ({CHUNK_SIZE} messages per chunk)")
        
        # Estimate time
        estimated_minutes = len(chunks) * 0.2 / 60  # Rough estimate: 0.2 seconds per chunk
        st.info(f"Estimated time: ~{estimated_minutes:.1f} minutes")

        if st.button("Build / Rebuild Index", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                build_vectorstore(chunks, progress_bar, status_text)
                
                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                
                progress_bar.progress(1.0)
                status_text.empty()
                st.success(f"ector index ready! (Completed in {minutes}m {seconds}s)")
                
            except Exception as e:
                status_text.empty()
                st.error(f"Error building index: {str(e)}")


# =========================
# TAB 2 — QA
# =========================
with tabs[1]:
    st.header("Ask Questions")

    if not os.path.exists(VECTOR_DIR):
        st.warning("Index not found. Please build the index first in the 'Index' tab.")
    else:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_template(
            """You are analyzing a WhatsApp chat history.

Use ONLY the provided chat context to answer.
If the answer is not present, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        # Build RAG chain using LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        query = st.text_input("Ask about the chat", placeholder="e.g., What topics were discussed most?")

        if query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(query)

            st.markdown("### Answer")
            st.write(response)


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
        
        st.info(f"Total messages: {len(messages):,}")

        start = st.number_input(
            "Start index",
            min_value=0,
            max_value=len(messages) - 1,
            value=0
        )

        count = st.slider("Messages to replay", 1, 100, 20)
        speed = st.slider("Speed (seconds per message)", 0.1, 2.0, 0.5)

        if st.button("Play"):
            placeholder = st.empty()
            for idx, m in enumerate(messages[start:start + count]):
                placeholder.markdown(
                    f"""
                    **Message {start + idx + 1}/{len(messages)}**
                    
                    **{m['sender']}**  
                    {m['datetime'].strftime('%d %b %Y, %I:%M:%S %p')}  
                    {m['message']}
                    
                    ---
                    """
                )
                time.sleep(speed)
            
            st.success("Replay complete!")