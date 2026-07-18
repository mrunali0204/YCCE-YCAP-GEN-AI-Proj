# app.py

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------
# Load and Process Data
# -------------------------------
@st.cache_resource
def load_vector_db():

    loader = TextLoader("cppTextData.txt", encoding="utf-8")
    txt_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    documents = text_splitter.split_documents(txt_data)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(documents, embeddings)

    return db


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="C++ Assistant",
    page_icon="💻",
    layout="wide"
)

st.title("💻 C++ Assistant")
st.write("Ask any question related to C++")

db = load_vector_db()

query = st.text_input(
    "Enter your C++ question:"
)

if st.button("Search"):

    if query.strip():

        docs = db.similarity_search(query, k=5)

        st.subheader("Relevant Information")

        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Result {i}")
            st.write(doc.page_content)
            st.divider()

    else:
        st.warning("Please enter a question.")
