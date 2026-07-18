import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Gemma Chatbot", page_icon="🤖")
st.title("🤖 Gemma 2B Chatbot")

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Question:
{question}
""")

chain = prompt | llm | StrOutputParser()

input_text = st.text_input("Enter your question:")

if st.button("Ask"):
    if input_text.strip():
        try:
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": input_text})

            st.success("Answer")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")