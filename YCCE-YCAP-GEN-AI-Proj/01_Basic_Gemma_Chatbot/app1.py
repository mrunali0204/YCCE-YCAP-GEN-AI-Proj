from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#creating my prompt
prompt = ChatPromptTemplate.from_messages(
    [
       ("system", "You are a helpful assistant. Please respond to the user input in a concise and informative manner."),
       ("user", "Question: {question}")
    ]
)

#streamlit framework to create a simple web app
st.title("My GPT")
input_text = st.text_input("What que do u have?");

#ollama framework along with gemma2:latest LLM model
llm = Ollama(model = "gemma2:latest")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

#GPT output
if input_text:
    st.write(chain.invoke({"questions" : input_text}))