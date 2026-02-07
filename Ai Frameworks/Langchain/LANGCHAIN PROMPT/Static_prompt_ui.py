from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import streamlit as st 

load_dotenv()

model= ChatOllama(model='mistral')

st.header('Research tool')

user_input = st.text_input('enter your prompt: ')

if st.button('summarize'):
    result=model.invoke(user_input)
    st.write(result.content)
