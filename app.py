import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()



GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state.embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Virat_Kohli")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents , st.session_state.embeddings)

st.title("CHATGROQ DEMO")
llm = ChatGroq(groq_api_key=GROQ_API_KEY , model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
    Answer the question based on provided context only
    Please provide the best accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
""")

document_chain = create_stuff_documents_chain(llm=llm , prompt=prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever , document_chain)


prompt = st.text_input("Input your prompt here")


if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response time: " , time.process_time() - start)
    st.write(response['answer'])