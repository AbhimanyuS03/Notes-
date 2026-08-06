# =============STEP 1: LOAD MODULES===============
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import numpy
import time
from PIL import Image
from dotenv import load_dotenv
from getpass import getpass
import os

#====================STEP 2 API KEYS======================
st.set_page_config(page_title = "Chat-With-PDF",
              layout = "wide")


st.sidebar.title("SET API CONFIG")
st.title("Flashcard by Notes PDF")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY",type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if GOOGLE_API_KEY:
  st.sidebar.success("API key Loaded!!")
else:
  st.sidebar.info("Give API key")



#==============STEP 4: MODEL============
model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash',
    google_api_key = GOOGLE_API_KEY
)
response = model.invoke("hello buddy!")
response.content[-1]['text']


# ====================== STEP 5 : PROMPT TEMPLATES ======================

flashcard_prompt = """
You are StudyGen AI, an Intelligent Learning Assistant.
Generate well-structured study notes in simple language.
Include:
1. Introduction
2. Key Concepts
3. Definitions
All things in brief not full explanation in flash card format
"""


# ====================== STEP 6 : LOAD PDF ======================

from google.colab import files
uploaded = files.upload()
loader = PyPDFLoader(list(uploaded.keys())[0])

documents = loader.load()
print(f"Total Pages : {len(documents)}")

print(documents[0].page_content)



# ====================== STEP 8 : SPLIT PDF INTO CHUNKS ======================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks : {len(chunks)}")


# ====================== STEP 9 : CREATE EMBEDDINGS ======================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded successfully!")



# ====================== STEP 10 : CREATE VECTOR DATABASE ======================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)
print("Vector Store Built Successfully!")



# ====================== STEP 11 : CREATE RETRIEVER ======================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever Created Successfully!")



# ====================== STEP 12 : IMPORT LCEL COMPONENTS ======================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# ====================== STEP 13 : CREATE LCEL RAG CHAIN ======================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are StudyGen AI, an Intelligent Learning Assistant.
Answer the user's question ONLY using the context below.
If the answer is not present in the context, reply:
"I ccan't find this info in the uploaded study material."
Context:
{context}
Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG Chain Created Successfully!")



# ====================== STEP 14 : TEST RAG CHAIN ======================

question = "Summarize the uploaded document."
ans = rag_chain.invoke(question)

print(ans)



# ====================== STEP 15 : NOTES GENERATOR ======================
def generate_notes():
    prompt = """
Generate well-structured study notes from the uploaded study material.
Include:
1. Introduction
2. Key Concepts
3. Important Points
Use Markdown headings
Use bullet points
Highlight keywords
All things in brief not full explanation in flash card format
"""
    return rag_chain.invoke(prompt)

