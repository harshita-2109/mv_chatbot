import os
import warnings
import logging
import base64
from dotenv import load_dotenv  
import streamlit as st
import groq

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# ========== 🖼️ Background Image with Overlay ==========
def add_bg_with_overlay(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                              url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call it with your image path
add_bg_with_overlay("bg.png")  

# ========== 🔐 API Key Handling ==========
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ ERROR: GROQ_API_KEY is missing! Add it to the .env file.")
    st.stop()

# Suppress warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ========== 💬 Chatbot UI ==========
st.title('Ask ManoVaani!')

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# ========== 📄 PDF Vector Store (Cached) ==========
@st.cache_resource
def get_vectorstore():
    pdf_name = "./BMSL.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(
        model_name='all-MiniLM-L12-v2',
        model_kwargs={"device": "cpu"}  
    ),
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
).from_loaders(loaders)

    return index.vectorstore

# ========== 📝 Prompt Input ==========
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything. Answer the following Question: {user_prompt}.
           Start the answer directly. No small talk please."""
    )

    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("⚠️ Failed to load document")
            st.stop()
        
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ========== 🌟 Affirmation and Meditation Features ==========
def generate_affirmation():
    prompt = "Provide a positive affirmation to encourage someone who is feeling stressed or overwhelmed."
    response = groq.Groq(api_key=api_key).chat.completions.create(
        model="Llama-3.3-70b-Specdec",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_meditation_guide():
    prompt = "Provide a 5-minute guided meditation script to help someone relax and reduce stress."
    response = groq.Groq(api_key=api_key).chat.completions.create(
        model="Llama-3.3-70b-Specdec",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

col1, col2 = st.columns(2)

with col1:
    if st.button("Give me a positive Affirmation"):
        affirmation = generate_affirmation()
        st.markdown(f"**Affirmation:** {affirmation}")

with col2:
    if st.button("Give me a guided meditation"):
        meditation_guide = generate_meditation_guide()
        st.markdown(f"**Guided Meditation:** {meditation_guide}")
