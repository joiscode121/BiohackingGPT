__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import difflib
import logging
import time
from simple_eval import SimpleEval

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(openai_api_key=api_key)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
try:
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = SimpleEval()
except Exception as e:
    logger.error(f"Failed to initialize evaluator: {e}")
    st.session_state.evaluator = None

st.title("🧬 BiohackingGPT: Your Biohacking Protocol Assistant")

# Sidebar and UI code remains unchanged...
# [Your existing sidebar and main UI code here]

# Cache vector store initialization
@st.cache_resource
def initialize_vector_store():
    try:
        with st.spinner("Loading biohacking protocols..."):
            protocol_texts = []
            protocol_dir = os.path.join(os.path.dirname(__file__), "protocols")
            if not os.path.exists(protocol_dir):
                os.makedirs(protocol_dir)
                logger.info(f"Created protocols directory at {protocol_dir}")

            for filename in os.listdir(protocol_dir):
                if filename.endswith(".md"):
                    with open(os.path.join(protocol_dir, filename), 'r', encoding='utf-8') as f:
                        protocol_texts.append(f.read())

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            chunks = []
            for text in protocol_texts:
                chunks.extend(text_splitter.split_text(text))

            # Create embeddings and vector store
            api_key = os.getenv("OPENAI_API_KEY")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
            return vector_store
    except Exception as e:
        logger.error(f"Failed to load protocols. Error: {str(e)}")
        st.error(f"Failed to load protocols. Error: {str(e)}")
        return None

# Initialize vector store if not already done
if st.session_state.vector_store is None:
    st.session_state.vector_store = initialize_vector_store()
