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

# Add evaluation metrics to sidebar
with st.sidebar:
    st.markdown("## 📊 Evaluation Metrics")
    
    st.markdown("""
   

    **Metrics update automatically after pressing Enter on your next question**
    
    ---
    **About Metrics:**
    - Total Interactions: Number of Q&A exchanges
    - Avg. Latency: Average response time in seconds
    - Avg. Response Length: Average number of words in responses
    """)
    
    # Reset metrics button
    if st.button("🔄 Reset Metrics", type="primary", use_container_width=True):
        if st.session_state.evaluator and st.session_state.evaluator.reset_metrics():
            st.success("Metrics reset!")
        else:
            st.error("Failed to reset metrics")
    
    st.divider()
    
    # Auto-refresh metrics
    if st.session_state.evaluator:
        metrics = st.session_state.evaluator.get_metrics_summary()
        if metrics:
            st.metric("Total Interactions", metrics["total_interactions"])
            st.metric("Avg. Latency", f"{metrics['average_latency']}s")
            st.metric("Avg. Response Length", f"{metrics['average_response_length']} words")
        else:
            st.info("No metrics available yet")
    else:
        st.error("Evaluation system not initialized")

st.markdown("""
## 📚 About This Assistant

This AI assistant helps you understand and implement biohacking protocols for optimal health and performance. It can answer questions about:

- Exercise and fitness optimization
- Nutrition and supplementation
- Sleep optimization
- Stress management
- Performance enhancement
- Recovery and stress management


""")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Main chat interface
st.header("💬 Ask Your Biohacking Question")
st.warning("⚠️ Important Note: This AI may generate incorrect or nonsensical responses, especially for unclear or ambiguous questions. Always verify information with reliable sources.")

# Question input with typo handling
question = st.chat_input("Ask about biohacking protocols and optimization...")

if question:
    question_lower = question.lower().strip()
    start_time = time.time()
    
    # Check for potential typos
    biohacking_topics = ["exercise", "nutrition", "stress", "recovery", "performance", "daily routines", "habits"]
    if not any(topic in question_lower for topic in biohacking_topics):
        closest_matches = difflib.get_close_matches(question_lower, biohacking_topics, n=1, cutoff=0.6)
        if closest_matches:
            st.warning(f"Did you mean to ask about {closest_matches[0]}?")
            st.stop()

    # Display user message
    with st.chat_message("user"):
        st.write(question)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Create retrieval chain
    if st.session_state.vector_store:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
        
        prompt_template = """You are a knowledgeable biohacking assistant. Your goal is to help users optimize their health and performance through evidence-based protocols and interventions.

        When answering biohacking questions:
        1. Provide specific, evidence-based advice with practical examples (e.g., timing, dosage, duration).
        2. Include safety considerations, potential contraindications, and the importance of consulting a healthcare professional.
        3. Be concise but thorough, and invite follow-up questions to deepen the conversation.
        4. If you don't know the answer, admit it clearly and suggest related biohacking topics the user might explore.

        Context: {context}

        Question: {question}

        Answer: """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Get answer with error handling
        with st.chat_message("assistant"):
            with st.spinner("Analyzing protocols..."):
                try:
                    response = qa.invoke(question)
                    answer = response["result"]
                    latency = time.time() - start_time
                    
                    # Log to evaluator
                    if st.session_state.evaluator:
                        st.session_state.evaluator.log_interaction(
                            question=question,
                            response=answer,
                            latency=latency
                        )
                    
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    st.error(f"An error occurred: {str(e)}. Please try again or ask a different question.")
    else:
        st.error("Vector store not initialized. Please check the logs for errors.")


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

            # Check if protocol_texts is empty
            if not protocol_texts:
                logger.warning("No protocol files found in 'protocols' directory.")
                st.warning("No biohacking protocols found. Please add .md files to the 'protocols' directory.")
                return None  # Return None to indicate failure gracefully

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            chunks = []
            for text in protocol_texts:
                split_chunks = text_splitter.split_text(text)
                if split_chunks:  # Ensure no None or empty results
                    chunks.extend(split_chunks)

            # Ensure chunks isn’t empty
            if not chunks:
                logger.error("No valid chunks created from protocol texts.")
                st.error("Failed to process protocol texts into chunks.")
                return None

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
