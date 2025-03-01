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
from simple_eval import SimpleEval  # Replace with your actual SimpleEval implementation if needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in environment variables.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(openai_api_key=api_key)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluator' not in st.session_state:
    try:
        st.session_state.evaluator = SimpleEval()
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        st.session_state.evaluator = None

st.title("🧬 BiohackingGPT: Your Biohacking Protocol Assistant")

# Sidebar metrics
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
    if st.button("🔄 Reset Metrics", type="primary", use_container_width=True):
        if st.session_state.evaluator and st.session_state.evaluator.reset_metrics():
            st.success("Metrics reset!")
        else:
            st.error("Failed to reset metrics")
    st.divider()
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
For questions outside these topics, it will provide general answers using GPT-3.5.
""")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Main chat interface
st.header("💬 Ask Your Biohacking Question")
st.warning("⚠️ Important Note: This AI may generate incorrect or nonsensical responses, especially for unclear or ambiguous questions. Always verify information with reliable sources.")

question = st.chat_input("Ask about biohacking protocols and optimization...")

if question:
    question_lower = question.lower().strip()
    start_time = time.time()
    
    # Define biohacking scope
    biohacking_topics = ["exercise", "nutrition", "stress", "recovery", "performance", "daily routines", "habits", "sleep", "biohacking"]
    is_in_scope = any(topic in question_lower for topic in biohacking_topics) or "biohacking" in question_lower.split()
    
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Biohacking-specific logic (in-scope questions)
    if is_in_scope and st.session_state.vector_store:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=api_key)
        
        prompt_template = """You are a knowledgeable biohacking assistant. Your goal is to help users optimize their health and performance through evidence-based protocols and interventions.
        When answering biohacking questions:
        1. Provide specific, evidence-based advice with practical examples (e.g., timing, dosage, duration).
        2. Include safety considerations, potential contraindications, and the importance of consulting a healthcare professional.
        3. Be concise but thorough, and invite follow-up questions to deepen the conversation.
        4. If you don’t know the answer, admit it clearly and suggest related biohacking topics the user might explore.
        Context: {context}
        Question: {question}
        Answer: """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing protocols..."):
                try:
                    logger.info(f"Invoking QA with in-scope question: {question}")
                    docs = st.session_state.vector_store.similarity_search(question, k=5)
                    logger.info(f"Documents retrieved for question '{question}': {[doc.page_content for doc in docs]}")
                    if not docs:
                        logger.warning("No documents retrieved for question: %s", question)
                        answer = "I don’t have enough specific biohacking data for this question. Here’s a general response instead."
                    else:
                        response = qa.invoke(question)
                        logger.info(f"QA response: {response}")
                        
                        if isinstance(response, dict) and "result" in response and response["result"].strip():
                            answer = response["result"].strip()
                        elif isinstance(response, str) and response.strip():
                            answer = response.strip()
                        else:
                            answer = "I’m sorry, I don’t have enough biohacking-specific information for this. Here’s a general answer instead."
                except Exception as e:
                    logger.error(f"Error in QA chain: {str(e)}")
                    answer = "An error occurred with the biohacking data. I’ll provide a general response instead."
    else:
        # Out-of-scope or vector store unavailable: Use GPT-3.5 directly
        logger.info(f"Question out of biohacking scope or vector store unavailable: {question}")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
                    general_prompt = f"""You are a helpful biohacking assistant. Provide a concise, informative response to the following question about biohacking, nutrition, or health optimization. If it’s unrelated to biohacking, answer generally; if it’s biohacking-related but lacks specific data, give a broad but useful reply and suggest consulting an expert where applicable.

                    Question: {question}
                    Answer: """
                    response = llm.invoke(general_prompt)
                    answer = response.content.strip() if hasattr(response, 'content') else response.strip()
                except Exception as e:
                    logger.error(f"Error in GPT-3.5 fallback: {str(e)}")
                    answer = "Sorry, I couldn’t process that question. Please try again!"

    # Finalize response
    latency = time.time() - start_time
    if st.session_state.evaluator:
        st.session_state.evaluator.log_interaction(
            question=question,
            response=answer,
            latency=latency
        )
    
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

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
                    filepath = os.path.join(protocol_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                protocol_texts.append(content)
                            else:
                                logger.warning(f"Skipping empty file: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to read {filename}: {str(e)}")

            if not protocol_texts:
                logger.warning("No valid protocol files found.")
                st.warning("No biohacking protocols found. Please add .md files to the 'protocols' directory.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
            chunks = []
            for text in protocol_texts:
                split_chunks = text_splitter.split_text(text)
                valid_chunks = [chunk for chunk in split_chunks if chunk and chunk.strip()]
                if valid_chunks:
                    chunks.extend(valid_chunks)
                else:
                    logger.warning(f"No valid chunks from text: {text[:50]}...")

            if not chunks:
                logger.error("No valid chunks created.")
                st.error("Failed to process protocol texts into valid chunks.")
                return None

            logger.info(f"Number of valid chunks: {len(chunks)}")
            logger.info(f"Sample chunk: {chunks[0] if chunks else 'None'}")

            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
            logger.info("Vector store successfully initialized with %d documents", len(chunks))
            return vector_store
    except Exception as e:
        logger.error(f"Failed to load protocols: {str(e)}")
        st.error(f"Failed to load protocols: {str(e)}")
        return None

# Initialize vector store if not already done
if st.session_state.vector_store is None:
    st.session_state.vector_store = initialize_vector_store()
    st.write("Vector store status:", "Initialized" if st.session_state.vector_store else "Failed")
