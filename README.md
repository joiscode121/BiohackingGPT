# 🧬 BiohackingGPT: Your Biohacking Protocol Assistant

A Streamlit-based chatbot that helps users understand and implement biohacking protocols using RAG (Retrieval Augmented Generation) technology.

## 🎯 Features

- Interactive chat interface for biohacking questions
- RAG-based responses using curated biohacking protocols
- Real-time evaluation metrics (Number of User Interactions, Latency, and Average Response Length)

## 🛠️ Technology Stack & Why We Chose It

### Core Technologies
- **Streamlit**: Chosen for rapid development of web interfaces and built-in chat components. Easily spins up an integrated frontend and backend and its simple API and cloud hosting feature made it perfect for our chat application.
- **LangChain**: Selected for its robust RAG implementation and easy integration with various LLMs and vector stores.
- **ChromaDB**: Used as our vector store due to its simplicity, persistence capabilities, and good performance for small to medium datasets.
- **OpenAI GPT-3.5**: Chosen for its strong performance in natural language understanding and reasonable cost-effectiveness.

### Design Decisions
- **File-based Protocols**: Markdown files were chosen for storing various biohacking protocols ensuring easy maintenance and version control.
- **JSONL for Metrics**: Simple, append-only format that's perfect for logging and parsing as well as easy to integrate with Python.
- **Local Vector Store**: ChromaDB's local storage is simple and has great performance.

## 📋 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd BiohackingGPT
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 🔄 What Worked & What Didn't

### Successes
1. **RAG Implementation**
   - Successfully integrated LangChain with ChromaDB
   - Effective retrieval of relevant protocol information
   - Good response quality for clear questions

2. **Chat Interface**
   - Clean and intuitive Streamlit chat components
   - Persistent chat history
   - Real-time response display

3. **Basic Evaluation**
   - Simple but effective metrics tracking
   - Real-time updates
   - Easy reset functionality

### Challenges & Solutions

1. **Phoenix Integration**
   - **Challenge**: Attempted to use Phoenix for advanced evaluation but faced compatibility issues
   - **Solution**: Implemented a simpler file-based evaluation system using JSONL
   - **Future**: Would explore Phoenix alternatives or contribute to fixing compatibility issues

2. **Expert Mode**
   - **Challenge**: Tried implementing an "expert mode" with more technical responses
   - **Solution**: Currently using a single response mode with comprehensive but accessible information
   - **Future**: Could implement response modes using different prompt templates

3. **Vector Store Performance**
   - **Challenge**: Initial setup with FAISS had memory issues
   - **Solution**: Switched to ChromaDB for better local performance
   - **Future**: Would benchmark different vector stores for larger datasets

## 🚀 Future Improvements

1. **Advanced Evaluation**
   - Implement response quality metrics
   - Add user feedback system
   - Track topic-specific performance
   - Explore Phoenix alternatives for better monitoring

2. **Enhanced RAG**
   - Implement hybrid search (keyword + semantic)
   - Add support for more document types
   - Implement document chunking strategies
   - Add support for real-time protocol updates

3. **User Experience**
   - Add user authentication
   - Implement conversation memory
   - Add protocol visualization
   - Create a protocol suggestion system

4. **Infrastructure**
   - Containerize the application
   - Add automated testing
   - Implement CI/CD pipeline
   - Add proper logging and monitoring

5. **Content**
   - Expand protocol database
   - Add scientific references
   - Implement automatic protocol updates
   - Add multimedia content support

## 📝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[MIT License](LICENSE)
