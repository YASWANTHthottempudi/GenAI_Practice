# ChatApp Demo - Multi-Language AI Chat with Vector Database

A comprehensive chat application demo built with LangChain, Groq API, and Chroma vector database. This project demonstrates multi-language conversation capabilities, message history management, and retrieval-augmented generation (RAG) using vector embeddings.

## Features

- 🌍 **Multi-language Support**: Chat in English, Telugu, Hindi, and other languages
- 🧠 **Multiple AI Models**: Integration with Groq's Gemma2 and Llama3 models
- 📚 **Vector Database**: Chroma DB with HuggingFace embeddings for document retrieval
- 💬 **Conversation History**: Persistent chat sessions with message trimming
- 🔍 **RAG Implementation**: Retrieval-augmented generation for context-aware responses

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YASWANTHthottempudi/GenAI_Practice.git
cd GenAI_Practice
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
1. Copy `env_example.txt` to `.env`
2. Add your API keys:
   - Get Groq API key from [Groq Console](https://console.groq.com/)
   - Get HuggingFace token from [HF Settings](https://huggingface.co/settings/tokens) (optional)

```bash
cp env_example.txt .env
# Edit .env with your actual API keys
```

### 5. Run the Notebook
```bash
jupyter notebook chatapp.ipynb
```

## Usage

### Basic Chat
```python
# Simple chat with message history
response = with_message_history.invoke(
    [HumanMessage(content="Hello, I am Yash working as an AI Engineer")], 
    config={"configurable": {"session_id": "chat1"}}
)
print(response.content)
```

### Multi-language Chat
```python
# Chat in different languages
telugu_response = chat_in_language("Telugu", "What is artificial intelligence?", "telugu_session")
english_response = chat_in_language("English", "What is artificial intelligence?", "english_session")
hindi_response = chat_in_language("Hindi", "What is artificial intelligence?", "hindi_session")
```

### Vector Database Search
```python
# Search documents using vector similarity
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
results = retriever.batch(["NLP", "Machine Learning"])
```

## Project Structure

```
ChatApp_Demo/
├── chatapp.ipynb          # Main notebook with all functionality
├── requirements.txt       # Python dependencies
├── env_example.txt       # Environment variables template
├── README.md             # This file
├── .gitignore           # Git ignore rules
├── test.ipynb           # Test notebook
└── test.py              # Test script
```

## Dependencies

- `langchain-groq`: Groq API integration
- `langchain-huggingface`: HuggingFace embeddings
- `langchain-chroma`: Chroma vector database
- `langchain-core`: Core LangChain functionality
- `python-dotenv`: Environment variable management
- `jupyter`: Notebook environment

## API Keys Required

1. **Groq API Key**: For accessing Groq's language models
   - Sign up at [Groq Console](https://console.groq.com/)
   - Generate API key in your dashboard

2. **HuggingFace Token** (Optional): For accessing certain models
   - Sign up at [HuggingFace](https://huggingface.co/)
   - Generate token in settings

## Security Note

⚠️ **Never commit API keys to version control!**
- Use environment variables for all sensitive data
- The `.env` file is already in `.gitignore`
- Always use the provided `env_example.txt` as a template

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Author

**Yaswanth Thottempudi**
- GitHub: [@YASWANTHthottempudi](https://github.com/YASWANTHthottempudi)
- AI Engineer specializing in Generative AI and Machine Learning
