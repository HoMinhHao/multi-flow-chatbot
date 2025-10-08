# Multi-Flow Chatbot with RAG and LangGraph

A sophisticated chatbot system that classifies user inputs into three distinct flows (general chat, file analysis, and image generation) using GPT-4, with memory management and RAG capabilities powered by Weaviate.

## Architecture Overview

- **Classification**: GPT-4.1 classifies user input into one of three flows
- **Flow Processing**: Separate modules for each flow (general_chat, file_analysis, image_generation)
- **Memory**: Conversation history management with LangGraph
- **RAG**: Weaviate vector database for text and image storage/retrieval
- **Structure**: MVC pattern for clean code organization

## Project Structure

```
chatbot/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration settings
├── models/                         # Models (MVC)
│   ├── __init__.py
│   ├── classifier.py               # Input classification model
│   ├── memory.py                   # Memory management
│   └── rag.py                      # RAG with Weaviate
├── views/                          # Views (MVC)
│   ├── __init__.py
│   └── response_formatter.py       # Response formatting
├── controllers/                    # Controllers (MVC)
│   ├── __init__.py
│   ├── chat_controller.py          # Main chat controller
│   └── graph_builder.py            # LangGraph state graph
├── flows/                          # Flow-specific processing
│   ├── __init__.py
│   ├── general_chat/
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── file_analysis/
│   │   ├── __init__.py
│   │   └── processor.py
│   └── image_generation/
│       ├── __init__.py
│       └── processor.py
├── utils/
│   ├── __init__.py
│   └── helpers.py                  # Utility functions
└── tests/
    ├── __init__.py
    └── test_flows.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/HoMinhHao/multi-flow-chatbot.git
cd multi-flow-chatbot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export WEAVIATE_URL="your-weaviate-url"
export WEAVIATE_API_KEY="your-weaviate-api-key"  # Optional
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key
```

## Quick Start

### Interactive Mode

```bash
python app.py
```

### Programmatic Usage

```python
from app import ChatbotApp

# Initialize the chatbot
app = ChatbotApp()
app.start()

# Process messages
response = app.process("Hello, how are you?")
print(response)

# Add documents to knowledge base
app.add_document_to_knowledge("path/to/document.txt", "Document description")

# Add images to knowledge base
app.add_document_to_knowledge("path/to/image.jpg", "Image description")

# Clear conversation history
app.clear_conversation()

# Close resources
app.close()
```

## Features

### 🎯 Intelligent Classification
Automatically routes user input to the appropriate processing flow using GPT-4:
- **General Chat**: Conversational AI for questions, greetings, and general assistance
- **File Analysis**: Document processing, analysis, and information extraction
- **Image Generation**: DALL-E 3 integration for creating images from text

### 💬 General Chat Flow
- Natural conversation with context awareness
- Access to RAG knowledge base
- Personality and tone consistency

### 📄 File Analysis Flow
- Document summarization
- Information extraction
- Question answering about file contents
- Supports multiple document formats

### 🎨 Image Generation Flow
- DALL-E 3 integration
- Natural language to image
- Prompt refinement and optimization

### 🧠 Memory Management
- Conversation history tracking
- Configurable message limits
- Session-based memory
- Metadata storage

### 🔍 RAG with Weaviate
- Vector search for text documents
- Image storage with CLIP embeddings
- Semantic similarity search
- Multi-modal retrieval

### 🔄 LangGraph Integration
- State-based conversation flow
- Conditional routing
- Parallel processing capabilities
- Error handling and recovery

## Usage Examples

### Interactive CLI Commands

```bash
# Start interactive mode
python app.py

# Available commands:
You: Hello!
Bot [general_chat]: Hi! How can I help you today?

You: /add documents/report.pdf
✓ Text document 'report.pdf' added successfully

You: Analyze the report I just uploaded
Bot [file_analysis]: Based on the document...

You: Create an image of a sunset over mountains
Bot [image_generation]: I've generated an image for you...
Image URL: https://...

You: /memory
=== Memory Summary ===
Messages: 8
...

You: /clear
✓ Conversation history cleared

You: /quit
Goodbye!
```

### Python API Usage

```python
from app import ChatbotApp

# Initialize
app = ChatbotApp()
app.start(session_id="user_123")

# General conversation
response = app.process("What's the weather like?")
print(response['message'])
print(f"Flow: {response['flow']}")
print(f"Confidence: {response['confidence']}")

# File analysis
app.add_document_to_knowledge(
    filepath="data/research_paper.pdf",
    description="Research paper on AI"
)
response = app.process("Summarize the research paper")
print(response['message'])
print(f"Files analyzed: {response.get('analyzed_files', [])}")

# Image generation
response = app.process("Generate an image of a futuristic city")
print(response['message'])
print(f"Image URL: {response.get('image_url')}")

# Memory management
app.show_memory_summary()
app.clear_conversation()

# Cleanup
app.close()
```

## Architecture Details

### MVC Pattern

**Models** (`models/`):
- `classifier.py`: Input classification using GPT-4
- `memory.py`: Conversation history management
- `rag.py`: Weaviate integration for RAG

**Views** (`views/`):
- `response_formatter.py`: Response formatting and presentation

**Controllers** (`controllers/`):
- `chat_controller.py`: Main orchestration logic
- `graph_builder.py`: LangGraph state machine

### Flow Processing

Each flow is isolated in its own module:

```
flows/
├── general_chat/processor.py      # Conversational AI
├── file_analysis/processor.py     # Document processing
└── image_generation/processor.py  # DALL-E integration
```

This structure makes it easy to:
- Add new flows
- Modify existing flows
- Test flows independently
- Maintain separation of concerns

### LangGraph State Machine

```
User Input
    ↓
[Classify] → Determine flow type
    ↓
[Retrieve Context] → Get relevant RAG data (if needed)
    ↓
    ├→ [General Chat] → END
    ├→ [File Analysis] → END
    └→ [Image Generation] → END
```

## Configuration

Edit `config/settings.py` to customize:

```python
class Settings(BaseModel):
    # Models
    classification_model: str = "gpt-4-turbo-preview"
    chat_model: str = "gpt-4-turbo-preview"
    
    # Memory
    max_memory_messages: int = 10
    
    # RAG
    rag_top_k: int = 5
    text_collection: str = "TextDocuments"
    image_collection: str = "ImageDocuments"
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_flows.py::TestGeneralChatProcessor

# Run with coverage
python -m pytest --cov=. tests/
```

## Extending the System

### Adding a New Flow

1. Create a new directory: `flows/new_flow/`
2. Create `processor.py`:

```python
class NewFlowProcessor:
    def __init__(self):
        # Initialize your processor
        pass
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Process the state
        return {
            **state,
            "response": "Your response",
            "flow_used": "new_flow"
        }
    
    def process_sync(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Synchronous version
        pass
```

3. Update `flows/__init__.py`
4. Update `controllers/graph_builder.py` to add the new flow
5. Update `models/classifier.py` to include the new flow type

## Troubleshooting

### Weaviate Connection Issues

```python
# Check if Weaviate is running
curl http://localhost:8080/v1/meta

# Run Weaviate with Docker
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

### OpenAI API Issues

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test connection
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

## Performance Optimization

- **Caching**: Implement response caching for common queries
- **Batch Processing**: Process multiple messages in parallel
- **Async Operations**: Use async methods for better concurrency
- **Vector Index**: Optimize Weaviate indexing for faster retrieval

## Security Considerations

- Store API keys in environment variables or secure vaults
- Implement rate limiting for production use
- Sanitize user inputs before processing
- Use HTTPS for Weaviate connections in production
- Implement authentication for multi-user scenarios

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Vector database powered by [Weaviate](https://weaviate.io/)
- AI models by [OpenAI](https://openai.com/)

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions

## Roadmap

- [ ] Add support for more document formats (PDF, DOCX, etc.)
- [ ] Implement streaming responses
- [ ] Add voice input/output
- [ ] Multi-language support
- [ ] Web interface with React
- [ ] Docker deployment configuration
- [ ] Kubernetes manifests
- [ ] More flow types (code generation, data analysis, etc.)
- [ ] Enhanced RAG with re-ranking
- [ ] Conversation summarization
- [ ] Export conversation history

---

**Built with ❤️ using LangGraph, GPT-4, and Weaviate**
