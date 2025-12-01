# Data-Persistent AI Agent

A production-ready AI agent that saves conversations to databases. Built with LangChain, GPT-4, and PostgreSQL/SQLite.

## What This Does

This agent system:
- Saves every conversation to a database
- Remembers past interactions across sessions
- Extracts entities and insights automatically
- Processes data in background workers
- Monitors performance with real-time metrics
- Handles errors with automatic retry

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ or SQLite 3.35+
- OpenAI API key
- (Optional) Bright Data API key for web data integration

## Installation

1. Clone the repository and navigate to the project:
```bash
cd data-persistant-agent
```

2. Create a virtual environment:
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-openai sqlalchemy psycopg2-binary python-dotenv pydantic langchain-chroma requests
```

4. Create a `.env` file with your credentials:
```bash
# Database Configuration
DATABASE_URL="sqlite:///./agent_data.db"
# Or for PostgreSQL: DATABASE_URL="postgresql://username:password@localhost:5432/agent_db"

# API Keys
OPENAI_API_KEY="your-openai-api-key"

# Optional
BRIGHT_DATA_API_KEY="your-bright-data-api-key"

# Application Settings
AGENT_MODEL="gpt-4-turbo-preview"
CONNECTION_POOL_SIZE=5
MAX_RETRIES=3
```

## Usage

Run the agent:
```bash
python db_agent.py
```

The system will:
1. Initialize the database and create tables
2. Build the AI agent with database tools
3. Start background processing workers
4. Run demo conversations
5. Display performance metrics and analytics

## Project Structure

- `db_agent.py` - Main agent implementation with all components
- `content.md` - Complete tutorial and documentation
- `.env` - Environment configuration (create this)
- `chroma_db/` - Vector store for RAG (created automatically)
- `agent_data.db` - SQLite database (created automatically)

## Key Features

### Database Schema
- User profiles with preferences
- Conversation sessions with status tracking
- Individual messages with metadata
- Extracted entities with confidence scores
- Operation logs for monitoring

### AI Agent Core
- LangChain integration with custom tools
- Function calling for database operations
- Conversation memory across sessions
- Automatic entity extraction

### Background Processing
- Conversation summarization
- User preference extraction
- Entity identification
- Queue-based architecture

### Monitoring
- Performance metrics tracking
- Error rate monitoring
- Health check system
- Query analytics

### RAG Enhancement (Optional)
- Bright Data web intelligence integration
- Vector store with Chroma
- Hybrid search (database + vector)
- Real-time web data ingestion

## Customization

Extend the system by:
- Adding new database tools
- Creating custom entity extractors
- Implementing user authentication
- Building REST API endpoints
- Adding more background workers
- Integrating additional data sources

## Architecture

```
User Input
    ↓
AI Agent (LangChain + GPT-4)
    ↓
Database Tools (SQLAlchemy)
    ↓
PostgreSQL/SQLite
    ↓
Background Workers (Threading)
    ↓
Data Collection & Analysis
```

## Database Models

- **User**: Stores user profiles and preferences
- **Conversation**: Tracks conversation sessions
- **Message**: Individual messages with role and content
- **Entity**: Extracted named entities
- **AgentLog**: Operation logs and metrics

## Error Handling

The system includes:
- Automatic retry with exponential backoff
- Transaction management
- Connection pooling with health checks
- Error logging to database
- Graceful degradation

## Performance

- Connection pooling for efficiency
- Background processing to avoid blocking
- Indexed database queries
- Batch processing support
- Queue monitoring

## Contributing

This is a tutorial project. Feel free to fork and customize for your needs.

## License

MIT

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Bright Data Platform](https://brightdata.com/)

## Support

For issues or questions, refer to the complete tutorial in `content.md`.
