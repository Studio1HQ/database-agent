<div align="center">

<img src="https://raw.githubusercontent.com/Studio1HQ/Studio1HQ/main/assets/banners/database-agent.webp" alt="Data-Persistent AI Agent with Bright Data: A LangChain agent that stores conversations in a database and adds web RAG" width="100%" />

<br />
<br />

# Data-Persistent AI Agent with Bright Data

### A LangChain agent that stores conversations in a database and adds web RAG

A single-file Python agent built with LangChain and GPT-4 that saves conversations, entities and logs to SQLite or PostgreSQL, processes them in background workers, and optionally adds web data through Bright Data.

<br />

[![Built by Studio1](https://img.shields.io/badge/Built_by_Studio1-f97316?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTYwIiBmaWxsPSJub25lIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTAgMzVDMCAxNi42ODA3IDE0LjA3NDQgMS42NDg0NCAzMiAwLjEyNjk1M1YxNDFINTEuMDUyM0M1Mi40MzA4IDEzNy4zNDQgNTQuNzQ0NiAxMzMuNjg5IDU3Ljk5MzggMTMwLjAzM0M2MS4xNDQ2IDEyNi41MDggNjQuNjg5MiAxMjMuNDQgNjguNjI3NyAxMjAuODI5QzcyLjU2NjIgMTE4LjM0OSA3Ni4zNTY5IDExNi43ODIgODAgMTE2LjEyOVY4Ni43NTQ0Qzc1LjQ3MDggODcuOTI5MiA3MS4zODQ2IDg5LjY5MTkgNjcuNzQxNSA5Mi4wNDE1QzY0IDk0LjUyMjUgNjAuNzUwOCA5Ny40NiA1Ny45OTM4IDEwMC44NTRDNTYuNTM0NSAxMDIuNzIgNTUuMjU0NSAxMDQuNjU5IDU0LjE1MzkgMTA2LjY3MVYwSDEyNUMxNDQuMzMgMCAxNjAgMTUuNjY5OSAxNjAgMzVWMTI1QzE2MCAxNDQuMzMgMTQ0LjMzIDE2MCAxMjUgMTYwSDEyNFYxOEgxMDQuOTQ4QzEwMy41NjkgMjEuNjgxNiAxMDEuMjU1IDI1LjM2MjggOTguMDA2MiAyOS4wNDQ0Qzk0Ljg1NTQgMzIuNTk0MiA5MS4zMTA4IDM1LjY4NDEgODcuMzcyMyAzOC4zMTRDODMuNDMzOCA0MC44MTIgNzkuNjQzMSA0Mi4zODk2IDc2IDQzLjA0NzRWNzIuNjMwNEM4MC41MjkyIDcxLjQ0NzMgODQuNjE1NCA2OS42NzI0IDg4LjI1ODUgNjcuMzA1N0M5MiA2NC44MDc2IDk1LjI0OTIgNjEuODQ5MSA5OC4wMDYyIDU4LjQzMDdDOTkuNDY1NSA1Ni41NTEzIDEwMC43NDUgNTQuNTk4NiAxMDEuODQ2IDUyLjU3MjNWMTYwSDM1QzE1LjY3IDE2MCAwIDE0NC4zMyAwIDEyNVYzNVoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4K)](https://studio1hq.com)
[![Built with Bright Data](https://img.shields.io/badge/Built_with_Bright_Data-141414?style=for-the-badge)](https://brightdata.com)

[Bright Data docs](https://docs.brightdata.com) &nbsp;·&nbsp; [Bright Data](https://brightdata.com) &nbsp;·&nbsp; [More examples](https://github.com/Studio1HQ) &nbsp;·&nbsp; [studio1hq.com](https://studio1hq.com)

</div>

<br />

## Overview

A LangChain agent that saves every conversation to a database and remembers past interactions across sessions. The whole system lives in `db_agent.py`: SQLAlchemy models, an agent with database tools, background workers, monitoring and an analytics interface.

Bright Data is an optional last step (Step 7 in `db_agent.py`). A `BrightDataRAGEnhancer` pulls records from the Bright Data Datasets API (`/datasets/v3/snapshot/<dataset_id>`), splits them into chunks, and stores them in a Chroma vector store so the agent can answer with real-time web data.

<p align="center"><img src="cursorful-video-1764615164579-ezgif.com-video-to-gif-converter%20(1).gif" alt="Demo of the data-persistent agent" width="100%" /></p>

## Features

- **Conversation persistence**: Saves users, sessions, messages, entities and operation logs.
- **Cross-session memory**: Retrieves a user's history on later conversations.
- **Entity extraction**: Pulls entities and insights from conversations with confidence scores.
- **Background workers**: Queue-based summarization, preference extraction and entity identification.
- **Monitoring**: Performance metrics, error rates and a health check.
- **Retry handling**: Automatic retry with a configurable maximum.
- **Optional web RAG**: Bright Data dataset records indexed in Chroma, with hybrid database and vector search.

## Tech stack

| Layer | Tool |
| --- | --- |
| Agent | [LangChain](https://www.langchain.com) |
| LLM | [OpenAI](https://platform.openai.com) (`gpt-4-turbo-preview` by default) |
| Database | SQLite or PostgreSQL through [SQLAlchemy](https://www.sqlalchemy.org) |
| Vector store | [Chroma](https://www.trychroma.com) |
| Web data (optional) | [Bright Data](https://brightdata.com) Datasets API |

## Getting started

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ or SQLite 3.35+
- An OpenAI API key
- Optional: a Bright Data API key for web data

### Installation

```bash
git clone https://github.com/Studio1HQ/database-agent.git
cd database-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install langchain langchain-openai sqlalchemy psycopg2-binary python-dotenv pydantic langchain-chroma requests
```

### Environment variables

Create a `.env` file:

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | `sqlite:///./agent_data.db` or `postgresql://user:password@localhost:5432/agent_db`. |
| `OPENAI_API_KEY` | OpenAI API key. |
| `BRIGHT_DATA_API_KEY` | Optional. Bright Data API key for the web RAG step. |
| `AGENT_MODEL` | Model name, for example `gpt-4-turbo-preview`. |
| `CONNECTION_POOL_SIZE` | Database pool size. Example `5`. |
| `MAX_RETRIES` | Retry count. Example `3`. |

### Run

```bash
python db_agent.py
```

The script initializes the database and tables, builds the agent with its database tools, starts the background workers, runs demo conversations, and prints performance metrics and analytics.

## How it works

```text
User Input
    |
AI Agent (LangChain + GPT-4)
    |
Database Tools (SQLAlchemy)
    |
PostgreSQL / SQLite
    |
Background Workers (threading)
    |
Data Collection and Analysis
```

Main classes in `db_agent.py`:

- **Models**: `User`, `Conversation`, `Message`, `Entity`, `AgentLog`.
- **`DatabaseManager`**: Connection pool, table creation, health check.
- **`DataPersistentAgent`**: Agent tools `save_user_info`, `retrieve_user_history`, `extract_entities`, and the `chat` method.
- **`DataCollector`** and **`DataProcessingPipeline`**: Summaries, preferences and entities in background queues.
- **`AgentMonitor`** and **`DataQueryInterface`**: Metrics, health and per-user analytics.
- **`BrightDataRAGEnhancer`**: Optional dataset fetch and ingestion into Chroma (`./chroma_db`).

To extend it, add new database tools, custom entity extractors, authentication, REST endpoints or more workers.

## Resources

- [Bright Data docs](https://docs.brightdata.com): Datasets API and web data products.
- [Bright Data](https://brightdata.com): Account and API key.
- [LangChain docs](https://python.langchain.com): Agents and tools.

<br />

---

<div align="center">

<a href="https://studio1hq.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-dark.svg" />
    <img src="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-light.svg" alt="Studio1" width="160" />
  </picture>
</a>

Built by **[Studio1](https://studio1hq.com)**, the technical content and DevRel partner for devtool teams. Made with [Bright Data](https://brightdata.com).<br />
Browse more open-source examples at [github.com/Studio1HQ](https://github.com/Studio1HQ).

[![Website](https://img.shields.io/badge/studio1hq.com-0b0807?style=for-the-badge&logo=googlechrome&logoColor=white)](https://studio1hq.com)
[![Book a call](https://img.shields.io/badge/Book_a_strategy_call-f97316?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://cal.com/studio1/collab)
[![X](https://img.shields.io/badge/%40Studio1HQ-141414?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Studio1HQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/studio1hq/)

<sub><a href="https://studio1hq.com">studio1hq.com</a> · Technical content, docs, video and DevRel for devtools</sub>

</div>
