# ARIA - AI-Powered Citizen Support Assistant

A Voice-First Service Assistant that enables citizens to ask questions (via voice or text) about government services and receive accurate, context-aware guidance powered by RAG (Retrieval-Augmented Generation).

## System Architecture

```
                                    ARIA SYSTEM ARCHITECTURE

    +------------------+          +------------------------------------------------+
    |                  |          |                   FRONTEND                      |
    |     CITIZEN      |          |              (Streamlit App)                    |
    |                  |          |                                                 |
    |  +------------+  |  HTTPS   |  +------------------------------------------+  |
    |  |   Voice    |------------>|  |              Chat Interface              |  |
    |  |   Input    |  |          |  |  - Text input with chat_input           |  |
    |  +------------+  |          |  |  - Audio recording (16kHz WAV)          |  |
    |                  |          |  |  - Message history display              |  |
    |  +------------+  |          |  |  - Source citations with confidence     |  |
    |  |   Text     |------------>|  |  - Audio playback for TTS responses     |  |
    |  |   Input    |  |          |  +------------------------------------------+  |
    |  +------------+  |          |                                                 |
    |                  |          |  +------------------------------------------+  |
    |  +------------+  |          |  |           Stats Dashboard                |  |
    |  | Audio Play |<------------|  |  - Response time metrics                 |  |
    |  |   back     |  |          |  |  - Cache hit rates                       |  |
    |  +------------+  |          |  |  - Performance charts                    |  |
    +------------------+          |  +------------------------------------------+  |
                                  +------------------------------------------------+
                                                        |
                                                        | HTTP/SSE
                                                        v
    +---------------------------------------------------------------------------------+
    |                              BACKEND API (FastAPI)                              |
    |                                                                                 |
    |  +---------------------------+    +---------------------------+                 |
    |  |      Query Routes         |    |     Document Routes       |                 |
    |  |  POST /api/query/text     |    |  POST /api/documents      |                 |
    |  |  POST /api/query/audio    |    |  GET  /api/documents      |                 |
    |  |  POST /api/query/stream   |    |  DELETE /api/documents    |                 |
    |  +---------------------------+    +---------------------------+                 |
    |               |                              |                                  |
    |               v                              v                                  |
    |  +-----------------------------------------------------------------------+     |
    |  |                         RAG PIPELINE (LlamaIndex)                      |     |
    |  |                                                                        |     |
    |  |   +----------------+    +------------------+    +------------------+   |     |
    |  |   |   Document     |    |   Vector Store   |    |  Query Engine    |   |     |
    |  |   |   Loaders      |--->|   (In-Memory)    |--->|  (top_k=5)       |   |     |
    |  |   | JSON/MD/PDF/   |    |                  |    |                  |   |     |
    |  |   | DOCX/TXT       |    |  OpenAI          |    |  GPT-4o-mini     |   |     |
    |  |   +----------------+    |  Embeddings      |    |  Temperature=0.1 |   |     |
    |  |                         +------------------+    +------------------+   |     |
    |  +-----------------------------------------------------------------------+     |
    |               |                                           |                     |
    +---------------|-------------------------------------------|---------------------+
                    |                                           |
                    v                                           v
    +---------------------------+               +---------------------------+
    |    EXTERNAL AI SERVICES   |               |     SESSION & CACHE       |
    |                           |               |        (Redis)            |
    |  +---------------------+  |               |                           |
    |  |   Groq API          |  |               |  +---------------------+  |
    |  |   - Whisper STT     |  |               |  |  Session Store      |  |
    |  |   - Orpheus TTS     |  |               |  |  - Chat history     |  |
    |  +---------------------+  |               |  |  - 30min TTL        |  |
    |                           |               |  +---------------------+  |
    |  +---------------------+  |               |                           |
    |  |   OpenAI API        |  |               |  +---------------------+  |
    |  |   - GPT-4o-mini     |  |               |  |  Query Cache        |  |
    |  |   - Embeddings      |  |               |  |  - Response cache   |  |
    |  |   - TTS (fallback)  |  |               |  |  - 1hr TTL          |  |
    |  +---------------------+  |               |  +---------------------+  |
    +---------------------------+               |                           |
                                                |  +---------------------+  |
                                                |  |  Embedding Cache    |  |
                                                |  |  - 24hr TTL         |  |
                                                |  +---------------------+  |
                                                +---------------------------+
```

## Request Flow

```
                              REQUEST FLOW DIAGRAM

    TEXT QUERY:
    ============

    User Input ──> [Cache Check] ──> Cache Hit? ──Yes──> Return Cached Response
                         |                                      |
                         No                                     v
                         |                              [TTS Generation]
                         v                                      |
                  [Vector Search]                               v
                    (top_k=5)                            Return to User
                         |
                         v
                  [LLM Generation]
                   (GPT-4o-mini)
                         |
                         v
                  [Cache Response]
                         |
                         v
                  [TTS Generation]
                   (Groq/OpenAI)
                         |
                         v
                  Return to User


    AUDIO QUERY:
    ============

    Audio File ──> [Groq Whisper STT] ──> Transcribed Text ──> [Text Query Flow]
                         |                                            |
                         v                                            v
                   Transcript                                   Response + Audio
                         |                                            |
                         +────────────────────────────────────────────+
                                              |
                                              v
                                    Return Combined Response


    STREAMING RESPONSE (SSE):
    =========================

    Query ──> [Session Event] ──> [Sources Event] ──> [Token Events...] ──> [Done Event] ──> [Audio Event]
                  |                      |                   |                    |               |
                  v                      v                   v                    v               v
            session_id             sources[]           "tok","en","s"      full_response    audio_base64
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit 1.40+ | Chat UI with voice input |
| **Backend** | FastAPI + Gunicorn | RESTful API server |
| **RAG Framework** | LlamaIndex | Document indexing & retrieval |
| **LLM** | OpenAI GPT-4o-mini | Response generation |
| **Embeddings** | OpenAI text-embedding-3-small | Vector embeddings |
| **STT** | Groq Whisper large-v3-turbo | Speech-to-text |
| **TTS** | Groq Orpheus / OpenAI TTS-1 | Text-to-speech |
| **Cache/Sessions** | Redis 7 | Caching & session management |
| **Container** | Docker + Docker Compose | Deployment |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key
- Groq API Key

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/aria-assistant.git
cd aria-assistant

# Create environment file
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
REDIS_PASSWORD=your-secure-password
```

### 2. Add Knowledge Documents

Place your knowledge documents in `backend/data/knowledge/`:

```bash
# Supported formats: .json (IremboGov), .md, .txt, .pdf, .docx
cp your-documents/* backend/data/knowledge/
```

### 3. Run with Docker

```bash
# Development
docker compose up --build

# Production
docker compose -f docker-compose.prod.yml up --build -d
```

### 4. Index Knowledge Documents

After adding documents, trigger indexing:

```bash
# Via curl
curl -X POST http://localhost:8635/api/documents/refresh

# Or use the API docs UI at http://localhost:8635/docs
```

### 5. Access the Application

- **Chat Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8635/docs
- **Stats Dashboard**: http://localhost:8501/stats

## API Reference

### Query Endpoints

#### Text Query
```http
POST /api/query/text
Content-Type: application/json

{
  "query": "How do I apply for a passport?",
  "session_id": "optional-session-id",
  "include_audio": true
}
```

Response:
```json
{
  "query": "How do I apply for a passport?",
  "answer": "To apply for a passport, you need to...",
  "sources": [
    {"title": "Passport Application", "url": "...", "score": 0.95}
  ],
  "session_id": "abc123",
  "audio_base64": "UklGRi..."
}
```

#### Text Query with Streaming
```http
POST /api/query/text/stream
Content-Type: application/json
Accept: text/event-stream

{
  "query": "What documents do I need?",
  "session_id": "abc123"
}
```

SSE Events:
```
event: session
data: {"type": "session", "session_id": "abc123"}

event: sources
data: {"type": "sources", "sources": [...], "confidence": 0.85}

event: token
data: {"type": "token", "token": "You"}

event: done
data: {"type": "done", "full_response": "You need..."}

event: audio
data: {"type": "audio", "audio_base64": "..."}
```

#### Audio Query
```http
POST /api/query/audio
Content-Type: multipart/form-data

file: <audio-file.wav>
session_id: optional-session-id
include_audio: true
```

Response:
```json
{
  "transcript": "How do I apply for a birth certificate?",
  "answer": "To apply for a birth certificate...",
  "sources": [...],
  "session_id": "abc123",
  "audio_base64": "..."
}
```

### Document Management

```http
# Upload document
POST /api/documents/upload
Content-Type: multipart/form-data
file: <document.pdf>

# List documents
GET /api/documents

# Delete document
DELETE /api/documents/{doc_id}

# Refresh index
POST /api/documents/refresh
```

### Health & Stats

```http
# Health check
GET /health

# Performance stats
GET /api/stats
```

## Design Choices

### Why These Technologies?

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **STT Model** | Groq Whisper | 10x faster than local Whisper, near real-time transcription |
| **LLM** | GPT-4o-mini | Best cost/performance ratio, fast inference |
| **RAG Framework** | LlamaIndex | Better RAG abstractions than LangChain for document Q&A |
| **Embeddings** | text-embedding-3-small | Good accuracy, lower cost, 1536 dimensions |
| **TTS** | Groq Orpheus + OpenAI fallback | Groq for speed, OpenAI for reliability |
| **Vector Store** | LlamaIndex In-Memory | Sufficient for prototype scale, persistent to disk |
| **Cache** | Redis | Fast, supports TTL, production-proven |

### Latency Optimization Strategies

1. **Response Streaming (SSE)**: Tokens stream as generated, reducing perceived latency
2. **Parallel TTS**: Audio generation starts while streaming completes
3. **Multi-Layer Caching**:
   - Embedding cache (24hr TTL) - avoid redundant API calls
   - Query cache (1hr TTL) - instant responses for repeated queries
   - Audio cache (1hr TTL) - skip TTS for cached responses
4. **Connection Pooling**: Redis and HTTP clients maintain connection pools
5. **Async Everything**: FastAPI async handlers, async Redis, async LLM calls

### Handling Ambiguity

1. **Confidence Scoring**: Each response includes a confidence score (0.0-1.0)
   - Formula: `0.7 * max_score + 0.3 * avg_top_3_scores`
   - Displayed to users with High/Medium/Low badges
2. **Source Citations**: All answers include source documents with relevance scores
3. **Follow-up Detection**: System detects pronouns ("it", "that", "this") and uses conversation history
4. **Out-of-Scope Handling**: System prompt instructs to politely decline unrelated queries

### Scalability Considerations

For 1,000+ concurrent users:

1. **Horizontal Scaling**: Gunicorn with 4 workers per container, scale containers
2. **Redis Cluster**: Replace single Redis with Redis Cluster for session distribution
3. **Vector Store**: Migrate to Pinecone/Qdrant for distributed vector search
4. **Load Balancer**: Traefik/nginx in front of API containers
5. **Rate Limiting**: Add per-session rate limits to prevent abuse

## Assignment Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Multi-Modal Input** | Done | `/api/query/text` and `/api/query/audio` endpoints |
| **Speech Processing (STT)** | Done | Groq Whisper large-v3-turbo |
| **Knowledge Ingestion** | Done | LlamaIndex document loaders (JSON, MD, PDF, DOCX, TXT) |
| **RAG Pattern** | Done | LlamaIndex VectorIndexRetriever + GPT-4o-mini |
| **Text Response** | Done | All endpoints return text answers |
| **Audio Response (TTS)** | Done | Groq Orpheus with OpenAI fallback |
| **Session Management** | Done | Redis-backed sessions with 30min TTL |
| **Low Latency** | Done | Streaming, caching, connection pooling |
| **No Hallucination** | Done | RAG grounding + explicit system prompt |
| **Production Quality** | Done | Docker, health checks, logging, error handling |
| **Architecture Diagram** | Done | ASCII diagrams in this README |
| **Dockerized Setup** | Done | docker-compose.yml and docker-compose.prod.yml |

## Project Structure

```
aria-assistant/
├── backend/
│   ├── agent/
│   │   ├── rag/
│   │   │   ├── indexer.py       # Vector index management
│   │   │   ├── loaders.py       # Document loaders
│   │   │   ├── query_engine.py  # RAG query engine
│   │   │   └── prompts.py       # System prompts
│   │   ├── session/
│   │   │   └── redis_store.py   # Session management
│   │   └── utils/
│   │       └── config.py        # Configuration
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py         # Query endpoints
│   │   │   ├── documents.py     # Document management
│   │   │   ├── health.py        # Health checks
│   │   │   └── stats.py         # Performance stats
│   │   └── server.py            # FastAPI app
│   ├── data/knowledge/          # Knowledge documents
│   ├── storage/index/           # Persisted vector index
│   ├── Dockerfile.api           # API container
│   └── pyproject.toml           # Dependencies
├── frontend-streamlit/
│   ├── app.py                   # Main chat interface
│   ├── pages/
│   │   └── stats.py             # Stats dashboard
│   ├── .streamlit/
│   │   └── config.toml          # Streamlit config
│   └── Dockerfile               # Frontend container
├── docker-compose.yml           # Development setup
├── docker-compose.prod.yml      # Production setup
└── README.md                    # This file
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for LLM and embeddings |
| `GROQ_API_KEY` | Yes | - | Groq API key for STT and TTS |
| `REDIS_PASSWORD` | Yes | - | Redis authentication password |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection URL |
| `API_URL` | No | `http://localhost:8000` | Backend API URL (for frontend) |
| `LOG_LEVEL` | No | `info` | Logging level |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins |

## Development

### Local Development (without Docker)

```bash
# Backend
cd backend
pip install -e .
uvicorn api.server:app --reload --port 8000

# Frontend
cd frontend-streamlit
pip install -r requirements.txt
streamlit run app.py
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Built for the AI-Powered Citizen Support Assistant take-home assignment
- Designed for Rwanda's Irembo e-government platform context
