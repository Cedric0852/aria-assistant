# ARIA - AI-Powered Citizen Support Assistant

An intelligent voice-first assistant for Rwanda government services (IremboGov) using RAG-powered knowledge retrieval with intelligent domain classification.

## Features

- **Voice Input**: Real-time speech recognition via LiveKit + Groq Whisper
- **Text Input**: Type questions directly with instant responses
- **Audio Output**: Text-to-speech responses via Groq PlayAI TTS
- **RAG Knowledge Base**: LlamaIndex-powered retrieval from IremboGov documents
- **Domain Classification**: Pydantic AI classifier filters off-topic queries before RAG
- **Session Memory**: Follow-up questions with conversation context
- **Session-Aware Caching**: Redis caching for text + audio responses (12x faster repeat queries)
- **Multi-format Documents**: JSON, PDF, DOCX, MD, TXT support
- **Scalable**: Docker Compose with replicas + Gunicorn workers

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENTS                                        │
│  ┌─────────────────────────────┐     ┌─────────────────────────────┐            │
│  │      Next.js Frontend       │     │      REST API Clients       │            │
│  │   (Voice + Text Interface)  │     │    (curl, Postman, etc)     │            │
│  └──────────────┬──────────────┘     └──────────────┬──────────────┘            │
└─────────────────┼───────────────────────────────────┼────────────────────────────┘
                  │                                   │
                  │ WebRTC/WebSocket                  │ HTTP/REST
                  ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INFRASTRUCTURE                                      │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐           │
│  │     LiveKit Server      │          │    Nginx Load Balancer      │           │
│  │  ┌───────────────────┐  │          │  ┌───────────────────────┐  │           │
│  │  │ WebRTC Signaling  │  │          │  │   Round-robin routing │  │           │
│  │  │ Room Management   │  │          │  │   SSL Termination     │  │           │
│  │  │ Audio Transport   │  │          │  └───────────────────────┘  │           │
│  │  └───────────────────┘  │          └──────────────┬──────────────┘           │
│  └──────────────┬──────────┘                         │                          │
└─────────────────┼────────────────────────────────────┼──────────────────────────┘
                  │                                    │
                  ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND SERVICES                                    │
│                                                                                  │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │      LiveKit Agent          │    │           FastAPI Server                │ │
│  │  ┌───────────────────────┐  │    │  ┌───────────────────────────────────┐  │ │
│  │  │   CitizenAgent Class  │  │    │  │   REST Endpoints                  │  │ │
│  │  │   - Voice handling    │  │    │  │   - /api/query/text               │  │ │
│  │  │   - STT/TTS pipeline  │  │    │  │   - /api/query/audio              │  │ │
│  │  │   - Tool execution    │  │    │  │   - /api/documents                │  │ │
│  │  └───────────┬───────────┘  │    │  │   - /api/stats                    │  │ │
│  └──────────────┼──────────────┘    │  └───────────────┬───────────────────┘  │ │
│                 │                    └─────────────────┼───────────────────────┘ │
│                 │                                      │                         │
│                 └──────────────────┬───────────────────┘                         │
│                                    ▼                                             │
│              ┌─────────────────────────────────────────────────────┐             │
│              │                 RAG PIPELINE                        │             │
│              │  ┌─────────────────────────────────────────────┐    │             │
│              │  │  Layer 0: Domain Classifier (Pydantic AI)   │    │             │
│              │  │  - Filters off-topic queries                │    │             │
│              │  │  - Handles greetings/small talk directly    │    │             │
│              │  └──────────────────┬──────────────────────────┘    │             │
│              │                     │ (Irembo queries only)         │             │
│              │                     ▼                               │             │
│              │  ┌─────────────────────────────────────────────┐    │             │
│              │  │  Layer 1: Vector Search (LlamaIndex)        │    │             │
│              │  │  - OpenAI Embeddings (text-embedding-3-small)│   │             │
│              │  │  - Similarity search (top-k=3)              │    │             │
│              │  └──────────────────┬──────────────────────────┘    │             │
│              │                     │                               │             │
│              │                     ▼                               │             │
│              │  ┌─────────────────────────────────────────────┐    │             │
│              │  │  Layer 2: LLM Response Generation           │    │             │
│              │  │  - OpenAI gpt-4o-mini (temp=0.0)            │    │             │
│              │  │  - Context-grounded answers                 │    │             │
│              │  └─────────────────────────────────────────────┘    │             │
│              └─────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────────┐ │
│  │         Redis           │    │            Knowledge Base                   │ │
│  │  ┌───────────────────┐  │    │  ┌───────────────────────────────────────┐  │ │
│  │  │ Session Storage   │  │    │  │  Documents: JSON, PDF, DOCX, MD, TXT  │  │ │
│  │  │ Query Cache       │  │    │  │  Vector Index (in-memory)             │  │ │
│  │  │ Audio Cache       │  │    │  └───────────────────────────────────────┘  │ │
│  │  └───────────────────┘  │    └─────────────────────────────────────────────┘ │
│  └─────────────────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL AI SERVICES                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐    │
│  │    Groq STT       │  │    OpenAI LLM     │  │       Groq TTS            │    │
│  │  whisper-large-   │  │  gpt-4o-mini      │  │   playai-tts              │    │
│  │  v3-turbo         │  │  (+ embeddings)   │  │   (OpenAI tts-1 fallback) │    │
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Voice Request Flow

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐
│ User │    │ Frontend │    │ LiveKit  │    │   Agent   │    │ AI Services │
└──┬───┘    └────┬─────┘    └────┬─────┘    └─────┬─────┘    └──────┬──────┘
   │             │               │                │                  │
   │ Speaks      │               │                │                  │
   │────────────>│               │                │                  │
   │             │ Audio Stream  │                │                  │
   │             │ (WebRTC)      │                │                  │
   │             │──────────────>│                │                  │
   │             │               │ Audio frames   │                  │
   │             │               │───────────────>│                  │
   │             │               │                │                  │
   │             │               │                │ Transcribe (STT) │
   │             │               │                │─────────────────>│
   │             │               │                │<─────────────────│
   │             │               │                │ "How do I get    │
   │             │               │                │  a passport?"    │
   │             │               │                │                  │
   │             │               │                │──┐ Domain        │
   │             │               │                │  │ Classifier    │
   │             │               │                │<─┘ (Layer 0)     │
   │             │               │                │                  │
   │             │               │                │──┐ RAG Search    │
   │             │               │                │  │ (Layer 1)     │
   │             │               │                │<─┘               │
   │             │               │                │                  │
   │             │               │                │ Generate (LLM)   │
   │             │               │                │─────────────────>│
   │             │               │                │<─────────────────│
   │             │               │                │                  │
   │             │               │                │ Synthesize (TTS) │
   │             │               │                │─────────────────>│
   │             │               │                │<─────────────────│
   │             │               │                │ Audio bytes      │
   │             │               │ Audio response │                  │
   │             │               │<───────────────│                  │
   │             │ Audio Stream  │                │                  │
   │             │<──────────────│                │                  │
   │ Hears       │               │                │                  │
   │<────────────│               │                │                  │
   │             │               │                │                  │
```

### Text Request Flow (with Caching)

```
┌────────┐    ┌─────────┐    ┌───────┐    ┌─────────────┐    ┌─────────────┐
│ Client │    │ FastAPI │    │ Redis │    │ RAG Engine  │    │  Groq TTS   │
└───┬────┘    └────┬────┘    └───┬───┘    └──────┬──────┘    └──────┬──────┘
    │              │             │               │                   │
    │ POST /api/   │             │               │                   │
    │ query/text   │             │               │                   │
    │─────────────>│             │               │                   │
    │              │             │               │                   │
    │              │ Check cache │               │                   │
    │              │────────────>│               │                   │
    │              │             │               │                   │
    │              │      ┌──────┴───────┐       │                   │
    │              │      │  CACHE HIT?  │       │                   │
    │              │      └──────┬───────┘       │                   │
    │              │             │               │                   │
    │              │    ┌────────┴────────┐      │                   │
    │              │    │                 │      │                   │
    │              │  [HIT]             [MISS]   │                   │
    │              │    │                 │      │                   │
    │              │    │                 │ Query│                   │
    │              │    │                 │─────>│                   │
    │              │    │                 │      │                   │
    │              │    │                 │<─────│                   │
    │              │    │                 │Answer│                   │
    │              │    │                 │      │                   │
    │              │    │           Cache │      │                   │
    │              │    │           result│      │                   │
    │              │    │<────────────────│      │                   │
    │              │    │                 │      │                   │
    │              │    │                 │ TTS  │                   │
    │              │    │                 │─────────────────────────>│
    │              │    │                 │<─────────────────────────│
    │              │    │                 │Audio │                   │
    │              │    │                 │      │                   │
    │              │<───┴─────────────────┘      │                   │
    │              │                             │                   │
    │<─────────────│ SSE Stream                  │                   │
    │ (text+audio) │                             │                   │
    │              │                             │                   │
```

### Domain Classification Flow

```
                              User Query
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    DOMAIN CLASSIFIER        │
                    │    (Pydantic AI + GPT-4o)   │
                    └─────────────┬───────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐    ┌─────────────────┐    ┌────────────────┐
   │   GREETING    │    │  IREMBO_SERVICE │    │   OFF_TOPIC    │
   │  "Hi", "Hey"  │    │ Passport, Visa  │    │ Cooking, Math  │
   └───────┬───────┘    └────────┬────────┘    └───────┬────────┘
           │                     │                     │
           ▼                     ▼                     ▼
   ┌───────────────┐    ┌─────────────────┐    ┌────────────────┐
   │    Direct     │    │   RAG Pipeline  │    │    Polite      │
   │   Response    │    │   (Retrieval +  │    │   Redirect     │
   │  "Hello! I'm  │    │    LLM Answer)  │    │  "I specialize │
   │   ARIA..."    │    └─────────────────┘    │   in Irembo.." │
   └───────────────┘                           └────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- Groq API Key

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/aria-agent.git
cd aria-agent
cp .env.example .env
cp livekit/livekit.yaml.example livekit/livekit.yaml
```

Edit `.env` with your API keys:
```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=your-secret-here
```

**IMPORTANT**: Configure `livekit/livekit.yaml` with the **same** API key and secret:

```yaml
# livekit/livekit.yaml
keys:
  devkey: your-secret-here   # Must match: LIVEKIT_API_KEY: LIVEKIT_API_SECRET
```

The format is `API_KEY: API_SECRET`. The key name (`devkey`) must match `LIVEKIT_API_KEY` in `.env`, and the value (`your-secret-here`) must match `LIVEKIT_API_SECRET`.

### 2. Add Knowledge Documents

Place IremboGov JSON documents in `backend/data/knowledge/`:

```bash
cp your-documents/*.json backend/data/knowledge/
```

Supported formats: `.json`, `.md`, `.txt`, `.pdf`, `.docx`

### 3. Start Services

```bash
# Development (single instance)
docker-compose up -d

# Production (with replicas)
docker-compose up -d --scale api=4 --scale agent=4
```

### 4. Access the Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| LiveKit | ws://localhost:7880 |

## API Reference

### Query Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query/text` | POST | Text query -> text + audio response |
| `/api/query/text/stream` | POST | Text query -> SSE streaming response |
| `/api/query/text/stream-with-audio` | POST | Text query -> SSE streaming + TTS audio |
| `/api/query/audio` | POST | Audio upload -> transcribe + respond |
| `/api/query/audio/stream` | POST | Audio upload -> streaming audio response |

#### Streaming Query Example (Recommended)

```bash
curl -X POST http://localhost:8000/api/query/text/stream-with-audio \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I apply for a visa?",
    "session_id": "user123"
  }'
```

SSE Response Events:
```
data: {"type": "session", "session_id": "user123"}
data: {"type": "token", "token": "To apply"}
data: {"type": "token", "token": " for a visa"}
...
data: {"type": "sources", "sources": [...], "confidence": 0.89}
data: {"type": "done", "full_response": "To apply for a visa..."}
data: {"type": "audio", "audio_base64": "UklGRi..."}
```

### Document Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload a document |
| `/api/documents/upload-batch` | POST | Upload multiple documents |
| `/api/documents/refresh` | POST | Rebuild vector index |
| `/api/documents` | GET | List all documents |

> **Important**: After adding or uploading new documents, hit `/api/documents/refresh` to rebuild the vector index and ensure all documents are properly indexed for RAG queries.
>
> ```bash
> curl -X POST http://localhost:8000/api/documents/refresh
> ```

### Stats Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Get all endpoint performance stats |
| `/api/stats/{endpoint}` | GET | Get stats for specific endpoint |
| `/api/stats/history/{endpoint}` | GET | Get request history for endpoint |

## Design Decisions

### Why API-Based Inference (Groq/OpenAI) vs Embedded Models

| Approach | Pros | Cons |
|----------|------|------|
| **API-based (chosen)** | No GPU required, instant scaling, always latest models, zero maintenance | Latency depends on network, usage costs, vendor dependency |
| **Embedded (self-hosted)** | Full control, no per-request costs, works offline | Requires GPU infrastructure, model updates are manual, scaling is complex |

**Decision**: API-based inference via **Groq** (STT/TTS) and **OpenAI** (LLM/embeddings) for these reasons:
1. **Latency**: Groq's custom LPU hardware delivers ~800ms STT latency - faster than most self-hosted solutions
2. **Scalability**: APIs auto-scale to handle 1000+ concurrent requests without infrastructure changes
3. **Cost efficiency**: For a prototype/MVP, pay-per-use beats provisioning GPU servers
4. **Maintenance**: No model serving infrastructure (Triton, vLLM, TGI) to manage

### Why Whisper via Groq (vs other STT options)

| STT Option | Latency | Accuracy | Cost |
|------------|---------|----------|------|
| **Groq Whisper (chosen)** | ~800ms | Excellent (large-v3-turbo) | ~$0.02/min |
| OpenAI Whisper API | ~2-3s | Excellent | ~$0.006/min |
| Self-hosted Whisper | ~1-5s (GPU dependent) | Excellent | Infrastructure cost |
| Google Speech-to-Text | ~1s | Good | ~$0.016/min |
| AssemblyAI | ~1s | Excellent | ~$0.015/min |

**Decision**: Groq's Whisper implementation runs on custom LPU (Language Processing Unit) silicon, achieving **3-4x faster** inference than standard GPU deployments. For a voice-first assistant, minimizing STT latency is critical to user experience.

### Why LlamaIndex (vs LangChain)

| Framework | Strengths | Weaknesses |
|-----------|-----------|------------|
| **LlamaIndex (chosen)** | Purpose-built for RAG, excellent document loaders, native streaming, simple indexing API | Smaller ecosystem, less general-purpose |
| LangChain | Large ecosystem, flexible chains, many integrations | Complex abstractions, overkill for pure RAG, steeper learning curve |

**Decision**: LlamaIndex for these reasons:
1. **RAG-native**: Built specifically for retrieval-augmented generation (our core use case)
2. **Document handling**: Superior multi-format loaders (JSON, PDF, DOCX, MD, TXT) out of the box
3. **Simplicity**: `VectorStoreIndex` + `RetrieverQueryEngine` = complete RAG pipeline in ~20 lines
4. **Streaming**: Native async streaming support for real-time token delivery

### Why In-Memory Vector Index (vs Dedicated Vector DB)

| Vector Store | Use Case | Complexity |
|--------------|----------|------------|
| **In-memory (chosen)** | <100K documents, single-node deployment | Minimal |
| Pinecone | Large scale, managed service | Medium |
| Weaviate | Hybrid search, self-hosted | High |
| Chroma | Local development, persistence | Low |

**Decision**: In-memory LlamaIndex vector store because:
1. **Knowledge base size**: ~50 IremboGov documents fit easily in memory
2. **Simplicity**: No external database to manage
3. **Performance**: Fastest possible retrieval (no network hop)
4. **Upgrade path**: Can migrate to Chroma/Pinecone later if needed

## Handling Ambiguity & Off-Topic Queries

The system uses a **3-layer classification pipeline** to handle ambiguous or off-topic queries before they reach the RAG pipeline:

### Layer 0: Domain Classifier (Pydantic AI)

Every query is first classified by a Pydantic AI agent into one of four categories:

```python
class QueryCategory(str, Enum):
    IREMBO_SERVICE = "irembo_service"  # → Proceed to RAG
    GREETING = "greeting"              # → Direct response
    SMALL_TALK = "small_talk"          # → Direct response
    OFF_TOPIC = "off_topic"            # → Polite redirect
```

**Examples**:
| User Query | Classification | Action |
|------------|----------------|--------|
| "How do I get a passport?" | `IREMBO_SERVICE` | RAG pipeline |
| "Hello!" | `GREETING` | "Hello! I'm ARIA, your assistant for Rwandan government services..." |
| "Thank you" | `SMALL_TALK` | "You're welcome! If you have questions about Irembo services..." |
| "How do I cook rice?" | `OFF_TOPIC` | "I'm specialized in Rwandan government services through Irembo..." |

### Layer 1: Vector Similarity Filtering

For `IREMBO_SERVICE` queries, the RAG pipeline retrieves top-k documents with similarity scoring:
- **Top-k**: 5 documents retrieved
- **Similarity cutoff**: 0.5 (configurable)
- Low-similarity results trigger a "low confidence" warning prefix

### Layer 2: LLM Grounding

The system prompt enforces strict factual grounding:
- LLM must use **only** retrieved context for factual claims
- If information isn't in context, LLM must say "I don't have specific information about that"
- Never invent fees, requirements, processing times, or URLs

### Fallback Behavior

If classification fails (API error, timeout), the system **defaults to RAG** to avoid blocking legitimate queries. False positives (off-topic → RAG) are safer than false negatives (Irembo query → rejected).

## Hallucination Prevention

The system implements multiple safeguards to prevent the LLM from inventing information:

### 1. Temperature = 0.1
Low temperature produces deterministic, conservative outputs - the LLM sticks closely to retrieved context.

### 2. Strict System Prompt
```
## Critical Rules - HALLUCINATION PREVENTION
1. Use information from the provided context - Don't invent factual information
2. If specific information is not in the context, say so
3. Never make up: Fees, Processing times, Required documents, URLs
4. Always cite your sources
```

### 3. Domain Pre-filtering
Off-topic queries never reach the LLM with RAG context, eliminating the risk of the LLM "trying to help" with invented government information.

### 4. Confidence Scoring
```python
if confidence < 0.5:
    return NO_INFORMATION_RESPONSE  # "I don't have info about that"
elif confidence < 0.7:
    answer = LOW_CONFIDENCE_PREFIX + answer  # "Based on available info (not certain)..."
```

### 5. Source Citations
Every response includes source document citations, making it easy to verify information.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | OpenAI gpt-4o-mini (temp=0.1 for determinism) |
| **Domain Classifier** | Pydantic AI + gpt-4o-mini |
| **STT** | Groq whisper-large-v3-turbo |
| **TTS** | Groq playai-tts + OpenAI tts-1 fallback |
| **RAG** | LlamaIndex + OpenAI embeddings |
| **Voice** | LiveKit Agents |
| **Frontend** | Next.js 14 + LiveKit React |
| **Backend** | FastAPI + Gunicorn |
| **Caching** | Redis (session-aware text + audio) |
| **Package Manager** | uv (Python) |

## Project Structure

```
aria-agent/
├── docker-compose.yml          # Full stack orchestration
├── nginx.conf                  # Load balancer config
├── .env.example                # Environment template
│
├── backend/
│   ├── Dockerfile              # Agent worker
│   ├── Dockerfile.api          # API server
│   ├── pyproject.toml          # Python dependencies (uv)
│   ├── agent/
│   │   ├── main.py             # Agent entrypoint
│   │   ├── citizen_agent.py    # Agent with RAG tools
│   │   ├── rag/
│   │   │   ├── domain_classifier.py  # Pydantic AI classifier
│   │   │   ├── loaders.py      # Document loaders
│   │   │   ├── indexer.py      # Vector indexing
│   │   │   ├── query_engine.py # RAG queries
│   │   │   └── prompts.py      # System prompts
│   │   ├── session/
│   │   │   ├── manager.py      # Session management
│   │   │   └── redis_store.py  # Redis persistence
│   │   └── utils/
│   │       └── config.py       # Configuration
│   ├── api/
│   │   ├── server.py           # FastAPI app
│   │   └── routes/
│   │       ├── token.py        # LiveKit tokens
│   │       ├── query.py        # Query endpoints
│   │       ├── documents.py    # Document management
│   │       └── stats.py        # Performance stats
│   └── data/knowledge/         # Knowledge documents
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx            # Main page
│   │   ├── layout.tsx          # App layout
│   │   └── api/                # API routes
│   │       ├── token/route.ts
│   │       └── query/
│   ├── components/
│   │   ├── VoiceAssistant.tsx  # Main voice UI
│   │   ├── TranscriptDisplay.tsx
│   │   ├── TextInput.tsx
│   │   └── ControlBar.tsx
│   └── lib/
│       └── api.ts              # API client
│
└── livekit/
    └── livekit.yaml            # LiveKit server config
```

## Scalability

### Architecture for Horizontal Scaling

```
                         ┌─────────────────────────────┐
                         │        Load Balancer        │
                         │          (Nginx)            │
                         └─────────────┬───────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
   ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
   │  API Replica  │           │  API Replica  │           │  API Replica  │
   │   (Gunicorn   │           │   (Gunicorn   │           │   (Gunicorn   │
   │   4 workers)  │           │   4 workers)  │           │   4 workers)  │
   └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │           Redis             │
                         │    (Shared Session Store)   │
                         └─────────────────────────────┘
```

- **4 API replicas** x 4 Gunicorn workers = 16 processes
- Each async process handles ~100 concurrent I/O requests
- **Total: ~1600 concurrent requests**

### Scaling Commands

```bash
# Scale API and Agent services
docker-compose up -d --scale api=8 --scale agent=8

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api agent
```

## Performance & Caching

### Session-Aware Caching

| Cache Type | Key Pattern | TTL | Description |
|------------|-------------|-----|-------------|
| Text Response | `query_cache:{md5(session:query)}` | 1 hour | Cached RAG answers |
| Audio Response | `audio:query_cache:{md5(session:query)}` | 1 hour | Cached TTS audio |
| Vector Index | In-memory | Permanent | Loaded once, reused |

### Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────┐
│          FIRST REQUEST (Cache Miss) - ~15.5s total          │
├─────────────────────────────────────────────────────────────┤
│ Index Load     │█░░░░░░░░░░░░░░░░░│  ~500ms (cached after) │
│ RAG Retrieval  │█░░░░░░░░░░░░░░░░░│  ~200ms                │
│ LLM Streaming  │█████████████░░░░░│  ~12s                  │
│ TTS Generation │███░░░░░░░░░░░░░░░│  ~2.5s                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           CACHED REQUEST (Cache Hit) - ~1.2s total          │
├─────────────────────────────────────────────────────────────┤
│ Redis Lookup   │░░░░░░░░░░░░░░░░░░│  ~9ms                  │
│ Stream Cache   │████░░░░░░░░░░░░░░│  ~400ms                │
│ Audio Cache    │██████░░░░░░░░░░░░│  ~800ms                │
└─────────────────────────────────────────────────────────────┘
```

### Voice Latency (LiveKit)

```
┌─────────────────────────────────────────────────┐
│              Total: ~1.5s TTFT                  │
├─────────────────────────────────────────────────┤
│ STT (Groq)     │████████░░░░░░░░░░│  ~800ms    │
│ RAG Retrieval  │██░░░░░░░░░░░░░░░░│  ~200ms    │
│ LLM TTFT       │██░░░░░░░░░░░░░░░░│  ~200ms    │
│ TTS Start      │███░░░░░░░░░░░░░░░│  ~350ms    │
└─────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `GROQ_API_KEY` | Groq API key | Yes |
| `LIVEKIT_API_KEY` | LiveKit API key | Yes |
| `LIVEKIT_API_SECRET` | LiveKit secret | Yes |
| `LIVEKIT_URL` | LiveKit server URL | Yes |
| `REDIS_URL` | Redis connection | Yes |

## Development

### Local Development (without Docker)

```bash
# Backend
cd backend
uv sync
uv run python -m agent.main start

# In another terminal
cd backend
uv run uvicorn api.server:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
cd backend
uv run pytest
```

## License

MIT

---

Built with LiveKit, LlamaIndex, Pydantic AI, OpenAI, and Groq.
