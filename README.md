# ARIA - AI-Powered Citizen Support Assistant

A Voice-First Service Assistant that enables citizens to ask questions (via voice or text) about government services and receive accurate, context-aware guidance powered by RAG (Retrieval-Augmented Generation).

## Branches

| Branch | Description | Status |
|--------|-------------|--------|
| **main** | Streamlit chat interface using the ARIA API | Deployed |
| **aria-agent** | Advanced features: LiveKit integration with automatic voice detection for real-time conversations | Local only |

> **Note:** The `aria-agent` branch includes a more advanced implementation with **LiveKit-powered live conversations** featuring automatic voice activity detection (VAD) and turn-taking. Due to infrastructure limitations, this branch is not deployed online but works locally. Feel free to test it!

## Live Demo

| Service | URL |
|---------|-----|
| **Chat Interface** | [aria.lunaroot.rw](https://aria.lunaroot.rw/) |
| **API Documentation** | [aria-api.lunaroot.rw/docs](https://aria-api.lunaroot.rw/docs) |

## System Architecture

```
═══════════════════════════════════════════════════════════════════════════════
                            ARIA SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

                              ┌─────────────────┐
                              │    CITIZEN      │
                              │                 │
                              │  Voice / Text   │
                              └────────┬────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    FRONTEND (Streamlit)                          │
        │                                                                  │
        │   Chat Interface (app.py)  │     Stats Dashboard (pages/stats.py)│
        │   ─────────────────────    │     ────────────────────────────    │
        │   • Text input             │     • Response time metrics         │
        │   • Audio recording        │     • Cache hit rates               │
        │   • Message history        │     • Performance visualization     │
        │   • Source citations       │     • System health monitoring      │
        │   • Audio playback (TTS)   │                                     │
        └──────────────────────────────────────────────────────────────────┘
                                       │
                                       │ HTTP / SSE
                                       ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    BACKEND API (FastAPI + Gunicorn)              │
        │                         api/server.py                            │
        │                                                                  │
        │   ┌─────────────────────────────────────────────────────────┐    │
        │   │                    API ROUTES                           │    │
        │   │   routes/query.py      routes/documents.py              │    │
        │   │   ─────────────────    ────────────────────             │    │
        │   │   POST /api/query/     POST /api/documents/upload       │    │
        │   │     text               POST /api/documents/refresh      │    │
        │   │   POST /api/query/     GET  /api/documents              │    │
        │   │     text/stream        DELETE /api/documents/{id}       │    │
        │   │   POST /api/query/                                      │    │
        │   │     audio              routes/health.py  routes/stats.py│    │
        │   │                        GET /health       GET /api/stats │    │
        │   └─────────────────────────────────────────────────────────┘    │
        │                              │                                   │
        │                              ▼                                   │
        │   ┌─────────────────────────────────────────────────────────┐    │
        │   │              AGENT LAYER (agent/)                       │    │
        │   │                                                         │    │
        │   │   ┌─────────────────────────────────────────────────┐   │    │
        │   │   │  Domain Classifier (rag/domain_classifier.py)   │   │    │
        │   │   │  • Pydantic AI + GPT-4o-mini                    │   │    │
        │   │   │  • Routes: irembo_service → RAG pipeline        │   │    │
        │   │   │           greeting/small_talk → Direct response │   │    │
        │   │   │           off_topic → Polite decline            │   │    │
        │   │   └─────────────────────────────────────────────────┘   │    │
        │   │                         │                               │    │
        │   │                         ▼                               │    │
        │   │   ┌─────────────────────────────────────────────────┐   │    │
        │   │   │        RAG PIPELINE (LlamaIndex)                │   │    │
        │   │   │                                                 │   │    │
        │   │   │   rag/loaders.py ─► rag/indexer.py              │   │    │
        │   │   │   (JSON/MD/PDF/     (Vector Store +             │   │    │
        │   │   │    DOCX/TXT)         OpenAI Embeddings)         │   │    │
        │   │   │         │                  │                    │   │    │
        │   │   │         └────────┬─────────┘                    │   │    │
        │   │   │                  ▼                              │   │    │
        │   │   │   rag/query_engine.py ◄── rag/prompts.py        │   │    │
        │   │   │   (GPT-4o-mini, top_k=5,  (QA & Refine          │   │    │
        │   │   │    response_mode=refine)   templates)           │   │    │
        │   │   └─────────────────────────────────────────────────┘   │    │
        │   │                                                         │    │
        │   │   session/redis_store.py    utils/config.py             │    │
        │   │   (Session management)      (Environment config)        │    │
        │   └─────────────────────────────────────────────────────────┘    │
        └──────────────────────────────────────────────────────────────────┘
                       │                               │
                       ▼                               ▼
        ┌─────────────────────────┐     ┌─────────────────────────┐
        │   EXTERNAL AI SERVICES  │     │   SESSION & CACHE       │
        │                         │     │       (Redis)           │
        │   ┌───────────────────┐ │     │                         │
        │   │ OpenAI API        │ │     │   Session Store         │
        │   │ • GPT-4o-mini     │ │     │   ─────────────         │
        │   │   (LLM + Class.)  │ │     │   Chat history (30m)    │
        │   │ • Embeddings      │ │     │                         │
        │   │   (text-embed-3)  │ │     │   Query Cache           │
        │   │ • gpt-4o-mini-tts │ │     │   ───────────           │
        │   │   (streaming TTS) │ │     │   Responses (1hr)       │
        │   └───────────────────┘ │     │                         │
        │                         │     │   Embedding Cache       │
        │   ┌───────────────────┐ │     │   ───────────────       │
        │   │ Groq API          │ │     │   Vectors (24hr)        │
        │   │ • Whisper STT     │ │     │                         │
        │   │   (large-v3-turbo)│ │     │   Audio Cache           │
        │   │ • Orpheus TTS     │ │     │   ───────────           │
        │   │   (fallback only) │ │     │   TTS output (1hr)      │
        │   └───────────────────┘ │     │                         │
        └─────────────────────────┘     └─────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────┐
        │   KNOWLEDGE BASE        │
        │   data/knowledge/       │
        │                         │
        │   • IremboGov JSON docs │
        │   • Immigration services│
        │   • Passport/Visa info  │
        │                         │
        │   storage/index/        │
        │   • Persisted vectors   │
        └─────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

## Request Flow

```
═══════════════════════════════════════════════════════════════════════════════
                              REQUEST FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════


  TEXT QUERY FLOW
  ───────────────

                    ┌──────────────┐
                    │  User Input  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐         ┌─────────────────────┐
                    │ Cache Check  │───Yes──►│ Return Cached       │
                    └──────┬───────┘         │ Response + TTS      │
                           │ No              └─────────────────────┘
                           ▼
                    ┌──────────────┐
                    │Vector Search │
                    │  (top_k=5)   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │LLM Generate  │
                    │(GPT-4o-mini) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Cache Response│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │TTS Generation│
                    │(Groq/OpenAI) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Return to User│
                    └──────────────┘


  AUDIO QUERY FLOW
  ────────────────

    ┌────────────┐      ┌─────────────┐      ┌───────────────┐      ┌──────────────┐
    │ Audio File │─────►│Groq Whisper │─────►│ Transcribed   │─────►│ Text Query   │
    └────────────┘      │    STT      │      │    Text       │      │    Flow      │
                        └─────────────┘      └───────────────┘      └──────┬───────┘
                                                                          │
                                                                          ▼
                                                                   ┌──────────────┐
                                                                   │Response +    │
                                                                   │Transcript +  │
                                                                   │Audio Output  │
                                                                   └──────────────┘


  STREAMING RESPONSE (SSE) - WITH STREAMING AUDIO
  ─────────────────────────────────────────────────

    Query ──► Session ──► Tokens ──► Done ──► Sources ──► Audio Stream
                │            │          │         │            │
                ▼            ▼          ▼         ▼            ▼
           session_id   "tok","en"  full_resp  sources[]   audio_start
                                                            audio_chunk (×N)
                                                            audio_end

    Event Order:
    1. session      → Session ID for conversation tracking
    2. tokens       → Streaming text response (multiple events)
    3. done         → Text complete with full_response
    4. sources      → Source documents with relevance scores
    5. audio_start  → TTS streaming begins (sample_rate, channels, bits)
    6. audio_chunk  → PCM audio chunks (base64, streamed as generated)
    7. audio_end    → Audio streaming complete


═══════════════════════════════════════════════════════════════════════════════
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit 1.40+ | Chat UI with voice input |
| **Backend** | FastAPI + Gunicorn | RESTful API server |
| **RAG Framework** | LlamaIndex | Document indexing & retrieval |
| **Domain Classifier** | Pydantic AI + GPT-4o-mini | Query classification |
| **LLM** | OpenAI GPT-4o-mini | Response generation |
| **Embeddings** | OpenAI text-embedding-3-small | Vector embeddings |
| **STT** | Groq Whisper large-v3-turbo | Speech-to-text |
| **TTS** | OpenAI gpt-4o-mini-tts (streaming) | Text-to-speech with streaming audio |
| **TTS Fallback** | Groq Orpheus | Fallback when OpenAI unavailable |
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

**Included Knowledge Base:** IremboGov Immigration and Emigration services documentation (passport applications, visas, travel documents, etc.)

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

SSE Events (with streaming audio):
```
event: session
data: {"type": "session", "session_id": "abc123"}

event: token
data: {"type": "token", "token": "You"}

event: done
data: {"type": "done", "full_response": "You need..."}

event: sources
data: {"type": "sources", "sources": [...], "confidence": 0.85}

event: audio_start
data: {"type": "audio_start", "sample_rate": 24000, "channels": 1, "bits_per_sample": 16}

event: audio_chunk
data: {"type": "audio_chunk", "chunk": "<base64 PCM data>", "index": 0}

event: audio_chunk
data: {"type": "audio_chunk", "chunk": "<base64 PCM data>", "index": 1}
... (multiple chunks)

event: audio_end
data: {"type": "audio_end", "total_chunks": 42}
```

**Note:** Audio is streamed as PCM chunks (24kHz, 16-bit, mono) for low-latency playback. The frontend combines chunks and adds a WAV header for playback.

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

## Key Concerns Addressed

### 1. Architecture for Inference (Embedded vs. API Calls)

**Decision: API-based inference for all AI models**

| Model | Provider | Rationale |
|-------|----------|-----------|
| **STT** | Groq API (Whisper) | 10x faster than local, no GPU required |
| **LLM** | OpenAI API (GPT-4o-mini) | Consistent quality, managed scaling |
| **TTS** | OpenAI API (gpt-4o-mini-tts) | Streaming audio support, lower latency |
| **TTS Fallback** | Groq API (Orpheus) | Fallback when OpenAI unavailable (rate-limited on free tier) |
| **Embeddings** | OpenAI API | Cached in Redis to minimize API calls |

**Why not embedded models?**
- Eliminates GPU infrastructure costs
- Faster cold starts (no model loading)
- Automatic scaling by provider
- Focus on application logic, not ML ops

### 2. Handling Ambiguity

**Vague queries:**
- Confidence scoring (0.0-1.0) based on document similarity
- Users see High/Medium/Low confidence badges
- Sources displayed with relevance percentages

**Out-of-scope queries:**
- System prompt explicitly instructs: "If the question is unrelated to government services, politely explain you can only help with Irembo services"
- RAG retrieval returns low scores for irrelevant queries
- No hallucination: answers grounded strictly in retrieved documents

**Vague audio:**
- Whisper provides transcription confidence
- If transcription unclear, user sees transcript to verify
- Follow-up question detection uses conversation history

### 3. Latency Optimization

| Strategy | Impact | Implementation |
|----------|--------|----------------|
| **Response Streaming** | 2-3s TTFT vs 5-8s wait | SSE events stream tokens |
| **Streaming TTS** | Faster time-to-first-audio | OpenAI gpt-4o-mini-tts streams PCM chunks |
| **Multi-layer Caching** | 10-35x speedup | Redis: queries (1hr), embeddings (24hr), audio (1hr) |
| **Connection Pooling** | -50ms per request | Reused Redis/HTTP connections |
| **Async Everything** | Higher throughput | FastAPI async, async Redis, async LLM |

### 4. Scalability (1,000+ Concurrent Users)

**Current architecture (prototype scale):**
- Single Redis instance (sufficient for demo/prototype)
- Gunicorn with 4 workers = ~100 concurrent per container
- Stateless API = horizontal scaling ready

**Future scaling with Kubernetes HPA:**
```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │  (Traefik/Ingress)  │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │ API Pod 1 │       │ API Pod 2 │       │ API Pod N │
    │(4 workers)│       │(4 workers)│       │(4 workers)│
    └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Redis Cluster     │
                    │   (3+ nodes)        │
                    └─────────────────────┘

    * Kubernetes HPA auto-scales pods based on CPU/memory
    * Currently using Gunicorn workers for concurrency
```

**Scaling steps (future):**
1. Deploy 10+ API containers behind load balancer
2. Migrate Redis to cluster mode (3+ nodes)
3. Add Pinecone/Qdrant for distributed vector search
4. Implement rate limiting (10 req/min per session)
5. Add CDN for static assets

### 5. Accuracy & Hallucination Prevention

> **✅ Production Tested**: The Pydantic AI domain classifier successfully blocks off-topic queries (e.g., "how to cook rice", "what is 2+2") while allowing Irembo government service questions through to the RAG pipeline.

Preventing hallucination is critical for a government services assistant. ARIA implements a multi-layer defense:

```
═══════════════════════════════════════════════════════════════════════════════
                        HALLUCINATION PREVENTION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

  User Query: "How to cook rice?"
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 0: Pydantic AI Domain Classifier (gpt-4o-mini)                    │
  │  ─────────────────────────────────────────────────────────              │
  │  • Uses OpenAI gpt-4o-mini for fast structured classification           │
  │  • Structured output: QueryCategory (irembo_service/greeting/off_topic) │
  │  • Handles greetings & small talk with direct responses (no RAG)        │
  │  • Off-topic → Polite decline; Irembo queries → Continue to RAG         │
  │  • Fallback to pattern matching if classifier fails                     │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 1: Retrieval Filtering (SimilarityPostprocessor)                 │
  │  ─────────────────────────────────────────────────────────              │
  │  • Retrieve top 5 documents from vector store                           │
  │  • Filter out documents with similarity < 40%                           │
  │  • If all filtered → LLM receives EMPTY context                         │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 2: Strict Prompt Engineering (QA_PROMPT_TEMPLATE)                │
  │  ─────────────────────────────────────────────────────────              │
  │  • 2-step evaluation: Is it about Irembo? Does context answer it?       │
  │  • Automatic decline triggers for cooking, math, coding, etc.           │
  │  • "If in doubt, DECLINE" instruction                                   │
  │  • FORBIDDEN: Using training knowledge, guessing, being "helpful"       │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 3: Refine Response Mode (REFINE_PROMPT_TEMPLATE)                 │
  │  ─────────────────────────────────────────────────────────              │
  │  • Process each context chunk sequentially                              │
  │  • If existing answer DECLINES → Keep the decline                       │
  │  • Only refine with DIRECTLY relevant Irembo information                │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER 4: Post-Generation Validation                                    │
  │  ─────────────────────────────────────────────────────────              │
  │  • Check max source score after LLM response                            │
  │  • If max_score < 40% → Replace with NO_INFORMATION_RESPONSE            │
  │  • If max_score < 60% → Add LOW_CONFIDENCE_PREFIX                       │
  │  • Don't return sources if query rejected (prevents confusion)          │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Result: "I can only help with questions about Irembo government services."

═══════════════════════════════════════════════════════════════════════════════
```

**Current Implementation:**
| Layer | Technique | Purpose |
|-------|-----------|---------|
| **Layer 0** | Pydantic AI + `gpt-4o-mini` | Intelligent domain classification before RAG |
| **Layer 1** | `SimilarityPostprocessor(cutoff=0.6)` | Filter irrelevant documents before LLM |
| **Layer 2** | Strict QA template with decline triggers | Force LLM to refuse off-topic questions |
| **Layer 3** | `response_mode="refine"` | Careful multi-step response generation |
| **Layer 4** | Max score threshold check | Reject low-confidence answers post-generation |

**Pydantic AI Classifier Details (✅ Production Tested):**
```python
# domain_classifier.py - Uses OpenAI gpt-4o-mini for fast structured classification
# Tested: Successfully blocks "how to cook rice", "what is 2+2", etc.
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel("gpt-4o-mini")

class QueryCategory(Enum):
    IREMBO_SERVICE = "irembo_service"  # → Continue to RAG pipeline
    GREETING = "greeting"              # → Direct friendly response
    SMALL_TALK = "small_talk"          # → Direct response (thank you, bye, etc.)
    OFF_TOPIC = "off_topic"            # → Polite decline with redirect
```

### 6. Future: Agentic RAG Architecture

**Current Limitations of Simple RAG:**
- ~~No reasoning step before answering~~ ✅ Pydantic AI classifier (Layer 0) classifies queries before RAG
- ~~Off-topic queries bypass domain filtering~~ ✅ Classifier integrated in both streaming & non-streaming endpoints
- Cannot decompose complex multi-part questions
- Limited ability to handle follow-up questions with context
- Single-shot retrieval without query refinement

**Reference Implementation: [RWAKA.rw](https://rwaka.rw)** - An advanced Agentic RAG system for Rwandan legal precedents that demonstrates reasoning capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RWAKA EXAMPLE: Agentic RAG with Reasoning                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User: "how to cook rice"                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Reasoning                                                          │   │
│  │  ──────────                                                         │   │
│  │  STEP 1                                                             │   │
│  │                                                                     │   │
│  │  I'm specialized in Rwandan legal precedents and can help retrieve  │   │
│  │  and analyze relevant caselaws. For cooking or culinary questions,  │   │
│  │  you may want to refer to cooking guides or online recipes. If you  │   │
│  │  have any legal-related queries, especially concerning Rwandan      │   │
│  │  law, feel free to ask!                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Key Features:                                                              │
│  • Explicit reasoning step visible to user                                  │
│  • Domain classification before retrieval                                   │
│  • Graceful decline with redirection                                        │
│  • Maintains professional, helpful tone                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Future Agentic RAG Improvements for ARIA:**

```
═══════════════════════════════════════════════════════════════════════════════
                        AGENTIC RAG ARCHITECTURE (FUTURE)
═══════════════════════════════════════════════════════════════════════════════

  User Query: "What documents do I need for a passport, and how much does it cost?"
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  QUERY ANALYSIS AGENT                                                   │
  │  ─────────────────────                                                  │
  │  • Decompose into sub-questions:                                        │
  │    1. "What documents are required for passport application?"           │
  │    2. "How much does a passport cost in Rwanda?"                        │
  │  • Classify query type: factual, procedural, comparison                 │
  │  • Detect if follow-up question (use conversation context)              │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  ┌────────────┐        ┌────────────┐        ┌────────────┐
  │ Sub-Query  │        │ Sub-Query  │        │ Metadata   │
  │ Retriever  │        │ Retriever  │        │ Filter     │
  │ (docs)     │        │ (fees)     │        │ Agent      │
  └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RESPONSE SYNTHESIS AGENT                                               │
  │  ─────────────────────────                                              │
  │  • Combine sub-answers coherently                                       │
  │  • Cross-reference for consistency                                      │
  │  • Add source citations per claim                                       │
  └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FAITHFULNESS EVALUATOR                                                 │
  │  ─────────────────────────                                              │
  │  • Check each claim against source documents                            │
  │  • Score: faithfulness (0-1), relevancy (0-1)                           │
  │  • If faithfulness < 0.8 → Regenerate or decline                        │
  └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

**Recommended Future Enhancements:**

| Enhancement | Technology | Benefit |
|-------------|------------|---------|
| **Agentic Reasoning** | Claude/GPT-4 with CoT prompting | Visible reasoning steps like RWAKA (domain check → retrieval decision → response) |
| **Sub-Question Decomposition** | LlamaIndex `SubQuestionQueryEngine` | Handle complex multi-part questions |
| **Metadata Filtering** | `AutoRetriever` with filters | Route queries to relevant document types |
| **Reranking** | Cohere Rerank / BGE Reranker | Improve retrieval precision |
| **Faithfulness Evaluation** | LlamaIndex `FaithfulnessEvaluator` | Verify claims against sources |
| **Hybrid Search** | BM25 + Vector (Pinecone/Qdrant) | Better keyword + semantic matching |
| **Query Routing** | `RouterQueryEngine` | Different strategies for different query types |
| **Conversation Memory** | `ChatMemoryBuffer` | Better follow-up question handling |
| **Self-Correction** | Agent loop with reflection | Retry with refined query if confidence low |

**Implementation Priority:**
1. **High:** Agentic Reasoning + Reranking (transparency + accuracy)
2. **Medium:** Sub-question decomposition + Faithfulness Evaluation
3. **Lower:** Full agentic loop with self-correction (requires more compute)

---

### 7. Data Privacy (Public Sector Context)

**Data Privacy:**
- Session IDs are auto-generated UUIDs (not tied to personal identity)
- Chat history accessible only via session ID (no user accounts)
- Redis protected by password authentication
- Sessions expire after 30 minutes, chat history after 7 days
- Audio files processed in-memory, not persisted to disk
- API keys stored in environment variables, never in code
- *Future improvements: encryption at rest, audit logging*

---

## Design Choices

### Why These Technologies?

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **STT Model** | Groq Whisper | 10x faster than local Whisper, near real-time transcription |
| **LLM** | GPT-4o-mini | Best cost/performance ratio, fast inference |
| **RAG Framework** | LlamaIndex | Better RAG abstractions than LangChain for document Q&A |
| **Embeddings** | text-embedding-3-small | Good accuracy, lower cost, 1536 dimensions |
| **TTS** | OpenAI gpt-4o-mini-tts | Streaming audio support, faster time-to-first-audio |
| **TTS Fallback** | Groq Orpheus | Fallback only (heavily rate-limited on free tier: 3600 TPD) |
| **Vector Store** | LlamaIndex In-Memory | Sufficient for prototype scale, persistent to disk |
| **Cache** | Redis | Fast, supports TTL, production-proven |

### Latency Optimization Strategies

1. **Response Streaming (SSE)**: Tokens stream as generated, reducing perceived latency
2. **Streaming TTS**: Audio chunks stream as generated using OpenAI gpt-4o-mini-tts, faster time-to-first-audio
3. **Multi-Layer Caching**:
   - Embedding cache (24hr TTL) - avoid redundant API calls
   - Query cache (1hr TTL) - instant responses for repeated queries
   - Audio cache (1hr TTL) - skip TTS for cached responses
4. **Connection Pooling**: Redis and HTTP clients maintain connection pools
5. **Async Everything**: FastAPI async handlers, async Redis, async LLM calls

### Performance Benchmarks

| Scenario | First Response | Cached Response | Speedup |
|----------|----------------|-----------------|---------|
| Text only | 2-5s | 50-200ms | 10-25x |
| Text + Audio | 7-15s | 100-500ms | 15-30x |
| Audio input + Audio output | 10-35s | 500ms-2s | 15-35x |
| Streaming text | 2-3s TTFT | <100ms TTFT | 20-30x |

*TTFT = Time to First Token*

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
| **No Hallucination** | Done | Pydantic AI classifier (Layer 0) + RAG grounding + strict prompts |
| **Production Quality** | Done | Docker, health checks, logging, error handling |
| **Architecture Diagram** | Done | ASCII diagrams in this README |
| **Dockerized Setup** | Done | docker-compose.yml and docker-compose.prod.yml |

## Project Structure

```
aria-assistant/
│
├─── backend/
│    │
│    ├─── agent/                          # Core AI Agent Layer
│    │    │
│    │    ├─── rag/                       # RAG Pipeline Components
│    │    │    ├── __init__.py
│    │    │    ├── domain_classifier.py   # Pydantic AI query classifier (GPT-4o-mini)
│    │    │    ├── indexer.py             # Vector index management & persistence
│    │    │    ├── loaders.py             # Document loaders (JSON/MD/PDF/DOCX/TXT)
│    │    │    ├── query_engine.py        # RAG query engine with streaming
│    │    │    └── prompts.py             # QA & Refine prompt templates
│    │    │
│    │    ├─── session/                   # Session Management
│    │    │    ├── __init__.py
│    │    │    └── redis_store.py         # Redis-backed session & cache store
│    │    │
│    │    └─── utils/                     # Utilities
│    │         ├── __init__.py
│    │         └── config.py              # Environment configuration
│    │
│    ├─── api/                            # FastAPI Application
│    │    ├── __init__.py
│    │    ├── server.py                   # FastAPI app entry point
│    │    │
│    │    └─── routes/                    # API Endpoints
│    │         ├── __init__.py
│    │         ├── query.py               # /api/query/* (text, audio, stream)
│    │         ├── documents.py           # /api/documents/* (upload, list, delete)
│    │         ├── health.py              # /health endpoint
│    │         ├── stats.py               # /api/stats endpoint
│    │         └── token.py               # Token validation
│    │
│    ├─── data/
│    │    └─── knowledge/                 # Knowledge base documents (43 JSON files)
│    │
│    ├─── storage/
│    │    └─── index/                     # Persisted vector index
│    │
│    ├── Dockerfile.api                   # Development container
│    ├── Dockerfile.api.prod              # Production container
│    ├── gunicorn.conf.py                 # Gunicorn server config
│    └── pyproject.toml                   # Python dependencies
│
├─── frontend-streamlit/
│    ├── app.py                           # Main chat interface
│    │
│    ├─── pages/
│    │    └── stats.py                    # Performance stats dashboard
│    │
│    ├─── .streamlit/
│    │    └── config.toml                 # Streamlit configuration
│    │
│    ├── requirements.txt                 # Frontend dependencies
│    ├── Dockerfile                       # Development container
│    └── Dockerfile.prod                  # Production container
│
├── docker-compose.yml                    # Development environment
├── docker-compose.prod.yml               # Production environment
├── .env.example.local                    # Local env template
├── .env.example.prod                     # Production env template
├── .gitignore
├── LICENSE                               # MIT License
└── README.md                             # This file
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

## Latency Highlights

The caching system delivers **exceptional performance gains**:

```
                    CACHE SPEEDUP VISUALIZATION

    First Request:     |████████████████████████████████████| 5-15s
    Cached Request:    |██|                                   50-500ms

    Improvement:       10-35x FASTER
```

- **Text queries**: From 2-5s down to **50-200ms** (25x faster)
- **Voice + TTS**: From 10-35s down to **500ms-2s** (35x faster)
- **Streaming TTFT**: From 2-3s down to **<100ms** (30x faster)

This means returning users get **near-instant responses** for previously asked questions.

---

## Future Considerations

### Short-term Improvements
- [ ] Add Kinyarwanda language support (STT/TTS)
- [ ] Implement conversation memory summarization for longer contexts
- [ ] Add rate limiting per session to prevent abuse
- [ ] WebSocket support for real-time bidirectional audio

### Medium-term Enhancements
- [ ] Multi-tenant support (multiple government departments)
- [ ] Admin dashboard for knowledge base management
- [ ] Analytics dashboard for citizen query patterns
- [ ] A/B testing framework for response quality

### Long-term Vision
- [ ] On-premise deployment option for sensitive data
- [ ] Fine-tuned model for Rwandan government terminology
- [ ] Integration with actual Irembo service APIs
- [ ] Mobile app with offline-first architecture
- [ ] Multi-language support (French, Swahili)

### Security Hardening
- [ ] Encryption at rest for Redis data
- [ ] Audit logging for all queries
- [ ] PII detection and redaction
- [ ] SOC 2 compliance preparation

---

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Built by [@Cedric0852](https://github.com/Cedric0852)
- AI-Powered Citizen Support Assistant take-home assignment
- Designed for Rwanda's Irembo e-government platform context
