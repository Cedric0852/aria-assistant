"""
Citizen Agent - AI Citizen Support Assistant with RAG capabilities.
Uses LiveKit Agents framework for voice interactions.

Integrates domain classifier to filter off-topic queries and ensure
the agent only responds to Irembo government service questions.
"""

import logging
from typing import Optional
from livekit.agents import Agent, function_tool, RunContext

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

from agent.utils.config import settings
from agent.rag.domain_classifier import (
    classify_query,
    QueryCategory,
    OFF_TOPIC_RESPONSE,
)

logger = logging.getLogger("citizen-agent")

# Global index for RAG (initialized once)
_index: Optional[VectorStoreIndex] = None
_session_context: dict = {}


def initialize_rag():
    """Initialize the RAG system with the knowledge base."""
    global _index

    if _index is not None:
        return  # Already initialized

    try:
        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        # Check if knowledge directory exists and has documents
        knowledge_path = settings.KNOWLEDGE_DIR
        if knowledge_path.exists() and any(knowledge_path.iterdir()):
            logger.info(f"Loading documents from {knowledge_path}")
            documents = SimpleDirectoryReader(
                input_dir=str(knowledge_path),
                recursive=True,
                required_exts=[".pdf", ".docx", ".txt", ".md", ".json"],
            ).load_data()

            if documents:
                _index = VectorStoreIndex.from_documents(documents)
                logger.info(f"Loaded {len(documents)} documents into index")
            else:
                logger.warning("No documents found in knowledge directory")
        else:
            logger.warning(f"Knowledge directory not found or empty: {knowledge_path}")

    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        _index = None


class CitizenAgent(Agent):
    """
    AI Citizen Support Assistant agent with RAG capabilities.
    Provides helpful information to citizens by searching a knowledge base.
    """

    def __init__(self) -> None:
        """Initialize the Citizen Agent with system instructions."""
        super().__init__(
            instructions="""You are ARIA (AI Rwanda Irembo Assistant), specialized in Rwandan government services through Irembo.

## YOUR SCOPE - STRICTLY IREMBO SERVICES ONLY:
- Passports, visas, travel documents
- Birth, marriage, death certificates
- Business registration, permits, licenses
- Driving licenses, vehicle registration
- National ID, residence permits
- Government fees, requirements, procedures
- Any service available through https://irembo.gov.rw

## STRICT BOUNDARIES - DO NOT ANSWER:
- Cooking, recipes, food preparation
- Math problems, calculations, coding
- General knowledge, trivia, history
- Health advice, medical questions
- Weather, news, sports
- Stories, jokes, poems
- ANY topic not related to Irembo government services

## Guidelines:
- Always be polite, patient, and respectful
- Use search_knowledge_base tool BEFORE answering any Irembo-related question
- If a query is off-topic, politely redirect to Irembo services
- Keep responses clear and concise
- Always cite sources from the knowledge base
- End responses with: "*This is informational guidance only - for official requirements, visit https://irembo.gov.rw*"

## For OFF-TOPIC queries, respond with:
"I'm specialized in Rwandan government services through Irembo. I can help you with passports, visas, permits, certificates, and other official documents. For this topic, you may want to use a general search engine. Is there anything about Irembo services I can help you with?"

Remember: You ONLY assist with Irembo government services. Politely decline all other requests."""
        )
        # Initialize RAG on first agent creation
        initialize_rag()

    @function_tool
    async def search_knowledge_base(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        Search the knowledge base for relevant information about citizen services,
        policies, procedures, and government information. Use this tool to find
        accurate information before answering citizen questions.

        Args:
            query: The search query to find relevant information.
        """
        global _index, _session_context

        logger.info(f"Searching knowledge base for: {query}")

        # ============================================
        # LAYER 0: Domain Classification
        # Filter out off-topic queries before RAG
        # ============================================
        try:
            classification = await classify_query(query)
            logger.info(
                f"[CITIZEN_AGENT] Query classified as: {classification.category.value} "
                f"(should_use_rag={classification.should_use_rag})"
            )

            # Handle non-Irembo queries
            if not classification.should_use_rag:
                if classification.category == QueryCategory.OFF_TOPIC:
                    logger.info(f"[CITIZEN_AGENT] Filtering off-topic query: '{query[:50]}...'")
                    return OFF_TOPIC_RESPONSE

                # For greetings and small_talk, return direct response
                if classification.category in (QueryCategory.GREETING, QueryCategory.SMALL_TALK):
                    return classification.direct_response or "Hello! How can I help you with Irembo services?"

        except Exception as e:
            logger.warning(f"[CITIZEN_AGENT] Classification failed: {e}, proceeding with RAG")
            # Continue with RAG if classification fails

        # ============================================
        # LAYER 1+: RAG Pipeline (Irembo queries only)
        # ============================================
        if _index is None:
            return ("The knowledge base is currently unavailable. "
                    "I'll do my best to help with general information.")

        try:
            # Create a query engine and search
            query_engine = _index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact",
            )
            response = await query_engine.aquery(query)

            if response and response.response:
                # Store context for follow-up questions
                _session_context["last_query"] = query
                _session_context["last_response"] = str(response.response)

                # Format response with source information
                sources = []
                if response.source_nodes:
                    for node in response.source_nodes:
                        if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                            sources.append(node.metadata['file_name'])

                result = str(response.response)
                if sources:
                    unique_sources = list(set(sources))
                    result += f"\n\n[Sources: {', '.join(unique_sources)}]"

                return result
            else:
                return "I couldn't find specific information about that in the knowledge base."

        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return "I encountered an error while searching. Let me try to help with general information."

    @function_tool
    async def get_conversation_context(self, context: RunContext) -> str:
        """
        Get conversation history and context from the current session.
        Use this to recall previous questions and answers in the conversation.
        """
        global _session_context

        if not _session_context:
            return "No previous conversation context available."

        context_parts = []
        if "last_query" in _session_context:
            context_parts.append(f"Last question: {_session_context['last_query']}")
        if "last_response" in _session_context:
            # Truncate long responses
            response = _session_context['last_response']
            if len(response) > 500:
                response = response[:500] + "..."
            context_parts.append(f"Last answer: {response}")

        return "\n".join(context_parts) if context_parts else "No context available."
