"""
Pydantic AI Domain Classifier for ARIA.

This module provides an intelligent query classifier using Pydantic AI
with OpenAI's reasoning model (gpt-5-nano) to determine if queries
are related to Irembo government services or should be handled differently.
"""

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

logger = logging.getLogger(__name__)

# Model configuration - using gpt-5-nano with reasoning for fast classification
CLASSIFIER_MODEL = "gpt-5-nano-2025-08-07"
REASONING_EFFORT = "low"  # Fast classification


class QueryCategory(str, Enum):
    """Categories for user queries."""
    IREMBO_SERVICE = "irembo_service"  # Questions about Irembo government services
    GREETING = "greeting"              # Hi, Hello, Good morning, etc.
    SMALL_TALK = "small_talk"          # Thank you, Bye, Who are you, How are you
    OFF_TOPIC = "off_topic"            # Cooking, math, coding, general knowledge


class ClassificationResult(BaseModel):
    """Structured output from the domain classifier."""
    category: QueryCategory = Field(
        description="The category of the user's query"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="Direct response for greetings/small_talk (None for irembo_service or off_topic)"
    )
    should_use_rag: bool = Field(
        description="Whether the query should be processed by the RAG pipeline"
    )


# System prompt for the classifier agent
CLASSIFIER_SYSTEM_PROMPT = """You are ARIA's query classifier. Your ONLY job is to classify user queries into categories.

## Categories:
1. **irembo_service**: Questions about Rwandan government services through Irembo
   - Passports, visas, travel documents
   - Birth, marriage, death certificates
   - Business registration, permits, licenses
   - Driving licenses, vehicle registration
   - National ID, residence permits
   - Government fees, requirements, procedures
   - ANY question mentioning "Irembo", "Rwanda", or government services

2. **greeting**: Simple greetings
   - "Hi", "Hello", "Hey", "Good morning/afternoon/evening"
   - "Muraho" (Kinyarwanda greeting), "Bonjour" (French)

3. **small_talk**: Basic conversational exchanges
   - "Thank you", "Thanks", "Murakoze"
   - "Bye", "Goodbye", "See you"
   - "Who are you?", "What can you do?"
   - "How are you?"

4. **off_topic**: Everything NOT about Irembo services
   - Cooking, recipes, food
   - Math, calculations, coding, programming
   - General knowledge, trivia, history
   - Health advice, medical questions
   - Weather, news, sports
   - Stories, jokes, poems
   - Any other non-government topic

## Rules:
- If ANY part of the query mentions government services, passports, visas, permits → irembo_service
- For greetings and small_talk, provide a friendly direct_response
- Be STRICT: cooking rice is off_topic, even if phrased as a question
- should_use_rag = true ONLY for irembo_service category
"""

# Greeting responses
GREETING_RESPONSES = {
    "en": "Hello! I'm ARIA, your assistant for Rwandan government services through Irembo. How can I help you today?",
    "rw": "Muraho! Ndi ARIA, umufasha wawe ku mirimo ya Leta y'u Rwanda binyuze muri Irembo. Nakubasha iki uyu munsi?",
    "fr": "Bonjour! Je suis ARIA, votre assistant pour les services gouvernementaux rwandais via Irembo. Comment puis-je vous aider?",
}

# Small talk responses
SMALL_TALK_RESPONSES = {
    "thank_you": "You're welcome! If you have any questions about Irembo government services, feel free to ask.",
    "goodbye": "Goodbye! Feel free to return if you have questions about Rwandan government services.",
    "who_are_you": "I'm ARIA (AI Rwanda Irembo Assistant), created by Cedric (@Cedric0852). I help citizens with questions about government services available through Irembo.",
    "how_are_you": "I'm doing well, thank you for asking! I'm here to help you with Irembo government services. What would you like to know?",
}

# Off-topic response
OFF_TOPIC_RESPONSE = """I'm specialized in Rwandan government services through Irembo and can help you with passports, visas, permits, certificates, and other official documents.

For this topic, you may want to refer to general search engines or specialized resources.

If you have any questions about Irembo government services, feel free to ask!

*This is informational guidance only - for official requirements, visit https://irembo.gov.rw*"""


def _create_classifier_agent() -> Agent[None, ClassificationResult]:
    """Create the Pydantic AI classifier agent with reasoning model."""
    model = OpenAIResponsesModel(CLASSIFIER_MODEL)
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort=REASONING_EFFORT,
        openai_reasoning_summary="concise",
    )

    return Agent(
        model,
        model_settings=settings,
        output_type=ClassificationResult,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    )


# Singleton agent instance
_classifier_agent: Optional[Agent[None, ClassificationResult]] = None


def get_classifier_agent() -> Agent[None, ClassificationResult]:
    """Get or create the singleton classifier agent."""
    global _classifier_agent
    if _classifier_agent is None:
        _classifier_agent = _create_classifier_agent()
    return _classifier_agent


async def classify_query(query: str) -> ClassificationResult:
    """
    Classify a user query using the Pydantic AI reasoning agent.

    This is Layer 0 of the hallucination prevention pipeline.
    It uses GPT-5-nano with reasoning to intelligently classify queries
    before any RAG retrieval happens.

    Args:
        query: The user's input query

    Returns:
        ClassificationResult with category, reasoning, and optional direct response
    """
    agent = get_classifier_agent()

    try:
        result = await agent.run(query)
        classification = result.output

        # Log the classification
        logger.info(
            f"Query classified: '{query[:50]}...' -> {classification.category.value} "
            f"(should_use_rag={classification.should_use_rag})"
        )

        return classification

    except Exception as e:
        logger.error(f"Classification failed for query '{query}': {e}")
        # Fallback: assume it might be Irembo-related and let RAG handle it
        return ClassificationResult(
            category=QueryCategory.IREMBO_SERVICE,
            reasoning="Classification failed, defaulting to RAG pipeline",
            direct_response=None,
            should_use_rag=True,
        )


def classify_query_sync(query: str) -> ClassificationResult:
    """
    Synchronous version of classify_query for non-async contexts.

    Args:
        query: The user's input query

    Returns:
        ClassificationResult with category, reasoning, and optional direct response
    """
    import asyncio
    return asyncio.run(classify_query(query))
