"""
Agent worker entrypoint for the Citizen Support Assistant.
Configures LiveKit agents with STT, LLM, and TTS providers.
"""

import logging
from livekit.agents import (
    AgentSession,
    AgentServer,
    JobContext,
    RoomInputOptions,
    cli,
)
from livekit.plugins import silero, openai, groq

from agent.citizen_agent import CitizenAgent
from agent.utils.config import settings

logger = logging.getLogger("citizen-agent")
logger.setLevel(logging.INFO)

# Prewarm VAD model at module level for single process mode
_vad = None


def get_vad():
    """Get or create the VAD model."""
    global _vad
    if _vad is None:
        logger.info("Loading VAD model...")
        _vad = silero.VAD.load()
        logger.info("VAD model loaded")
    return _vad


# Create the agent server
server = AgentServer()


@server.rtc_session()
async def agent_session(ctx: JobContext):
    """
    RTC session handler called when a new room session starts.
    Creates and configures the AgentSession with STT, LLM, and TTS.
    """
    logger.info(f"Starting agent session for room: {ctx.room.name}")

    # Create the agent session using direct provider plugins (not LiveKit Cloud inference)
    # Transcription forwarding is enabled by default in AgentSession
    # STT Options (uncomment one):
    # stt=groq.STT(model="whisper-large-v3"),  # Groq Whisper - fast and free tier available
    # stt=openai.STT(model="whisper-1"),       # OpenAI Whisper
    # stt=openai.STT(model="gpt-4o-transcribe"),  # GPT-4o Transcribe - better accuracy
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-transcribe"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # tts=openai.TTS(model="tts-1-hd-1106", voice="nova"),  # HD quality
        tts=openai.TTS(model="tts-1", voice="nova"),
        vad=get_vad(),
    )

    # Create the citizen agent with RAG capabilities
    agent = CitizenAgent()

    # Start the session with the agent
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
        ),
    )

    # Connect to the room
    await ctx.connect()

    logger.info("Agent session started successfully")


def main():
    """
    Main entry point for the agent worker.
    """
    cli.run_app(server)


if __name__ == "__main__":
    main()
