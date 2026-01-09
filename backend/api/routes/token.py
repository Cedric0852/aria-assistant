"""LiveKit token generation endpoint."""

from datetime import timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from livekit.api import AccessToken, VideoGrants
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["token"])


class TokenRequest(BaseModel):
    """Request model for LiveKit token generation."""
    room_name: str = Field(..., min_length=1, description="Name of the LiveKit room")
    participant_name: str = Field(..., min_length=1, description="Name of the participant")


class TokenResponse(BaseModel):
    """Response model for LiveKit token generation."""
    token: str
    room_name: str
    participant_name: str
    livekit_url: str


def get_livekit_credentials() -> tuple[str, str, str]:
    """Get LiveKit credentials from environment variables."""
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    # Use public URL for browser connections (not internal Docker URL)
    livekit_url = os.getenv("LIVEKIT_PUBLIC_URL", "ws://localhost:7880")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured. Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET.",
        )

    return api_key, api_secret, livekit_url


@router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest) -> TokenResponse:
    """
    Generate a LiveKit access token for a participant.

    The token includes:
    - Room join permissions
    - Audio/video publish permissions
    - Agent dispatch configuration for the citizen support agent

    Args:
        request: Token request with room_name and participant_name

    Returns:
        TokenResponse with the JWT token and connection details
    """
    api_key, api_secret, livekit_url = get_livekit_credentials()

    logger.info(f"Generating token with API key: {api_key}, room: {request.room_name}")

    # Create access token using fluent API
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(request.participant_name)
        .with_name(request.participant_name)
        .with_grants(VideoGrants(
            room=request.room_name,
            room_join=True,
            room_create=True,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))
        .with_ttl(timedelta(hours=1))
        .with_metadata('{"agent_dispatch": true}')
    )

    # Generate JWT
    jwt_token = token.to_jwt()

    return TokenResponse(
        token=jwt_token,
        room_name=request.room_name,
        participant_name=request.participant_name,
        livekit_url=livekit_url,
    )


@router.post("/token/agent", response_model=TokenResponse)
async def generate_agent_token(
    room_name: str,
    agent_name: str = "citizen-support-agent",
) -> TokenResponse:
    """
    Generate a LiveKit access token for the agent.

    This is used internally when the agent needs to join a room.

    Args:
        room_name: Name of the room to join
        agent_name: Identity for the agent (default: citizen-support-agent)

    Returns:
        TokenResponse with the agent's JWT token
    """
    api_key, api_secret, livekit_url = get_livekit_credentials()

    # Create access token using fluent API
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(agent_name)
        .with_name("Citizen Support Agent")
        .with_grants(VideoGrants(
            room=room_name,
            room_join=True,
            room_create=True,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
            room_admin=True,
        ))
        .with_ttl(timedelta(hours=1))
    )

    jwt_token = token.to_jwt()

    return TokenResponse(
        token=jwt_token,
        room_name=room_name,
        participant_name=agent_name,
        livekit_url=livekit_url,
    )
