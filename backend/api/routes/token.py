"""Token endpoints disabled.

RTC integration has been removed from this deployment. These routes
now return HTTP 501 to indicate the feature is not available.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["token"])


class TokenRequest(BaseModel):
    room_name: str = Field(..., min_length=1)
    participant_name: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    token: str
    room_name: str
    participant_name: str
    connection_url: str


@router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest) -> TokenResponse:
    raise HTTPException(status_code=501, detail="RTC support has been removed.")


@router.post("/token/agent", response_model=TokenResponse)
async def generate_agent_token(room_name: str, agent_name: str = "citizen-support-agent") -> TokenResponse:
    raise HTTPException(status_code=501, detail="RTC support has been removed.")
