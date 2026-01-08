"""
ARIA Chat - Streamlit Chat Application

Main chat interface for the ARIA AI Citizen Support Assistant.
Connects to backend API for text and audio queries with streaming responses.
"""

import base64
import json
import os
import uuid
from io import BytesIO

import requests
import sseclient
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://localhost:8000")
LOCAL_STORAGE_KEY = "aria_session_id"

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="ARIA Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
    }
    .source-card {
        background-color: #1e1e2e;
        border: 1px solid #3d3d5c;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #e0e0e0;
    }
    .source-card strong {
        color: #ffffff;
    }
    .source-card a {
        color: #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_chat_history(session_id: str) -> list:
    """Load chat history from backend API with short timeout to prevent skeleton hang."""
    try:
        response = requests.get(
            f"{API_URL}/api/chat/history/{session_id}",
            timeout=2,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("messages", [])
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass
    return []


def save_chat_history(session_id: str, messages: list):
    """Save chat history to backend API."""
    try:
        # Filter out audio_base64 to reduce payload size
        messages_to_save = []
        for msg in messages:
            msg_copy = {k: v for k, v in msg.items() if k != "audio_base64"}
            messages_to_save.append(msg_copy)

        requests.post(
            f"{API_URL}/api/chat/history/{session_id}",
            json={"messages": messages_to_save},
            timeout=5,
        )
    except Exception as e:
        # Silently fail - don't block UI for save failures
        pass


def clear_chat_history_api(session_id: str):
    """Clear chat history from backend API."""
    try:
        requests.delete(
            f"{API_URL}/api/chat/history/{session_id}",
            timeout=5,
        )
    except Exception:
        pass


def get_local_storage_session_id():
    """Inject JavaScript to get session ID from localStorage and store in a hidden div."""
    components.html(
        f"""
        <script>
            const key = '{LOCAL_STORAGE_KEY}';
            let sessionId = localStorage.getItem(key);
            if (!sessionId) {{
                sessionId = '{str(uuid.uuid4())}';
                localStorage.setItem(key, sessionId);
            }}
            // Store in parent window for Streamlit to access via query params
            const url = new URL(window.parent.location);
            if (url.searchParams.get('sid') !== sessionId) {{
                url.searchParams.set('sid', sessionId);
                window.parent.history.replaceState({{}}, '', url);
            }}
        </script>
        """,
        height=0,
    )


def save_session_to_local_storage(session_id: str):
    """Save session ID to localStorage."""
    components.html(
        f"""
        <script>
            localStorage.setItem('{LOCAL_STORAGE_KEY}', '{session_id}');
        </script>
        """,
        height=0,
    )


def init_session_state():
    """Initialize session state variables with persistent session_id via localStorage."""
    # Inject localStorage script to sync session ID
    get_local_storage_session_id()

    # Get session_id from query params (set by localStorage script)
    query_session_id = st.query_params.get("sid")

    if "session_id" not in st.session_state:
        if query_session_id:
            st.session_state.session_id = query_session_id
        else:
            st.session_state.session_id = str(uuid.uuid4())

    # Initialize messages as empty list first to prevent skeleton hang
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history_loaded = False

    # Load history in background (non-blocking)
    if not st.session_state.get("history_loaded", False):
        loaded_messages = load_chat_history(st.session_state.session_id)
        if loaded_messages:
            st.session_state.messages = loaded_messages
        st.session_state.history_loaded = True


def display_sources(sources: list):
    """Display source documents."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            title = source.get("title", "Unknown source")
            url = source.get("url", "")
            score = source.get("score", 0)

            source_html = f'<div class="source-card"><strong>{title}</strong>'
            if score:
                source_html += f" (relevance: {score:.0%})"
            if url:
                source_html += f'<br><a href="{url}" target="_blank">{url}</a>'
            source_html += "</div>"

            st.markdown(source_html, unsafe_allow_html=True)


def play_audio(audio_base64: str):
    """Play audio from base64 encoded string."""
    if not audio_base64:
        return

    try:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format="audio/wav", autoplay=True)
    except Exception as e:
        st.error(f"Failed to play audio: {e}")


def stream_text_response(query: str) -> dict:
    """
    Stream text response from backend API using SSE.

    Args:
        query: User query text

    Returns:
        Dict with full_response, sources, confidence, and audio_base64
    """
    url = f"{API_URL}/api/query/text/stream-with-audio"

    result = {
        "full_response": "",
        "sources": [],
        "confidence": 0.0,
        "audio_base64": None,
        "error": None,
    }

    try:
        response = requests.post(
            url,
            json={
                "query": query,
                "session_id": st.session_state.session_id,
            },
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=120,
        )

        if response.status_code != 200:
            result["error"] = f"API error: {response.status_code}"
            return result

        client = sseclient.SSEClient(response)

        response_placeholder = st.empty()
        accumulated_text = ""

        for event in client.events():
            if not event.data:
                continue

            try:
                data = json.loads(event.data)
                event_type = data.get("type", "")

                if event_type == "session":
                    new_session_id = data.get("session_id")
                    if new_session_id:
                        st.session_state.session_id = new_session_id
                        save_session_to_local_storage(new_session_id)

                elif event_type == "sources":
                    result["sources"] = data.get("sources", [])
                    result["confidence"] = data.get("confidence", 0.0)

                elif event_type == "token":
                    token = data.get("token", "")
                    accumulated_text += token
                    response_placeholder.markdown(accumulated_text + "▌")

                elif event_type == "done":
                    result["full_response"] = data.get("full_response", accumulated_text)
                    response_placeholder.markdown(result["full_response"])

                elif event_type == "audio":
                    result["audio_base64"] = data.get("audio_base64")

                elif event_type == "error":
                    result["error"] = data.get("error", "Unknown error")
                    break

                elif event_type == "heartbeat":
                    pass

            except json.JSONDecodeError:
                continue

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        result["error"] = f"Could not connect to API at {API_URL}. Please check the server is running."
    except Exception as e:
        result["error"] = f"Error: {str(e)}"

    return result


def send_audio_query(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Send audio query to backend API.

    Args:
        audio_bytes: Raw audio bytes
        filename: Original filename

    Returns:
        Dict with transcript, answer, sources, and audio_base64
    """
    url = f"{API_URL}/api/query/audio"

    result = {
        "transcript": "",
        "answer": "",
        "sources": [],
        "audio_base64": None,
        "error": None,
    }

    try:
        # Prepare multipart form data
        files = {
            "file": (filename, BytesIO(audio_bytes), "audio/wav"),
        }
        data = {
            "session_id": st.session_state.session_id,
            "include_audio": "true",
        }

        response = requests.post(url, files=files, data=data, timeout=120)

        if response.status_code != 200:
            result["error"] = f"API error: {response.status_code} - {response.text}"
            return result

        response_data = response.json()

        result["transcript"] = response_data.get("transcript", "")
        result["answer"] = response_data.get("answer", "")
        result["sources"] = response_data.get("sources", [])
        result["audio_base64"] = response_data.get("audio_base64")

        new_session_id = response_data.get("session_id")
        if new_session_id:
            st.session_state.session_id = new_session_id
            save_session_to_local_storage(new_session_id)

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        result["error"] = f"Could not connect to API at {API_URL}. Please check the server is running."
    except Exception as e:
        result["error"] = f"Error: {str(e)}"

    return result


def display_chat_history():
    """Display all messages in chat history."""
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        sources = message.get("sources", [])
        confidence = message.get("confidence", 0.0)
        audio_base64 = message.get("audio_base64")

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                if sources:
                    display_sources(sources)

                if audio_base64:
                    st.audio(base64.b64decode(audio_base64), format="audio/wav")


def main():
    """Main application entry point."""
    init_session_state()

    st.title("🤖 ARIA Chat")
    st.markdown(
        "AI Citizen Support Assistant for Irembo - Rwanda's e-government platform"
    )

    # Sidebar with session info
    with st.sidebar:
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

        if st.button("🗑️ Clear Chat"):
            clear_chat_history_api(st.session_state.session_id)
            st.session_state.messages = []
            new_session_id = str(uuid.uuid4())
            st.session_state.session_id = new_session_id
            save_session_to_local_storage(new_session_id)
            st.rerun()

        st.markdown("---")
        st.markdown("### About ARIA")
        st.markdown(
            """
            **AI Rwanda Irembo Assistant**

            A Voice-First citizen support system for
            Rwanda's e-government services.

            **Features:**
            - Text & voice queries
            - RAG-powered accurate answers
            - Source citations
            - Audio responses (TTS)

            **Tech Stack:**
            - LlamaIndex + GPT-4o-mini
            - Groq Whisper (STT)
            - Redis caching

            [View Stats](/stats) | [API Docs](https://aria-api.lunaroot.rw/docs)

            ---
            Built by [@Cedric0852](https://github.com/Cedric0852)
            """
        )

    display_chat_history()

    user_input = st.chat_input(
        "Ask a question about government services...",
        accept_audio=True,
        audio_sample_rate=16000,
    )

    if user_input:
        has_audio = hasattr(user_input, 'audio') and user_input.audio is not None
        has_text = hasattr(user_input, 'text') and user_input.text

        if has_audio:
            audio_bytes = user_input.audio.read()

            with st.chat_message("user"):
                st.markdown("🎤 *[Voice message]*")

            st.session_state.messages.append({
                "role": "user",
                "content": "🎤 *[Voice message]*",
            })

            with st.chat_message("assistant"):
                with st.spinner("Processing audio..."):
                    result = send_audio_query(audio_bytes)

                if result["error"]:
                    st.error(result["error"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {result['error']}",
                    })
                else:
                    response_content = ""
                    if result["transcript"]:
                        response_content = f"**You said:** {result['transcript']}\n\n---\n\n"
                    response_content += result["answer"]

                    st.markdown(response_content)

                    if result["sources"]:
                        display_sources(result["sources"])

                    if result["audio_base64"]:
                        play_audio(result["audio_base64"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "sources": result["sources"],
                        "audio_base64": result["audio_base64"],
                    })

            save_chat_history(st.session_state.session_id, st.session_state.messages)

        elif has_text:
            user_text = user_input.text

            with st.chat_message("user"):
                st.markdown(user_text)

            st.session_state.messages.append({
                "role": "user",
                "content": user_text,
            })

            with st.chat_message("assistant"):
                result = stream_text_response(user_text)

                if result["error"]:
                    st.error(result["error"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {result['error']}",
                    })
                else:
                    if result["sources"]:
                        display_sources(result["sources"])

                    if result["audio_base64"]:
                        play_audio(result["audio_base64"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["full_response"],
                        "sources": result["sources"],
                        "confidence": result["confidence"],
                        "audio_base64": result["audio_base64"],
                    })

            save_chat_history(st.session_state.session_id, st.session_state.messages)

        else:
            user_text = str(user_input)

            with st.chat_message("user"):
                st.markdown(user_text)

            st.session_state.messages.append({
                "role": "user",
                "content": user_text,
            })

            with st.chat_message("assistant"):
                result = stream_text_response(user_text)

                if result["error"]:
                    st.error(result["error"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {result['error']}",
                    })
                else:
                    if result["sources"]:
                        display_sources(result["sources"])

                    if result["audio_base64"]:
                        play_audio(result["audio_base64"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["full_response"],
                        "sources": result["sources"],
                        "confidence": result["confidence"],
                        "audio_base64": result["audio_base64"],
                    })

            save_chat_history(st.session_state.session_id, st.session_state.messages)


if __name__ == "__main__":
    main()
