'use client';

import '@livekit/components-styles';
import { useCallback, useEffect, useState } from 'react';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useVoiceAssistant,
  useRoomContext,
  useChat,
  AgentState,
} from '@livekit/components-react';
import TranscriptDisplay from './TranscriptDisplay';
import TextInput from './TextInput';
import ControlBar from './ControlBar';
import { getToken, TokenResponse } from '@/lib/api';

interface VoiceAssistantProps {
  roomName: string;
  participantName: string;
  onDisconnect: () => void;
}

/**
 * Main Voice Assistant component
 * Wraps LiveKitRoom and internal components
 */
export default function VoiceAssistant({
  roomName,
  participantName,
  onDisconnect,
}: VoiceAssistantProps) {
  const [token, setToken] = useState<string | null>(null);
  const [livekitUrl, setLivekitUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Fetch token on mount
  useEffect(() => {
    async function fetchToken() {
      try {
        const response: TokenResponse = await getToken(roomName, participantName);
        setToken(response.token);
        setLivekitUrl(response.livekit_url);
        setSessionId(response.room_name); // Use room name as session ID
      } catch (err) {
        console.error('Failed to get token:', err);
        setError(err instanceof Error ? err.message : 'Failed to get token');
      }
    }

    fetchToken();
  }, [roomName, participantName]);

  // Handle disconnect
  const handleDisconnect = useCallback(() => {
    setToken(null);
    setLivekitUrl(null);
    onDisconnect();
  }, [onDisconnect]);

  // Show error state
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-red-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Connection Error
          </h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={handleDisconnect}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  // Show loading state
  if (!token || !livekitUrl) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-dots mb-4">
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
          </div>
          <p className="text-gray-500">Connecting to assistant...</p>
        </div>
      </div>
    );
  }

  return (
    <LiveKitRoom
      token={token}
      serverUrl={livekitUrl}
      connect={true}
      audio={true}
      video={false}
      onDisconnected={handleDisconnect}
      onError={(err) => {
        console.error('LiveKit error:', err);
        setError(err.message);
      }}
      className="flex-1 flex flex-col"
    >
      <RoomAudioRenderer />
      <VoiceAssistantContent
        sessionId={sessionId || roomName}
        onDisconnect={handleDisconnect}
      />
    </LiveKitRoom>
  );
}

/**
 * Inner component that uses LiveKit hooks
 * Must be inside LiveKitRoom context
 */
interface VoiceAssistantContentProps {
  sessionId: string;
  onDisconnect: () => void;
}

function VoiceAssistantContent({
  sessionId,
  onDisconnect,
}: VoiceAssistantContentProps) {
  const room = useRoomContext();
  const { state, audioTrack } = useVoiceAssistant();
  const { send: sendChatMessage } = useChat();

  const [isMuted, setIsMuted] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [messages, setMessages] = useState<
    Array<{
      id: string;
      role: 'user' | 'agent';
      content: string;
      timestamp: Date;
    }>
  >([]);
  // State for streaming message
  const [streamingMessage, setStreamingMessage] = useState<{
    userMessage: string;
    partialResponse: string;
  } | null>(null);

  // Animated bars state for visualizer
  const [barHeights, setBarHeights] = useState<number[]>(Array(32).fill(8));

  // Log state changes for debugging
  useEffect(() => {
    console.log('[VoiceAssistant] State changed:', state, 'AudioTrack:', audioTrack ? 'present' : 'none');
  }, [state, audioTrack]);

  // Animate bars based on agent state
  useEffect(() => {
    let animationFrame: number;
    let lastTime = 0;

    const animate = (time: number) => {
      if (time - lastTime > 40) { // Update every 40ms for smoother animation
        lastTime = time;
        setBarHeights((prev: number[]) =>
          prev.map((_: number, i: number) => {
            if (state === 'speaking') {
              // More dynamic speaking animation with varied heights
              const baseHeight = 16;
              const variation = Math.random() * 22 + Math.sin(time / 80 + i * 0.3) * 10;
              return baseHeight + variation; // 16-48px
            } else if (state === 'thinking') {
              // Smooth wave effect for thinking
              return 12 + Math.sin(time / 120 + i * 0.4) * 12; // 0-24px wave
            } else if (state === 'listening') {
              // Subtle breathing animation for listening
              return 10 + Math.sin(time / 180 + i * 0.2) * 8; // 2-18px subtle wave
            }
            return 6; // Idle - small base bars
          })
        );
      }
      animationFrame = requestAnimationFrame(animate);
    };

    // Always start animation
    animationFrame = requestAnimationFrame(animate);

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [state]);

  // Track if we should use LiveKit for text messages (when in active voice session)
  const isVoiceSessionActive = room.state === 'connected' && state !== 'disconnected';

  // Send text message through LiveKit data channel
  const handleLiveKitMessage = useCallback(
    async (message: string) => {
      if (sendChatMessage) {
        await sendChatMessage(message);
      }
    },
    [sendChatMessage]
  );

  // Handle mute toggle
  const handleMuteToggle = useCallback(async () => {
    const newMuted = !isMuted;
    await room.localParticipant.setMicrophoneEnabled(!newMuted);
    setIsMuted(newMuted);
  }, [room, isMuted]);

  // Handle TTS toggle
  const handleTtsToggle = useCallback(() => {
    setTtsEnabled((prev) => !prev);
  }, []);

  // Push-to-talk: Space key to temporarily unmute
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        // Unmute when Space is pressed
        room.localParticipant.setMicrophoneEnabled(true);
        setIsMuted(false);
      } else if (e.code === 'Escape') {
        e.preventDefault();
        onDisconnect();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        // Mute when Space is released (push-to-talk behavior)
        room.localParticipant.setMicrophoneEnabled(false);
        setIsMuted(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [room, onDisconnect]);

  // Add message to list
  const addMessage = useCallback(
    (role: 'user' | 'agent', content: string) => {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
          role,
          content,
          timestamp: new Date(),
        },
      ]);
    },
    []
  );

  // Handle streaming updates
  const handleStreamingUpdate = useCallback(
    (userMessage: string, partialResponse: string, isComplete: boolean) => {
      if (isComplete) {
        // Clear streaming state when complete
        setStreamingMessage(null);
      } else {
        // Update streaming message
        setStreamingMessage({ userMessage, partialResponse });
      }
    },
    []
  );

  // Get state display info
  const getStateInfo = (agentState: AgentState) => {
    switch (agentState) {
      case 'listening':
        return {
          label: 'Listening',
          color: 'bg-agent-listening',
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          ),
        };
      case 'thinking':
        return {
          label: 'Thinking',
          color: 'bg-agent-thinking',
          icon: (
            <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          ),
        };
      case 'speaking':
        return {
          label: 'Speaking',
          color: 'bg-agent-speaking',
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
            </svg>
          ),
        };
      default:
        return {
          label: 'Ready',
          color: 'bg-agent-idle',
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.636 18.364a9 9 0 010-12.728m12.728 0a9 9 0 010 12.728m-9.9-2.829a5 5 0 010-7.07m7.072 0a5 5 0 010 7.07M13 12a1 1 0 11-2 0 1 1 0 012 0z" />
            </svg>
          ),
        };
    }
  };

  const stateInfo = getStateInfo(state);

  return (
    <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
      {/* Agent State Display - Modern Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-white/80 backdrop-blur-sm">
        {/* Left: Agent Info */}
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-md">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            {/* Online status dot */}
            <div className={`absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full border-2 border-white ${
              state === 'speaking' ? 'bg-violet-500' :
              state === 'thinking' ? 'bg-amber-400' :
              state === 'listening' ? 'bg-green-500' :
              'bg-gray-400'
            }`} />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-gray-900">Aria</h2>
            <p className="text-xs text-gray-500">{stateInfo.label}</p>
          </div>
        </div>

        {/* Center: Audio Visualizer */}
        <div className="flex-1 max-w-md mx-4">
          <div className="h-14 flex items-center justify-center bg-white rounded-2xl px-4 overflow-hidden border border-gray-200 shadow-sm">
            <div className="flex items-end justify-center gap-[3px] h-12 w-full">
              {barHeights.map((height: number, i: number) => (
                <div
                  key={i}
                  className="w-[4px] rounded-full"
                  style={{
                    height: `${Math.max(height, 4)}px`,
                    minHeight: '4px',
                    maxHeight: '44px',
                    background: state === 'speaking'
                      ? 'linear-gradient(to top, #7c3aed, #a78bfa)'
                      : state === 'thinking'
                      ? 'linear-gradient(to top, #f59e0b, #fcd34d)'
                      : state === 'listening'
                      ? 'linear-gradient(to top, #10b981, #6ee7b7)'
                      : 'linear-gradient(to top, #8b5cf6, #c4b5fd)',
                    transition: 'height 50ms ease-out'
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right: State Badge */}
        <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium ${
          state === 'speaking' ? 'bg-violet-100 text-violet-700' :
          state === 'thinking' ? 'bg-amber-100 text-amber-700' :
          state === 'listening' ? 'bg-green-100 text-green-700' :
          'bg-gray-100 text-gray-600'
        }`}>
          <div className={`w-2 h-2 rounded-full animate-pulse ${
            state === 'speaking' ? 'bg-violet-500' :
            state === 'thinking' ? 'bg-amber-500' :
            state === 'listening' ? 'bg-green-500' :
            'bg-gray-400'
          }`} />
          <span>{stateInfo.label}</span>
        </div>
      </div>

      {/* Transcript Display */}
      <div className="flex-1 overflow-hidden">
        <TranscriptDisplay
          messages={messages}
          state={state}
          streamingMessage={streamingMessage}
        />
      </div>

      {/* Text Input */}
      <TextInput
        sessionId={sessionId}
        onMessageSent={(userMessage, agentResponse) => {
          addMessage('user', userMessage);
          // Only add agent response if not using LiveKit (LiveKit responses come via transcription)
          if (agentResponse) {
            addMessage('agent', agentResponse);
          }
        }}
        onStreamingUpdate={handleStreamingUpdate}
        onLiveKitMessage={handleLiveKitMessage}
        ttsEnabled={ttsEnabled}
        streamingEnabled={!isVoiceSessionActive}
        useLiveKit={isVoiceSessionActive}
      />

      {/* Control Bar */}
      <ControlBar
        isMuted={isMuted}
        ttsEnabled={ttsEnabled}
        isConnected={room.state === 'connected'}
        onMuteToggle={handleMuteToggle}
        onTtsToggle={handleTtsToggle}
        onDisconnect={onDisconnect}
      />
    </div>
  );
}
