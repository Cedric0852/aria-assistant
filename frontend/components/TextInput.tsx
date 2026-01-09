'use client';

import { useState, useCallback, useRef, useEffect, KeyboardEvent } from 'react';
import { queryText, queryTextStream, playAudioBase64 } from '@/lib/api';

interface TextInputProps {
  sessionId: string;
  onMessageSent: (userMessage: string, agentResponse: string) => void;
  onStreamingUpdate?: (userMessage: string, partialResponse: string, isComplete: boolean) => void;
  onLiveKitMessage?: (message: string) => Promise<void>;
  ttsEnabled: boolean;
  streamingEnabled?: boolean;
  useLiveKit?: boolean;
}

/**
 * Text input component for typed queries
 * Supports both regular REST API and LiveKit data channel for messages
 * When useLiveKit is true and onLiveKitMessage is provided, sends through LiveKit
 */
export default function TextInput({
  sessionId,
  onMessageSent,
  onStreamingUpdate,
  onLiveKitMessage,
  ttsEnabled,
  streamingEnabled = true,
  useLiveKit = false,
}: TextInputProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const currentQueryRef = useRef<string>('');

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle streaming query
  const handleStreamingSubmit = useCallback(
    async (query: string) => {
      currentQueryRef.current = query;
      setIsStreaming(true);

      try {
        await queryTextStream(
          query,
          {
            onToken: (token, accumulated) => {
              // Update parent with partial response
              onStreamingUpdate?.(query, accumulated, false);
            },
            onComplete: (fullResponse) => {
              // Notify parent of complete message
              onMessageSent(query, fullResponse);
              onStreamingUpdate?.(query, fullResponse, true);
            },
            onError: (errorMsg) => {
              setError(errorMsg);
            },
          },
          sessionId
        );
      } catch (err) {
        console.error('Streaming query failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to get response');
      } finally {
        setIsStreaming(false);
        inputRef.current?.focus();
      }
    },
    [sessionId, onMessageSent, onStreamingUpdate]
  );

  // Handle regular (non-streaming) query
  const handleRegularSubmit = useCallback(
    async (query: string) => {
      try {
        const response = await queryText(query, sessionId, ttsEnabled);
        onMessageSent(query, response.answer);

        // Play TTS audio if enabled and available
        if (ttsEnabled && response.audio_base64) {
          try {
            await playAudioBase64(response.audio_base64);
          } catch (audioError) {
            console.error('TTS playback failed:', audioError);
          }
        }
      } catch (err) {
        console.error('Query failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to get response');
      }
    },
    [sessionId, ttsEnabled, onMessageSent]
  );

  // Handle LiveKit message submission (sends through data channel to agent)
  const handleLiveKitSubmit = useCallback(
    async (query: string) => {
      if (!onLiveKitMessage) return;

      try {
        // Add user message immediately to chat
        onMessageSent(query, ''); // Empty response - agent will respond via transcription
        await onLiveKitMessage(query);
      } catch (err) {
        console.error('LiveKit message failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to send message');
      }
    },
    [onLiveKitMessage, onMessageSent]
  );

  // Handle form submission
  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();

      const query = input.trim();
      if (!query || isLoading || isStreaming) return;

      setIsLoading(true);
      setError(null);
      setInput('');

      try {
        if (useLiveKit && onLiveKitMessage) {
          // Send through LiveKit data channel (agent responds via voice/transcription)
          await handleLiveKitSubmit(query);
        } else if (streamingEnabled && onStreamingUpdate) {
          // Use streaming REST API mode
          await handleStreamingSubmit(query);
        } else {
          // Use regular REST API mode
          await handleRegularSubmit(query);
        }
      } finally {
        setIsLoading(false);
        inputRef.current?.focus();
      }
    },
    [input, isLoading, isStreaming, useLiveKit, onLiveKitMessage, streamingEnabled, onStreamingUpdate, handleLiveKitSubmit, handleStreamingSubmit, handleRegularSubmit]
  );

  // Handle Enter key press
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div className="bg-white border-t border-gray-200 px-4 py-3">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        {/* Error message */}
        {error && (
          <div className="mb-2 px-3 py-2 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Input row */}
        <div className="flex items-center space-x-3">
          {/* Text input */}
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your question..."
              disabled={isLoading || isStreaming}
              className="text-input pr-12"
              aria-label="Type your question"
            />

            {/* TTS indicator */}
            {ttsEnabled && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <span className="text-xs text-gray-400" title="Voice response enabled">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                    />
                  </svg>
                </span>
              </div>
            )}
          </div>

          {/* Send button */}
          <button
            type="submit"
            disabled={!input.trim() || isLoading || isStreaming}
            className="send-button"
            aria-label="Send message"
          >
            {(isLoading || isStreaming) ? (
              <svg
                className="w-5 h-5 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            ) : (
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            )}
          </button>
        </div>

        {/* Helper text */}
        <p className="mt-2 text-xs text-gray-400 text-center">
          Press Enter to send • Voice input is also available
        </p>
      </form>
    </div>
  );
}
