'use client';

import { useState, useCallback, useEffect } from 'react';
import dynamic from 'next/dynamic';

// Dynamic import for VoiceAssistant to avoid SSR issues with LiveKit
const VoiceAssistant = dynamic(
  () => import('@/components/VoiceAssistant'),
  {
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-dots mb-4">
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
          </div>
          <p className="text-gray-500">Loading voice assistant...</p>
        </div>
      </div>
    )
  }
);

export default function Home() {
  const [roomName, setRoomName] = useState<string>('');
  const [participantName, setParticipantName] = useState<string>('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generate a unique room name on mount
  useEffect(() => {
    const generateId = () => {
      return `citizen-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    };
    setRoomName(generateId());
    setParticipantName(`user-${Math.random().toString(36).substr(2, 6)}`);
  }, []);

  const handleConnect = useCallback(async () => {
    if (!roomName || !participantName) {
      setError('Room name and participant name are required');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      // Test token endpoint
      const response = await fetch('/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_name: roomName,
          participant_name: participantName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to get token: ${response.status}`);
      }

      setIsConnected(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed');
      console.error('Connection error:', err);
    } finally {
      setIsConnecting(false);
    }
  }, [roomName, participantName]);

  const handleDisconnect = useCallback(() => {
    setIsConnected(false);
    // Generate new room name for next session
    setRoomName(`citizen-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  // Show connected voice assistant
  if (isConnected) {
    return (
      <VoiceAssistant
        roomName={roomName}
        participantName={participantName}
        onDisconnect={handleDisconnect}
      />
    );
  }

  // Show connect screen
  return (
    <div className="flex-1 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* Welcome Section */}
          <div className="text-center mb-8">
            <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-10 h-10 text-primary-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Welcome to ARIA
            </h2>
            <p className="text-gray-600">
              Your AI assistant for Rwandan government services through Irembo.
            </p>
          </div>

          {/* Features List */}
          <div className="space-y-3 mb-8">
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">Voice Interaction</p>
                <p className="text-xs text-gray-500">Speak naturally to ask questions</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">Text Input</p>
                <p className="text-xs text-gray-500">Type your questions if you prefer</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">24/7 Availability</p>
                <p className="text-xs text-gray-500">Get help anytime you need it</p>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          {/* Connect Button */}
          <button
            onClick={handleConnect}
            disabled={isConnecting}
            className="w-full py-3 px-4 bg-primary-600 text-white font-medium rounded-full
                     hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500
                     focus:ring-offset-2 transition-colors duration-200
                     disabled:opacity-50 disabled:cursor-not-allowed
                     flex items-center justify-center space-x-2"
          >
            {isConnecting ? (
              <>
                <div className="loading-dots">
                  <div className="loading-dot bg-white"></div>
                  <div className="loading-dot bg-white"></div>
                  <div className="loading-dot bg-white"></div>
                </div>
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <span>Start Conversation</span>
              </>
            )}
          </button>

          {/* Privacy Note */}
          <p className="mt-4 text-xs text-gray-400 text-center">
            Your conversation may be recorded for quality assurance purposes.
            No personal data is stored without your consent.
          </p>
        </div>
      </div>
    </div>
  );
}
