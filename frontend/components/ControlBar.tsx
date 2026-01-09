'use client';

interface ControlBarProps {
  isMuted: boolean;
  ttsEnabled: boolean;
  isConnected: boolean;
  onMuteToggle: () => void;
  onTtsToggle: () => void;
  onDisconnect: () => void;
}

/**
 * Control bar component with microphone, TTS, and disconnect controls
 */
export default function ControlBar({
  isMuted,
  ttsEnabled,
  isConnected,
  onMuteToggle,
  onTtsToggle,
  onDisconnect,
}: ControlBarProps) {
  return (
    <div className="bg-white border-t border-gray-200 px-4 py-4">
      <div className="max-w-4xl mx-auto flex items-center justify-center space-x-6">
        {/* Microphone Toggle */}
        <div className="flex flex-col items-center">
          <button
            onClick={onMuteToggle}
            className={`control-button ${
              isMuted ? 'control-button-secondary' : 'control-button-primary'
            }`}
            aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
            title={isMuted ? 'Tap to unmute' : 'Tap to mute'}
          >
            {isMuted ? (
              <svg
                className="w-6 h-6"
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
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 3l18 18"
                />
              </svg>
            ) : (
              <svg
                className="w-6 h-6"
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
            )}
          </button>
          <span className="mt-1 text-xs text-gray-500">
            {isMuted ? 'Mic Off' : 'Mic On'}
          </span>
        </div>

        {/* TTS Toggle (Speaker) */}
        <div className="flex flex-col items-center">
          <button
            onClick={onTtsToggle}
            className={`control-button ${
              ttsEnabled ? 'control-button-primary' : 'control-button-secondary'
            }`}
            aria-label={ttsEnabled ? 'Mute speaker' : 'Unmute speaker'}
            title={ttsEnabled ? 'Tap to mute speaker' : 'Tap to unmute speaker'}
          >
            {ttsEnabled ? (
              <svg
                className="w-6 h-6"
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
            ) : (
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2"
                />
              </svg>
            )}
          </button>
          <span className="mt-1 text-xs text-gray-500">
            {ttsEnabled ? 'Speaker On' : 'Speaker Off'}
          </span>
        </div>

        {/* Connection Status Indicator */}
        <div className="flex flex-col items-center">
          <div className="flex items-center space-x-2 px-3 py-2 bg-gray-100 rounded-full h-12">
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              }`}
            />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
          <span className="mt-1 text-xs text-gray-500">Status</span>
        </div>

        {/* Disconnect Button */}
        <div className="flex flex-col items-center">
          <button
            onClick={onDisconnect}
            className="control-button control-button-danger"
            aria-label="End conversation"
            title="End conversation"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 8l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M5 3a2 2 0 00-2 2v1c0 8.284 6.716 15 15 15h1a2 2 0 002-2v-3.28a1 1 0 00-.684-.948l-4.493-1.498a1 1 0 00-1.21.502l-1.13 2.257a11.042 11.042 0 01-5.516-5.517l2.257-1.128a1 1 0 00.502-1.21L9.228 3.683A1 1 0 008.279 3H5z"
              />
            </svg>
          </button>
          <span className="mt-1 text-xs text-gray-500">End</span>
        </div>
      </div>

      {/* Control hints - hidden on mobile */}
      <div className="mt-3 hidden sm:flex items-center justify-center space-x-6 text-xs text-gray-400">
        <span className="flex items-center space-x-1">
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500">
            Space
          </kbd>
          <span>Push to talk</span>
        </span>
        <span className="flex items-center space-x-1">
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500">
            Esc
          </kbd>
          <span>End call</span>
        </span>
      </div>
    </div>
  );
}
