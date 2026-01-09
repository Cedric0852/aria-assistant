'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  AgentState,
  useChat,
  useTranscriptions,
  useLocalParticipant,
} from '@livekit/components-react';

interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: Date;
}

// Avatar components for user and bot
const UserAvatar = () => (
  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-md">
    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
    </svg>
  </div>
);

const BotAvatar = () => (
  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-md">
    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  </div>
);

// Timing constant: ~300ms per word (180 WPM speaking rate)
const MS_PER_WORD = 300;

/**
 * Custom hook for progressive word-by-word text reveal
 * Tracks which messages are currently being revealed and how many words are shown
 */
function useTextReveal(state: AgentState) {
  // Map of message ID -> number of words currently revealed
  const [revealingMessages, setRevealingMessages] = useState<Map<string, number>>(new Map());
  // Track which messages have completed revealing
  const completedMessagesRef = useRef<Set<string>>(new Set());
  // Store timeout refs for cleanup
  const timeoutsRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Start revealing a new message
  const startReveal = useCallback((messageId: string, _totalWords: number) => {
    // Skip if already completed or already revealing
    if (completedMessagesRef.current.has(messageId)) return;
    if (revealingMessages.has(messageId)) return;

    setRevealingMessages((prev: Map<string, number>) => {
      const next = new Map(prev);
      next.set(messageId, 1); // Start with first word
      return next;
    });
  }, [revealingMessages]);

  // Advance the reveal for a specific message
  const advanceReveal = useCallback((messageId: string, totalWords: number) => {
    setRevealingMessages((prev: Map<string, number>) => {
      const currentCount = prev.get(messageId) || 0;
      if (currentCount >= totalWords) {
        // Mark as completed and remove from revealing
        completedMessagesRef.current.add(messageId);
        const next = new Map(prev);
        next.delete(messageId);
        return next;
      }
      const next = new Map(prev);
      next.set(messageId, currentCount + 1);
      return next;
    });
  }, []);

  // Complete all reveals instantly (when speaking ends)
  const completeAllReveals = useCallback(() => {
    // Clear all timeouts
    timeoutsRef.current.forEach((timeout: ReturnType<typeof setTimeout>) => clearTimeout(timeout));
    timeoutsRef.current.clear();

    // Mark all currently revealing messages as completed
    setRevealingMessages((prev: Map<string, number>) => {
      prev.forEach((_: number, messageId: string) => {
        completedMessagesRef.current.add(messageId);
      });
      return new Map(); // Clear all revealing messages
    });
  }, []);

  // Get the revealed text for a message
  const getRevealedText = useCallback((messageId: string, fullText: string): string => {
    // If completed, return full text
    if (completedMessagesRef.current.has(messageId)) {
      return fullText;
    }

    const revealCount = revealingMessages.get(messageId);
    if (revealCount === undefined) {
      // Not revealing yet - check if we should show full text (not speaking)
      return fullText;
    }

    const words = fullText.split(/\s+/);
    return words.slice(0, revealCount).join(' ');
  }, [revealingMessages]);

  // Check if a message is currently revealing
  const isRevealing = useCallback((messageId: string): boolean => {
    return revealingMessages.has(messageId) && !completedMessagesRef.current.has(messageId);
  }, [revealingMessages]);

  // Schedule next word reveal
  const scheduleNextReveal = useCallback((messageId: string, totalWords: number) => {
    // Clear existing timeout for this message
    const existingTimeout = timeoutsRef.current.get(messageId);
    if (existingTimeout) {
      clearTimeout(existingTimeout);
    }

    const timeout = setTimeout(() => {
      advanceReveal(messageId, totalWords);
    }, MS_PER_WORD);

    timeoutsRef.current.set(messageId, timeout);
  }, [advanceReveal]);

  // When state changes from 'speaking', complete all reveals
  const prevStateRef = useRef<AgentState>(state);
  useEffect(() => {
    if (prevStateRef.current === 'speaking' && state !== 'speaking') {
      completeAllReveals();
    }
    prevStateRef.current = state;
  }, [state, completeAllReveals]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach((timeout: ReturnType<typeof setTimeout>) => clearTimeout(timeout));
      timeoutsRef.current.clear();
    };
  }, []);

  return {
    startReveal,
    getRevealedText,
    isRevealing,
    scheduleNextReveal,
    revealingMessages,
  };
}

interface StreamingMessage {
  userMessage: string;
  partialResponse: string;
}

interface TranscriptDisplayProps {
  messages: Message[];
  state: AgentState;
  streamingMessage?: StreamingMessage | null;
}

/**
 * Transcript display component showing chat messages
 * Uses useTranscriptions hook to receive STT transcriptions from LiveKit
 * Auto-scrolls to bottom on new messages
 * Supports streaming responses with real-time updates
 */
export default function TranscriptDisplay({
  messages,
  state,
  streamingMessage,
}: TranscriptDisplayProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Track displayed messages to maintain stable order (prevents re-sorting disruption)
  const [displayedMessages, setDisplayedMessages] = useState<Message[]>([]);
  const seenIdsRef = useRef<Set<string>>(new Set());
  const displayedMessagesRef = useRef<Message[]>([]);
  // Track transcription segment IDs to their message IDs for updates
  const segmentToMsgIdRef = useRef<Map<string, string>>(new Map());

  // Use the text reveal hook for word-by-word animation
  const {
    startReveal,
    getRevealedText,
    isRevealing,
    scheduleNextReveal,
    revealingMessages,
  } = useTextReveal(state);

  // Track which agent messages we've seen to detect new ones during speaking
  const seenAgentMessagesRef = useRef<Set<string>>(new Set());

  // Keep ref in sync with state
  useEffect(() => {
    displayedMessagesRef.current = displayedMessages;
  }, [displayedMessages]);

  // Get local participant to compare identities
  const { localParticipant } = useLocalParticipant();
  const localIdentity = localParticipant?.identity || '';

  // Use LiveKit's useTranscriptions hook to get transcriptions
  // Returns array of transcription segments with text, participant info, etc.
  const transcriptionSegments = useTranscriptions();

  // Get chat messages from LiveKit data channel
  const { chatMessages } = useChat();

  // Collect all new messages and append to displayed list (maintains insertion order)
  useEffect(() => {
    const newMessages: Message[] = [];
    const updatedSegments: Map<string, string> = new Map(); // msgId -> new content

    // Helper to check if content is duplicate (use ref to avoid infinite loop)
    const isContentDuplicate = (content: string, timestamp: Date): boolean => {
      return displayedMessagesRef.current.some(
        (existing: Message) =>
          existing.content === content &&
          Math.abs(existing.timestamp.getTime() - timestamp.getTime()) < 2000
      );
    };

    // Process text input messages
    messages.forEach((msg) => {
      if (!seenIdsRef.current.has(msg.id) && !isContentDuplicate(msg.content, msg.timestamp)) {
        newMessages.push(msg);
        seenIdsRef.current.add(msg.id);
        if (typeof window !== 'undefined') console.debug('TranscriptDisplay: adding prop message', msg);
      }
    });

    // Debug: expose shapes to browser console for troubleshooting
    // (kept lightweight to avoid noisy logs)
    if (typeof window !== 'undefined') {
      try {
        if (transcriptionSegments && transcriptionSegments.length > 0) {
          console.debug('TranscriptDisplay: transcriptionSegments sample', transcriptionSegments[0]);
        }
        if (chatMessages && chatMessages.length > 0) {
          console.debug('TranscriptDisplay: chatMessages sample', chatMessages[0]);
        }
      } catch (e) {
        // ignore
      }
    }

    // Process transcription segments
    if (transcriptionSegments && transcriptionSegments.length > 0) {
      transcriptionSegments.forEach((segment: any) => {
        const text = segment.text || segment.content || '';
        // Determine final flag. LiveKit may provide a transcription_final attribute
        // inside streamInfo.attributes (string 'true'/'false') rather than a
        // boolean `final` property.
        const transcriptionFinalAttr = segment.streamInfo?.attributes?.['lk.transcription_final']
          ?? segment.info?.attributes?.['lk.transcription_final']
          ?? segment.streamInfo?.attributes?.lk_transcription_final
          ?? segment.info?.attributes?.lk_transcription_final;

        let isFinal: boolean;
        if (transcriptionFinalAttr !== undefined) {
          isFinal = transcriptionFinalAttr === true || String(transcriptionFinalAttr).toLowerCase() === 'true';
        } else {
          isFinal = segment.final !== false;
        }

        if (!text.trim()) return;

        const participantInfo = segment.participantInfo || segment.participant;
        const participantIdentity = (participantInfo?.identity || '').toLowerCase();
        const isLocal = participantInfo?.isLocal === true;
        const segmentIdentity = ((segment as any).participantIdentity || '').toLowerCase();
        const segmentSource = (segment as any).source || segment.publication?.source || '';
        const trackSource = segment.publication?.source || '';
        const segmentParticipantSid = participantInfo?.sid || (segment as any).participantSid || '';
        const isLocalBySid = localIdentity && segmentParticipantSid &&
                             localParticipant?.sid === segmentParticipantSid;

        const identityToCheck = participantIdentity || segmentIdentity;
        const isFromAgent = identityToCheck.includes('agent') ||
                            identityToCheck.includes('assistant') ||
                            identityToCheck.includes('citizen-agent');
        const isFromUser = identityToCheck.startsWith('user-') ||
                           identityToCheck === localIdentity.toLowerCase();
        const isFromMicrophone = trackSource === 'microphone' || segmentSource === 'microphone';

        let role: 'user' | 'agent' = 'agent';
        if (isFromUser) {
          role = 'user';
        } else if (isFromAgent) {
          role = 'agent';
        } else if (isLocal || isLocalBySid) {
          role = 'user';
        } else if (isFromMicrophone) {
          role = 'user';
        }

        // Use stable segment ID (support multiple possible keys, including nested streamInfo.id)
        const segmentId = segment.id
          || segment.streamId
          || segment.stream_id
          || segment.segment_id
          || segment.streamInfo?.id
          || segment.streamInfo?.streamId
          || '';
        if (!segmentId) return; // Skip segments without ID

        const msgId = `transcription-${segmentId}`;
        const transcriptionTime = segment.lastReceivedTime || segment.endTime ||
                                  segment.firstReceivedTime || segment.timestamp || Date.now();

        // Check if this segment already has a message displayed
        const existingMsgId = segmentToMsgIdRef.current.get(segmentId);

        if (existingMsgId) {
          // Update existing message with new content (handles streaming updates)
          updatedSegments.set(existingMsgId, text);
          if (typeof window !== 'undefined') {
            console.debug('TranscriptDisplay: updating message', { existingMsgId, text });
          }
        } else {
          // No existing displayed message for this segment
          // If final, add as finalized message and mark seen; if not final, add provisional message
          if (isFinal) {
            if (!seenIdsRef.current.has(msgId) && !isContentDuplicate(text, new Date(transcriptionTime))) {
              const msg: Message = {
                id: msgId,
                role,
                content: text,
                timestamp: new Date(transcriptionTime),
              };
              newMessages.push(msg);
              seenIdsRef.current.add(msgId);
              segmentToMsgIdRef.current.set(segmentId, msgId);
            }
          } else {
            // Provisional (interim) message: create and display, but don't mark as seen yet
            if (!isContentDuplicate(text, new Date(transcriptionTime))) {
              const provisionalId = msgId; // still use transcription-<segmentId>
              const msg: Message = {
                id: provisionalId,
                role,
                content: text,
                timestamp: new Date(transcriptionTime),
              };
              newMessages.push(msg);
              // Map segment -> msg so future updates replace content
              segmentToMsgIdRef.current.set(segmentId, provisionalId);
              if (typeof window !== 'undefined') {
                console.debug('TranscriptDisplay: adding provisional message', msg);
              }
            }
          }
        }
      });
    }

    // Process chat messages (tolerant to different shapes returned by useChat)
    chatMessages.forEach((chatMsg: any) => {
      const fromIdentity = (chatMsg.from?.identity || chatMsg.fromIdentity || '').toLowerCase();
      const isFromLocalUser = localIdentity && fromIdentity === localIdentity.toLowerCase();
      const isFromLocal = chatMsg.from?.isLocal === true || chatMsg.isLocal === true;

      // Skip messages sent by the local user (we already add them via props/messages)
      if (isFromLocalUser || isFromLocal) return;

      // Normalize message text (support message, text, body, payload.text)
      const messageText = (chatMsg.message || chatMsg.text || chatMsg.body || chatMsg.payload?.text || '').toString();
      if (!messageText.trim()) return;

      // Normalize timestamp
      const ts = chatMsg.timestamp || chatMsg.time || chatMsg.sentAt || Date.now();
      const msgId = chatMsg.id || `chat-${ts}`;

      if (!seenIdsRef.current.has(msgId) && !isContentDuplicate(messageText, new Date(ts))) {
        const msg: Message = {
          id: msgId,
          role: 'agent',
          content: messageText,
          timestamp: new Date(ts),
        };
        newMessages.push(msg);
        seenIdsRef.current.add(msgId);
      }
    });

    // Apply updates to existing messages and add new ones
    if (newMessages.length > 0 || updatedSegments.size > 0) {
      if (typeof window !== 'undefined') console.debug('TranscriptDisplay: will apply messages', { newCount: newMessages.length, updates: Array.from(updatedSegments.entries()) });
      setDisplayedMessages((prev: Message[]) => {
        let updated = prev;
        // Update existing messages
        if (updatedSegments.size > 0) {
          updated = prev.map(msg => {
            const newContent = updatedSegments.get(msg.id);
            return newContent ? { ...msg, content: newContent } : msg;
          });
        }
        // Add new messages
        return [...updated, ...newMessages];
      });
    }
  }, [transcriptionSegments, chatMessages, messages, localIdentity, localParticipant]);

  // Use displayedMessages for rendering (maintains stable insertion order)
  const allMessages = displayedMessages;

  // Detect new agent messages and start word reveal when speaking
  useEffect(() => {
    if (state === 'speaking') {
      allMessages.forEach((msg: Message) => {
        if (msg.role === 'agent' && !seenAgentMessagesRef.current.has(msg.id)) {
          // New agent message while speaking - start reveal
          const words = msg.content.split(/\s+/);
          startReveal(msg.id, words.length);
          seenAgentMessagesRef.current.add(msg.id);
        }
      });
    }
  }, [allMessages, state, startReveal]);

  // Progress the reveal animation for messages that are currently revealing
  useEffect(() => {
    revealingMessages.forEach((currentWordIndex: number, messageId: string) => {
      const msg = allMessages.find((m: Message) => m.id === messageId);
      if (msg) {
        const words = msg.content.split(/\s+/);
        if (currentWordIndex < words.length) {
          scheduleNextReveal(messageId, words.length);
        }
      }
    });
  }, [revealingMessages, allMessages, scheduleNextReveal]);

  // Auto-scroll to bottom on new messages or streaming updates
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [allMessages.length, streamingMessage?.partialResponse, revealingMessages]);

  // Format timestamp
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin bg-gradient-to-b from-slate-50 to-gray-100"
    >
      {/* Welcome message when empty */}
      {allMessages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-center px-4">
          <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-violet-200">
            <svg
              className="w-10 h-10 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-gray-900 mb-2">
            Start a Conversation
          </h3>
          <p className="text-gray-500 max-w-sm mb-6">
            Speak or type your question. I&apos;m here to help!
          </p>
          <div className="space-y-3 text-sm">
            <p className="text-gray-400 font-medium">Try asking:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              <span className="px-3 py-1.5 bg-white rounded-full text-gray-600 shadow-sm border border-gray-100 text-xs">
                &quot;How do I apply for a driver&apos;s license?&quot;
              </span>
              <span className="px-3 py-1.5 bg-white rounded-full text-gray-600 shadow-sm border border-gray-100 text-xs">
                &quot;What documents do I need?&quot;
              </span>
              <span className="px-3 py-1.5 bg-white rounded-full text-gray-600 shadow-sm border border-gray-100 text-xs">
                &quot;Where can I find forms?&quot;
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Message list */}
      {allMessages.map((message) => {
        // For agent messages, use revealed text if currently revealing
        const displayContent = message.role === 'agent'
          ? getRevealedText(message.id, message.content)
          : message.content;
        const currentlyRevealing = message.role === 'agent' && isRevealing(message.id);
        const isUser = message.role === 'user';

        return (
          <div
            key={message.id}
            className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
          >
            {/* Avatar */}
            {isUser ? <UserAvatar /> : <BotAvatar />}

            {/* Message container */}
            <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[75%]`}>
              {/* Label */}
              <span className={`text-xs font-semibold mb-1 px-1 ${
                isUser ? 'text-blue-600' : 'text-violet-600'
              }`}>
                {isUser ? 'You' : 'Aria'}
              </span>

              {/* Message bubble */}
              <div
                className={`chat-bubble ${
                  isUser ? 'chat-bubble-user' : 'chat-bubble-bot'
                }`}
              >
                {/* Message content */}
                {message.role === 'agent' ? (
                  <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:text-gray-800">
                    <ReactMarkdown>{displayContent}</ReactMarkdown>
                    {currentlyRevealing && (
                      <span className="inline-block w-0.5 h-4 bg-violet-400 animate-pulse ml-0.5 rounded" />
                    )}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{displayContent}</p>
                )}
              </div>

              {/* Timestamp */}
              <span className="text-[10px] mt-1 px-1 text-gray-400">
                {formatTime(message.timestamp)}
              </span>
            </div>
          </div>
        );
      })}

      {/* Streaming message display */}
      {streamingMessage && (
        <>
          {/* User message */}
          <div className="flex items-start gap-3 flex-row-reverse">
            <UserAvatar />
            <div className="flex flex-col items-end max-w-[75%]">
              <span className="text-xs font-semibold mb-1 px-1 text-blue-600">You</span>
              <div className="chat-bubble chat-bubble-user">
                <p className="whitespace-pre-wrap">{streamingMessage.userMessage}</p>
              </div>
              <span className="text-[10px] mt-1 px-1 text-gray-400">
                {formatTime(new Date())}
              </span>
            </div>
          </div>

          {/* Streaming agent response */}
          <div className="flex items-start gap-3 flex-row">
            <BotAvatar />
            <div className="flex flex-col items-start max-w-[75%]">
              <span className="text-xs font-semibold mb-1 px-1 text-violet-600">Aria</span>
              <div className="chat-bubble chat-bubble-bot">
                <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
                  <ReactMarkdown>{streamingMessage.partialResponse}</ReactMarkdown>
                  <span className="inline-block w-0.5 h-4 bg-violet-500 animate-pulse ml-0.5 rounded" />
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Typing indicator when agent is thinking (voice mode) */}
      {state === 'thinking' && !streamingMessage && (
        <div className="flex items-start gap-3 flex-row">
          <BotAvatar />
          <div className="flex flex-col items-start">
            <span className="text-xs font-semibold mb-1 px-1 text-violet-600">Aria</span>
            <div className="chat-bubble chat-bubble-bot">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm text-gray-500">Thinking...</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}
