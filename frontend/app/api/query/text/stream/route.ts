import { NextRequest } from 'next/server';

/**
 * Streaming text query request body interface
 */
interface StreamingQueryRequest {
  query: string;
  session_id?: string;
}

/**
 * Backend API URL
 */
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST /api/query/text/stream
 *
 * Proxy endpoint for streaming text queries to the backend.
 * Streams Server-Sent Events from the backend to the client.
 */
export async function POST(request: NextRequest): Promise<Response> {
  try {
    // Parse request body
    const body: StreamingQueryRequest = await request.json();

    // Validate required fields
    if (!body.query || body.query.trim() === '') {
      return new Response(
        JSON.stringify({ detail: 'query is required and cannot be empty' }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Forward request to backend streaming endpoint
    const backendResponse = await fetch(`${BACKEND_URL}/api/query/text/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: body.query,
        session_id: body.session_id,
      }),
    });

    // Handle backend errors
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({
        detail: `Backend error: ${backendResponse.status}`,
      }));
      return new Response(JSON.stringify(errorData), {
        status: backendResponse.status,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Stream the SSE response from backend to client
    const reader = backendResponse.body?.getReader();
    if (!reader) {
      return new Response(
        JSON.stringify({ detail: 'No response body from backend' }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Create a TransformStream to pass through the SSE data
    const stream = new ReadableStream({
      async start(controller) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              controller.close();
              break;
            }
            controller.enqueue(value);
          }
        } catch (error) {
          console.error('Stream read error:', error);
          controller.error(error);
        } finally {
          reader.releaseLock();
        }
      },
    });

    // Return streaming response with SSE headers
    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no', // Disable nginx buffering
      },
    });
  } catch (error) {
    console.error('Streaming query proxy error:', error);

    // Handle network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return new Response(
        JSON.stringify({ detail: 'Backend service unavailable. Please try again later.' }),
        {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Handle JSON parse errors
    if (error instanceof SyntaxError) {
      return new Response(
        JSON.stringify({ detail: 'Invalid request body' }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Generic error
    return new Response(
      JSON.stringify({ detail: 'Internal server error' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

/**
 * OPTIONS /api/query/text/stream
 *
 * Handle CORS preflight requests
 */
export async function OPTIONS(): Promise<Response> {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
    },
  });
}
