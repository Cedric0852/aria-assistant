import { NextRequest, NextResponse } from 'next/server';

/**
 * Text query request body interface
 */
interface TextQueryRequest {
  query: string;
  session_id?: string;
  include_audio?: boolean;
}

/**
 * Backend API URL
 */
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST /api/query/text
 *
 * Proxy endpoint for text queries to the backend.
 * This keeps the backend API internal and handles CORS.
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Parse request body
    const body: TextQueryRequest = await request.json();

    // Validate required fields
    if (!body.query || body.query.trim() === '') {
      return NextResponse.json(
        { detail: 'query is required and cannot be empty' },
        { status: 400 }
      );
    }

    // Forward request to backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/query/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: body.query,
        session_id: body.session_id,
        include_audio: body.include_audio ?? false,
      }),
    });

    // Handle backend errors
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({
        detail: `Backend error: ${backendResponse.status}`,
      }));
      return NextResponse.json(errorData, { status: backendResponse.status });
    }

    // Return query response
    const responseData = await backendResponse.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error('Query proxy error:', error);

    // Handle network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: 'Backend service unavailable. Please try again later.' },
        { status: 503 }
      );
    }

    // Handle JSON parse errors
    if (error instanceof SyntaxError) {
      return NextResponse.json(
        { detail: 'Invalid request body' },
        { status: 400 }
      );
    }

    // Generic error
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
}

/**
 * OPTIONS /api/query/text
 *
 * Handle CORS preflight requests
 */
export async function OPTIONS(): Promise<NextResponse> {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
    },
  });
}
