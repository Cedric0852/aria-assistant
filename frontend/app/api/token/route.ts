import { NextRequest, NextResponse } from 'next/server';

/**
 * Token request body interface
 */
interface TokenRequest {
  room_name: string;
  participant_name: string;
}

/**
 * Token response interface from backend
 */
interface TokenResponse {
  token: string;
  room_name: string;
  participant_name: string;
  livekit_url: string;
}

/**
 * Backend API URL
 */
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST /api/token
 *
 * Proxy endpoint to get a LiveKit token from the backend.
 * This keeps the backend API internal and handles CORS.
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Parse request body
    const body: TokenRequest = await request.json();

    // Validate required fields
    if (!body.room_name || !body.participant_name) {
      return NextResponse.json(
        { detail: 'room_name and participant_name are required' },
        { status: 400 }
      );
    }

    // Forward request to backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    // Handle backend errors
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({
        detail: `Backend error: ${backendResponse.status}`,
      }));
      return NextResponse.json(errorData, { status: backendResponse.status });
    }

    // Return token response
    const tokenData: TokenResponse = await backendResponse.json();

    return NextResponse.json(tokenData);
  } catch (error) {
    console.error('Token proxy error:', error);

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
 * OPTIONS /api/token
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
