import { NextRequest, NextResponse } from 'next/server';

/**
 * Backend API URL
 */
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST /api/query/audio
 *
 * Proxy endpoint for audio queries to the backend.
 * Forwards the multipart form data to the backend.
 * Returns binary MP3 audio with metadata in headers by default.
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Get the form data from the request
    const formData = await request.formData();

    // Validate that we have a file
    const file = formData.get('file');
    if (!file) {
      return NextResponse.json(
        { detail: 'Audio file is required' },
        { status: 400 }
      );
    }

    // Default to binary response format
    if (!formData.has('response_format')) {
      formData.append('response_format', 'binary');
    }

    // Forward request to backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/query/audio`, {
      method: 'POST',
      body: formData,
    });

    // Handle backend errors
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({
        detail: `Backend error: ${backendResponse.status}`,
      }));
      return NextResponse.json(errorData, { status: backendResponse.status });
    }

    // Check if response is binary audio (WAV)
    const contentType = backendResponse.headers.get('content-type');
    if (contentType?.includes('audio/')) {
      // Return binary audio response with headers
      const audioBuffer = await backendResponse.arrayBuffer();
      return new NextResponse(audioBuffer, {
        status: 200,
        headers: {
          'Content-Type': contentType,
          'X-Transcript': backendResponse.headers.get('X-Transcript') || '',
          'X-Answer': backendResponse.headers.get('X-Answer') || '',
          'X-Session-ID': backendResponse.headers.get('X-Session-ID') || '',
          'Content-Disposition': 'attachment; filename=response.wav',
        },
      });
    }

    // Return JSON response (fallback)
    const responseData = await backendResponse.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error('Audio query proxy error:', error);

    // Handle network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: 'Backend service unavailable. Please try again later.' },
        { status: 503 }
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
 * OPTIONS /api/query/audio
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
