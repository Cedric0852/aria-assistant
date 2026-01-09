import { NextResponse } from 'next/server';

/**
 * GET /api/health
 *
 * Health check endpoint for container orchestration and monitoring
 */
export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'citizen-support-frontend',
    timestamp: new Date().toISOString(),
  });
}
