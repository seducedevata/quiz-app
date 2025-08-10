import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { method, args = [] } = body;

    if (!method) {
      return NextResponse.json(
        { success: false, error: 'Method is required' },
        { status: 400 }
      );
    }

  // Forward the request to the Python bridge server (configurable)
  const baseUrl = process.env.NEXT_PUBLIC_PYTHON_BRIDGE_URL || 'http://localhost:8000';
  const pythonBridgeUrl = `${baseUrl}/api/call`;
    
  const response = await fetch(pythonBridgeUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        method,
        args,
      }),
    });

    if (!response.ok) {
      throw new Error(`Python bridge responded with status: ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error) {
    console.error('API call error:', error);
    
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      },
      { status: 500 }
    );
  }
}

// Handle preflight requests for CORS
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}
