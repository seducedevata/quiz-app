import { io, Socket } from 'socket.io-client';

const BASE_URL = process.env.NEXT_PUBLIC_PYTHON_BRIDGE_URL || 'http://localhost:8765';

interface PythonResponse {
  status: 'success' | 'error';
  data?: any;
  message?: string;
  id: number;
}

interface WebSocketEvent {
  event: string;
  data: any;
}

let socket: Socket | null = null;
const eventListeners: Map<string, ((data: any) => void)[]> = new Map();

const ensureWebSocketConnection = () => {
  if (!socket || !socket.connected) {
    socket = io(BASE_URL, {
      transports: ['websocket'],
      autoConnect: true,
    });

    socket.on('connect', () => {
      console.log('WebSocket connected to Python bridge');
    });

    socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected from Python bridge:', reason);
    });

    socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
    });

    // Listen for custom events from the Python backend
    socket.on('message', (message: string) => {
      try {
        const parsed: WebSocketEvent = JSON.parse(message);
        if (parsed.event) {
          eventListeners.get(parsed.event)?.forEach(listener => listener(parsed.data));
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e, message);
      }
    });
  }
};

export const callPythonMethod = async <T = any>(method: string, ...args: any[]): Promise<T> => {
  const requestBody = {
    method,
    args: args.length === 1 ? args[0] : args, // Handle single object arg vs multiple args
    id: Math.floor(Math.random() * 1000000), // Simple request ID
  };

  try {
    const response = await fetch(`${BASE_URL}/api/call`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
    }

    const result: PythonResponse = await response.json();

    if (result.status === 'success') {
      return result.data as T;
    } else {
      throw new Error(result.message || 'Unknown error from Python backend');
    }
  } catch (error) {
    console.error(`Error calling Python method ${method}:`, error);
    throw error;
  }
};

export const onPythonEvent = (eventName: string, listener: (data: any) => void) => {
  ensureWebSocketConnection();
  if (!eventListeners.has(eventName)) {
    eventListeners.set(eventName, []);
  }
  eventListeners.get(eventName)?.push(listener);
};

export const offPythonEvent = (eventName: string, listener: (data: any) => void) => {
  const listeners = eventListeners.get(eventName);
  if (listeners) {
    eventListeners.set(eventName, listeners.filter(l => l !== listener));
  }
};

export const disconnectPythonBridge = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  eventListeners.clear();
};

// Optional: Health check for the bridge
export const checkBridgeHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    if (response.ok) {
      const data = await response.json();
      console.log('Python Bridge Health:', data);
      return data.status === 'healthy';
    }
    return false;
  } catch (error) {
    console.error('Error checking Python Bridge health:', error);
    return false;
  }
};
