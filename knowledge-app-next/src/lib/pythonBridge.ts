import { io, Socket } from 'socket.io-client';
import { AppLogger } from './logger';
import { sessionTracker } from './sessionTracker';

const BASE_URL = process.env.NEXT_PUBLIC_PYTHON_BRIDGE_URL || 'http://localhost:8000';

// Enhanced connection status tracking
let connectionStatus = {
  connected: false,
  lastError: null as string | null,
  retryCount: 0,
  lastPing: 0,
  responseTime: 0,
  errorCount: 0,
  quality: 'critical' as 'excellent' | 'good' | 'poor' | 'critical',
  degradedMode: false,
  consecutiveFailures: 0,
  lastSuccessfulCall: Date.now()
};

// Method call queue for offline scenarios
let methodCallQueue: Array<{
  method: string;
  args: any[];
  resolve: Function;
  reject: Function;
  timestamp: number;
}> = [];

let isProcessingQueue = false;

// Retry configuration
const retryConfig = {
  maxAttempts: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: true
};

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
    AppLogger.info('BRIDGE_CONNECTION', 'Establishing WebSocket connection to Python bridge');
    
    socket = io(BASE_URL, {
      transports: ['websocket'],
      autoConnect: true,
      reconnectionAttempts: Infinity, // Attempt to reconnect indefinitely
      reconnectionDelay: 1000, // Initial delay
      reconnectionDelayMax: 5000, // Max delay
      forceNew: true, // Force new connection for better recovery
    });

    socket.on('connect', () => {
      AppLogger.success('BRIDGE_CONNECTION', 'WebSocket connected to Python bridge');
      
      // Update connection status
      connectionStatus.connected = true;
      connectionStatus.lastError = null;
      connectionStatus.consecutiveFailures = 0;
      connectionStatus.degradedMode = false;
      connectionStatus.quality = 'excellent';
      connectionStatus.lastPing = Date.now();
      
      // Process queued method calls
      processMethodCallQueue();
      
      // Emit connection status
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: true, 
          health: connectionStatus,
          queueSize: methodCallQueue.length 
        })
      );
      
      sessionTracker.logAction('BRIDGE_WEBSOCKET_CONNECTED', {
        timestamp: Date.now()
      });
    });

    socket.on('disconnect', (reason) => {
      AppLogger.warn('BRIDGE_CONNECTION', 'WebSocket disconnected from Python bridge', { reason });
      
      // Update connection status
      connectionStatus.connected = false;
      connectionStatus.quality = 'critical';
      
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: false, 
          reason,
          health: connectionStatus 
        })
      );
      
      sessionTracker.logAction('BRIDGE_WEBSOCKET_DISCONNECTED', {
        reason,
        timestamp: Date.now()
      });
    });

    socket.on('connect_error', (error) => {
      AppLogger.error('BRIDGE_CONNECTION', 'WebSocket connection error', {
        error: (error as Error).message
      });
      
      connectionStatus.errorCount++;
      connectionStatus.consecutiveFailures++;
      connectionStatus.lastError = (error as Error).message;
      
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: false, 
          error: (error as Error).message,
          health: connectionStatus 
        })
      );
    });

    socket.on('reconnect_attempt', (attemptNumber) => {
      AppLogger.info('BRIDGE_CONNECTION', `WebSocket reconnection attempt ${attemptNumber}`);
      
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: false, 
          attemptingReconnect: true, 
          attemptNumber,
          health: connectionStatus 
        })
      );
    });

    socket.on('reconnect_error', (error) => {
      AppLogger.error('BRIDGE_CONNECTION', 'WebSocket reconnection error', {
        error: (error as Error).message
      });
      
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: false, 
          reconnectionError: (error as Error).message,
          health: connectionStatus 
        })
      );
    });

    socket.on('reconnect_failed', () => {
      AppLogger.error('BRIDGE_CONNECTION', 'WebSocket reconnection failed - entering degraded mode');
      
      connectionStatus.degradedMode = true;
      
      eventListeners.get('connection_status')?.forEach(listener => 
        listener({ 
          connected: false, 
          reconnectionFailed: true,
          degradedMode: true,
          health: connectionStatus 
        })
      );
    });

    // Forward all named events to onPythonEvent listeners
    if ((socket as any).onAny) {
      (socket as any).onAny((eventName: string, ...args: any[]) => {
        try {
          const data = args && args.length === 1 ? args[0] : args;
          const listeners = eventListeners.get(eventName);
          if (listeners && listeners.length > 0) {
            listeners.forEach(listener => {
              try { listener(data); } catch (cbErr) {
                AppLogger.error('BRIDGE_EVENT', 'Listener callback error', { eventName, error: (cbErr as Error).message });
              }
            });
          }
        } catch (err) {
          AppLogger.error('BRIDGE_EVENT', 'onAny dispatch failed', { eventName, error: (err as Error).message });
        }
      });
    }

    // Enhanced message handling with error recovery
    socket.on('message', (message: string) => {
      try {
        const parsed: WebSocketEvent = JSON.parse(message);
        if (parsed.event) {
          eventListeners.get(parsed.event)?.forEach(listener => {
            try {
              listener(parsed.data);
            } catch (callbackError) {
              AppLogger.error('BRIDGE_EVENT', 'Event callback error', {
                event: parsed.event,
                error: (callbackError as Error).message
              });
            }
          });
        }
      } catch (e) {
        AppLogger.error('BRIDGE_EVENT', 'Failed to parse WebSocket message', {
          error: (e as Error).message,
          message
        });
      }
    });

    // Add ping/pong for health monitoring
    socket.on('pong', (data) => {
      const responseTime = Date.now() - connectionStatus.lastPing;
      connectionStatus.responseTime = responseTime;
      
      // Update connection quality based on response time
      if (responseTime < 100) {
        connectionStatus.quality = 'excellent';
      } else if (responseTime < 500) {
        connectionStatus.quality = 'good';
      } else if (responseTime < 2000) {
        connectionStatus.quality = 'poor';
      } else {
        connectionStatus.quality = 'critical';
      }
    });
  }
};

export const callPythonMethod = async <T = any>(method: string, ...args: any[]): Promise<T> => {
  const startTime = Date.now();
  
  // Emit method call start event for monitoring
  emitMethodCallEvent('method_call_start', {
    method,
    timestamp: startTime
  });
  
  // If in degraded mode, try to recover first
  if (connectionStatus.degradedMode) {
    AppLogger.info('BRIDGE_CALL', 'Attempting to exit degraded mode', { method });
    const recovered = await attemptRecovery();
    if (!recovered) {
      const error = new Error('Bridge is in degraded mode and recovery failed');
      emitMethodCallEvent('method_call_error', {
        method,
        error: error.message,
        duration: Date.now() - startTime,
        timestamp: Date.now()
      });
      throw error;
    }
  }
  
  return executeMethodWithRetry(method, args, startTime);
};

// Enhanced method execution with retry logic
const executeMethodWithRetry = async <T = any>(method: string, args: any[], startTime: number, attempt: number = 1): Promise<T> => {
  try {
    const result = await executeMethod<T>(method, args);
    
    // Update success metrics
    connectionStatus.lastSuccessfulCall = Date.now();
    connectionStatus.consecutiveFailures = 0;
    connectionStatus.connected = true;
    connectionStatus.lastError = null;
    
    const duration = Date.now() - startTime;
    AppLogger.debug('BRIDGE_CALL', `Method ${method} completed successfully`, {
      duration,
      attempt
    });
    
    emitMethodCallEvent('method_call_complete', {
      method,
      duration,
      timestamp: Date.now()
    });
    
    return result;
    
  } catch (error) {
    connectionStatus.consecutiveFailures++;
    const errorMessage = (error as Error).message;
    
    AppLogger.warn('BRIDGE_CALL', `Method ${method} failed (attempt ${attempt})`, {
      error: errorMessage,
      consecutiveFailures: connectionStatus.consecutiveFailures
    });
    
    // Determine if we should retry
    const shouldRetry = shouldRetryMethod(errorMessage, attempt);
    
    if (shouldRetry && attempt < retryConfig.maxAttempts) {
      // Calculate retry delay with exponential backoff
      const delay = calculateRetryDelay(attempt);
      
      AppLogger.info('BRIDGE_CALL', `Retrying method ${method} in ${delay}ms`, {
        attempt: attempt + 1,
        maxAttempts: retryConfig.maxAttempts
      });
      
      await new Promise(resolve => setTimeout(resolve, delay));
      
      return executeMethodWithRetry(method, args, startTime, attempt + 1);
    }
    
    // Log final failure
    const duration = Date.now() - startTime;
    AppLogger.error('BRIDGE_CALL', `Method ${method} failed permanently`, {
      error: errorMessage,
      totalAttempts: attempt,
      duration
    });
    
    emitMethodCallEvent('method_call_error', {
      method,
      error: errorMessage,
      duration,
      timestamp: Date.now()
    });
    
    sessionTracker.logAction('BRIDGE_METHOD_FAILED', {
      method,
      error: errorMessage,
      attempts: attempt,
      duration
    });
    
    // If connection error, queue for later processing
    if (isConnectionError(errorMessage)) {
      return queueMethodCall(method, args);
    }
    
    throw error;
  }
};

// Core method execution
const executeMethod = async <T = any>(method: string, args: any[]): Promise<T> => {
  const requestBody = {
    method,
    args, // Always pass args as an array for consistency
    id: Math.floor(Math.random() * 1000000), // Simple request ID
  };

  AppLogger.debug('BRIDGE_CALL', `Executing Python method: ${method}`, { args });
  
  const response = await fetch(`${BASE_URL}/api/call`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    connectionStatus.connected = false;
    connectionStatus.lastError = `HTTP ${response.status}: ${errorText}`;
    throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
  }

  const result: PythonResponse = await response.json();

  if (result.status === 'success') {
    return result.data as T;
  } else {
    throw new Error(result.message || 'Unknown error from Python backend');
  }
};

// Helper functions for retry logic
const shouldRetryMethod = (errorMessage: string, attempt: number): boolean => {
  if (attempt >= retryConfig.maxAttempts) {
    return false;
  }
  
  const retryableErrors = [
    'fetch',
    'Failed to fetch',
    'network',
    'timeout',
    'connection',
    'ECONNREFUSED',
    'ENOTFOUND',
    'ETIMEDOUT'
  ];
  
  return retryableErrors.some(error => 
    errorMessage.toLowerCase().includes(error.toLowerCase())
  );
};

const calculateRetryDelay = (attempt: number): number => {
  const baseDelay = retryConfig.baseDelay;
  const backoffDelay = baseDelay * Math.pow(retryConfig.backoffMultiplier, attempt - 1);
  const jitter = retryConfig.jitter ? Math.random() * 1000 : 0;
  
  return Math.min(backoffDelay + jitter, retryConfig.maxDelay);
};

const isConnectionError = (errorMessage: string): boolean => {
  const connectionErrors = [
    'fetch',
    'Failed to fetch',
    'network',
    'connection',
    'ECONNREFUSED',
    'ENOTFOUND'
  ];
  
  return connectionErrors.some(error => 
    errorMessage.toLowerCase().includes(error.toLowerCase())
  );
};

// Method call queue management
const queueMethodCall = async <T = any>(method: string, args: any[]): Promise<T> => {
  return new Promise((resolve, reject) => {
    methodCallQueue.push({
      method,
      args,
      resolve,
      reject,
      timestamp: Date.now()
    });
    
    AppLogger.info('BRIDGE_QUEUE', `Method ${method} queued for later execution`, {
      queueSize: methodCallQueue.length
    });
    
    // Clean up old queued calls (older than 5 minutes)
    cleanupMethodCallQueue();
  });
};

const processMethodCallQueue = async (): Promise<void> => {
  if (isProcessingQueue || methodCallQueue.length === 0) {
    return;
  }
  
  isProcessingQueue = true;
  
  AppLogger.info('BRIDGE_QUEUE', `Processing ${methodCallQueue.length} queued method calls`);
  
  while (methodCallQueue.length > 0 && connectionStatus.connected) {
    const queuedCall = methodCallQueue.shift();
    if (!queuedCall) break;
    
    try {
      const result = await executeMethod(queuedCall.method, queuedCall.args);
      queuedCall.resolve(result);
    } catch (error) {
      queuedCall.reject(error);
    }
  }
  
  isProcessingQueue = false;
};

const cleanupMethodCallQueue = (): void => {
  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  const originalLength = methodCallQueue.length;
  
  methodCallQueue = methodCallQueue.filter(call => {
    if (call.timestamp < fiveMinutesAgo) {
      call.reject(new Error('Method call expired in queue'));
      return false;
    }
    return true;
  });
  
  const removedCount = originalLength - methodCallQueue.length;
  if (removedCount > 0) {
    AppLogger.warn('BRIDGE_QUEUE', `Removed ${removedCount} expired method calls from queue`);
  }
};

// Event emission for monitoring
const emitMethodCallEvent = (eventType: string, data: any): void => {
  const callbacks = eventListeners.get(eventType);
  if (callbacks) {
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        AppLogger.error('BRIDGE_EVENT', 'Method call event callback error', {
          eventType,
          error: (error as Error).message
        });
      }
    });
  }
};

// Recovery mechanism
const attemptRecovery = async (): Promise<boolean> => {
  AppLogger.info('BRIDGE_RECOVERY', 'Attempting bridge recovery');
  
  try {
    // Try to reconnect WebSocket
    if (socket) {
      socket.disconnect();
      socket = null;
    }
    
    ensureWebSocketConnection();
    
    // Wait a moment for connection
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Test connection
    const isHealthy = await checkBridgeHealth();
    
    if (isHealthy) {
      connectionStatus.degradedMode = false;
      connectionStatus.consecutiveFailures = 0;
      AppLogger.success('BRIDGE_RECOVERY', 'Bridge recovery successful');
      return true;
    } else {
      AppLogger.error('BRIDGE_RECOVERY', 'Bridge recovery failed - health check failed');
      return false;
    }
  } catch (error) {
    AppLogger.error('BRIDGE_RECOVERY', 'Bridge recovery attempt failed', {
      error: (error as Error).message
    });
    return false;
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
  AppLogger.info('BRIDGE_CONNECTION', 'Disconnecting from Python bridge');
  
  // Clear method call queue
  clearQueue();
  
  // Disconnect socket
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  
  // Reset connection state
  connectionStatus.connected = false;
  connectionStatus.degradedMode = false;
  connectionStatus.consecutiveFailures = 0;
  
  // Update connection health
  connectionStatus.quality = 'critical';
  
  // Clear event listeners
  eventListeners.clear();
  
  sessionTracker.logAction('BRIDGE_DISCONNECTED_MANUAL', {
    timestamp: Date.now()
  });
  
  AppLogger.info('BRIDGE_CONNECTION', 'Disconnected from Python bridge');
};

// Enhanced health check for the bridge
export const checkBridgeHealth = async (): Promise<boolean> => {
  try {
    const startTime = Date.now();
    const response = await fetch(`${BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const responseTime = Date.now() - startTime;
    connectionStatus.responseTime = responseTime;
    
    if (response.ok) {
      const data = await response.json();
      
      AppLogger.debug('BRIDGE_HEALTH', 'Bridge health check passed', {
        responseTime,
        status: data.status
      });
      
      // Update connection quality based on response time
      if (responseTime < 100) {
        connectionStatus.quality = 'excellent';
      } else if (responseTime < 500) {
        connectionStatus.quality = 'good';
      } else if (responseTime < 2000) {
        connectionStatus.quality = 'poor';
      } else {
        connectionStatus.quality = 'critical';
      }
      
      connectionStatus.lastPing = Date.now();
      return data.status === 'healthy';
    }
    
    AppLogger.warn('BRIDGE_HEALTH', 'Bridge health check failed - bad response', {
      status: response.status,
      responseTime
    });
    return false;
    
  } catch (error) {
    AppLogger.error('BRIDGE_HEALTH', 'Bridge health check error', {
      error: (error as Error).message
    });
    connectionStatus.errorCount++;
    return false;
  }
};

// Get comprehensive connection status
export const getConnectionHealth = () => {
  return { ...connectionStatus };
};

// Get connection statistics
export const getConnectionStats = () => {
  return {
    isConnected: connectionStatus.connected,
    retryCount: connectionStatus.retryCount,
    consecutiveFailures: connectionStatus.consecutiveFailures,
    degradedMode: connectionStatus.degradedMode,
    queueSize: methodCallQueue.length,
    lastSuccessfulCall: connectionStatus.lastSuccessfulCall,
    connectionHealth: connectionStatus,
    uptime: connectionStatus.connected ? Date.now() - connectionStatus.lastPing : 0
  };
};

// Force reconnection (for manual recovery)
export const forceReconnect = async (): Promise<boolean> => {
  AppLogger.info('BRIDGE_RECOVERY', 'Force reconnection requested');
  
  // Reset connection state
  connectionStatus.connected = false;
  connectionStatus.degradedMode = false;
  connectionStatus.retryCount = 0;
  connectionStatus.consecutiveFailures = 0;
  
  // Disconnect current socket
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  
  // Clear method call queue
  clearQueue();
  
  // Wait a moment before reconnecting
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Attempt new connection
  try {
    ensureWebSocketConnection();
    const isHealthy = await checkBridgeHealth();
    
    if (isHealthy) {
      AppLogger.success('BRIDGE_RECOVERY', 'Force reconnection successful');
      return true;
    } else {
      AppLogger.error('BRIDGE_RECOVERY', 'Force reconnection failed - health check failed');
      return false;
    }
  } catch (error) {
    AppLogger.error('BRIDGE_RECOVERY', 'Force reconnection attempt failed', {
      error: (error as Error).message
    });
    return false;
  }
};

// Get queue status
export const getQueueStatus = () => {
  return {
    size: methodCallQueue.length,
    isProcessing: isProcessingQueue,
    oldestCall: methodCallQueue.length > 0 ? 
      Date.now() - methodCallQueue[0].timestamp : 0
  };
};

// Clear method call queue
export const clearQueue = (): void => {
  const queueSize = methodCallQueue.length;
  
  // Reject all queued calls
  methodCallQueue.forEach(call => {
    call.reject(new Error('Queue cleared manually'));
  });
  
  methodCallQueue = [];
  
  AppLogger.info('BRIDGE_QUEUE', `Cleared ${queueSize} queued method calls`);
};

// Enable/disable degraded mode
export const setDegradedMode = (enabled: boolean): void => {
  connectionStatus.degradedMode = enabled;
  AppLogger.info('BRIDGE_MODE', `Degraded mode ${enabled ? 'enabled' : 'disabled'}`);
  
  sessionTracker.logAction('BRIDGE_DEGRADED_MODE', {
    enabled,
    timestamp: Date.now()
  });
};

export const generateQuestion = (topic: string, difficulty: string) => {
  return callPythonMethod('generate_question', topic, difficulty);
};

export const submitAnswer = (questionId: string, answer: string) => {
  return callPythonMethod('submit_answer', questionId, answer);
};

export const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    AppLogger.info('BRIDGE_UPLOAD', 'Starting file upload', {
      fileName: file.name,
      fileSize: file.size
    });

    const response = await fetch(`${BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      AppLogger.error('BRIDGE_UPLOAD', 'File upload failed', {
        status: response.status,
        error: errorText
      });
      throw new Error(`File upload failed: ${errorText}`);
    }

    const result = await response.json();
    
    AppLogger.success('BRIDGE_UPLOAD', 'File upload completed', {
      fileName: file.name,
      result
    });

    return result;
  } catch (error) {
    AppLogger.error('BRIDGE_UPLOAD', 'File upload error', {
      fileName: file.name,
      error: (error as Error).message
    });
    throw error;
  }
};

export const startTraining = (config: object) => {
  return callPythonMethod('start_training', config);
};

export const getSettings = () => {
  return callPythonMethod('get_settings');
};

export const updateSettings = (settings: object) => {
  return callPythonMethod('update_settings', settings);
};

export const getStatus = () => {
  return callPythonMethod('get_status');
};

export const testConnection = () => {
  return callPythonMethod('test_connection');
};

// Enhanced connection status getter (legacy compatibility)
export const getConnectionStatus = () => {
  return { 
    connected: connectionStatus.connected,
    lastError: connectionStatus.lastError,
    retryCount: connectionStatus.retryCount
  };
};

// Add method to get question history with proper typing
export const getQuestionHistory = (filters: { limit?: number; offset?: number } = {}) => {
  return callPythonMethod('get_question_history', filters);
};
