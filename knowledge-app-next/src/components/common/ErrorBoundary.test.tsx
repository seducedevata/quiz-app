import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';
import { AppLogger } from '../../lib/logger';

// Mock the logger and other dependencies
jest.mock('../../lib/logger', () => ({
  AppLogger: {
    trackError: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    action: jest.fn(),
    getSessionSummary: jest.fn(() => ({
      sessionId: 'test-session-123',
      currentScreen: 'test-screen',
      startTime: '2025-01-01T00:00:00.000Z',
      duration: '10.00s',
      actionCount: 5,
    })),
    sessionId: 'test-session-123',
    currentScreen: 'test-screen',
  },
}));
jest.mock('../../lib/sessionTracker', () => ({
  sessionTracker: {
    logAction: jest.fn(),
    getSessionAnalytics: jest.fn(() => ({ totalActions: 5 })),
  },
}));
jest.mock('../../lib/pythonBridge', () => ({
  callPythonMethod: jest.fn(),
  checkBridgeHealth: jest.fn().mockResolvedValue(true),
}));

// Component that throws an error for testing
const ThrowError: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <div>No error</div>;
};

describe('ErrorBoundary', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock console.error to avoid noise in tests
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should render children when there is no error', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByText('No error')).toBeInTheDocument();
  });

  it('should render error UI when child component throws', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText(/Test error message/)).toBeInTheDocument();
  });

  it('should classify network errors correctly', () => {
    const NetworkError = () => {
      throw new Error('Network connection failed');
    };

    render(
      <ErrorBoundary>
        <NetworkError />
      </ErrorBoundary>
    );

    expect(screen.getByText('🌐')).toBeInTheDocument(); // Network error icon
    expect(screen.getByText(/network connectivity issue/i)).toBeInTheDocument();
  });

  it('should classify Python bridge errors correctly', () => {
    const BridgeError = () => {
      throw new Error('Python bridge connection lost');
    };

    render(
      <ErrorBoundary>
        <BridgeError />
      </ErrorBoundary>
    );

    expect(screen.getByText('🔗')).toBeInTheDocument(); // Bridge error icon
    expect(screen.getByText(/communicating with the Python backend/i)).toBeInTheDocument();
  });

  it('should show retry button and handle retry', async () => {
    let shouldThrow = true;
    const RetryableComponent = () => {
      if (shouldThrow) {
        throw new Error('Retryable error');
      }
      return <div>Retry successful</div>;
    };

    const { rerender } = render(
      <ErrorBoundary>
        <RetryableComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    const retryButton = screen.getByText(/Try Again/);
    expect(retryButton).toBeInTheDocument();

    // Simulate successful retry
    shouldThrow = false;
    fireEvent.click(retryButton);

    // The component should re-render without error
    await waitFor(() => {
      expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    });
  });

  it('should show go home button and handle navigation', () => {
    // Mock window.location
    delete (window as any).location;
    window.location = { href: '' } as any;

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const homeButton = screen.getByText(/Go to Home/);
    fireEvent.click(homeButton);

    expect(window.location.href).toBe('/');
  });

  it('should show reload button and handle page reload', () => {
    // Mock window.location.reload
    const mockReload = jest.fn();
    delete (window as any).location;
    window.location = { reload: mockReload } as any;

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const reloadButton = screen.getByText(/Reload Page/);
    fireEvent.click(reloadButton);

    expect(mockReload).toHaveBeenCalled();
  });

  it('should handle copy error report', async () => {
    // Mock clipboard API
    const mockWriteText = jest.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: mockWriteText,
      },
    });

    // Mock alert
    window.alert = jest.fn();

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const reportButton = screen.getByText(/Copy Error Report/);
    fireEvent.click(reportButton);

    await waitFor(() => {
      expect(mockWriteText).toHaveBeenCalled();
    });

    const clipboardContent = mockWriteText.mock.calls[0][0];
    const reportData = JSON.parse(clipboardContent);
    
    expect(reportData).toHaveProperty('errorId');
    expect(reportData).toHaveProperty('message', 'Test error message');
    expect(reportData).toHaveProperty('sessionId');
  });

  it('should disable retry button after max attempts', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const retryButton = screen.getByText(/Try Again/);
    
    // Click retry button multiple times to exceed max attempts
    fireEvent.click(retryButton);
    fireEvent.click(retryButton);
    fireEvent.click(retryButton);
    fireEvent.click(retryButton); // This should exceed max attempts

    expect(retryButton).toBeDisabled();
    expect(screen.getByText(/Max attempts reached/)).toBeInTheDocument();
  });

  it('should call custom error handler when provided', () => {
    const mockErrorHandler = jest.fn();

    render(
      <ErrorBoundary onError={mockErrorHandler}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(mockErrorHandler).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it('should use custom fallback component when provided', () => {
    const CustomFallback: React.FC<any> = ({ error, retry }) => (
      <div>
        <h1>Custom Error UI</h1>
        <p>{error.message}</p>
        <button onClick={retry}>Custom Retry</button>
      </div>
    );

    render(
      <ErrorBoundary fallbackComponent={CustomFallback}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom Error UI')).toBeInTheDocument();
    expect(screen.getByText('Custom Retry')).toBeInTheDocument();
  });

  it('should log error details to AppLogger', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(AppLogger.trackError).toHaveBeenCalledWith(
      'REACT_ERROR_BOUNDARY',
      expect.any(Error),
      expect.objectContaining({
        errorId: expect.any(String),
        classification: expect.any(Object),
        componentStack: expect.any(String),
      })
    );
  });
});