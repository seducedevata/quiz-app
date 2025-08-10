import React, { useState, useEffect } from 'react';
import { getConnectionStatus, checkBridgeHealth } from '../../lib/pythonBridge';
import { AppLogger } from '../../lib/logger';

interface ConnectionStatusProps {
  showDetails?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ showDetails = false }) => {
  const [status, setStatus] = useState({
    connected: false,
    lastError: null as string | null,
    retryCount: 0,
    bridgeHealthy: false
  });
  const [isChecking, setIsChecking] = useState(false);

  const checkConnection = async () => {
    setIsChecking(true);
    try {
      const bridgeHealthy = await checkBridgeHealth();
      const connectionStatus = getConnectionStatus();
      
      setStatus({
        ...connectionStatus,
        bridgeHealthy
      });
      
      AppLogger.info('CONNECTION', 'Connection status updated', { 
        connected: connectionStatus.connected, 
        bridgeHealthy 
      });
    } catch (error) {
      AppLogger.error('CONNECTION', 'Failed to check connection status', error);
      setStatus(prev => ({
        ...prev,
        connected: false,
        bridgeHealthy: false,
        lastError: (error as Error).message
      }));
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    // Check connection status on mount
    checkConnection();
    
    // Set up periodic health checks
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    if (isChecking) return '🔄';
    if (status.connected && status.bridgeHealthy) return '🟢';
    if (status.connected && !status.bridgeHealthy) return '🟡';
    return '🔴';
  };

  const getStatusText = () => {
    if (isChecking) return 'Checking...';
    if (status.connected && status.bridgeHealthy) return 'Connected';
    if (status.connected && !status.bridgeHealthy) return 'Bridge Issues';
    return 'Disconnected';
  };

  const getStatusColor = () => {
    if (status.connected && status.bridgeHealthy) return 'text-green-600';
    if (status.connected && !status.bridgeHealthy) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="connection-status flex items-center space-x-2">
      <span className="text-lg">{getStatusIcon()}</span>
      <span className={`text-sm font-medium ${getStatusColor()}`}>
        {getStatusText()}
      </span>
      
      {showDetails && (
        <div className="connection-details text-xs text-text-secondary">
          {status.lastError && (
            <div className="error-details text-red-500 mt-1">
              Error: {status.lastError}
            </div>
          )}
          {status.retryCount > 0 && (
            <div className="retry-count mt-1">
              Retries: {status.retryCount}
            </div>
          )}
        </div>
      )}
      
      <button
        onClick={checkConnection}
        disabled={isChecking}
        className="text-xs px-2 py-1 bg-bg-secondary rounded hover:bg-bg-tertiary transition-colors"
        title="Check connection status"
      >
        {isChecking ? '⏳' : '🔄'}
      </button>
    </div>
  );
};