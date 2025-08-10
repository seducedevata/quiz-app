// Simplified bridge monitoring service for function-based pythonBridge
import { AppLogger } from './logger';
import { sessionTracker } from './sessionTracker';
import { 
  onPythonEvent, 
  checkBridgeHealth, 
  getConnectionHealth, 
  getConnectionStats, 
  getQueueStatus 
} from './pythonBridge';

interface BridgeMetrics {
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  averageResponseTime: number;
  uptime: number;
  downtime: number;
  reconnections: number;
  lastError: string | null;
  lastErrorTime: number | null;
}

class SimpleBridgeMonitoringService {
  private metrics: BridgeMetrics = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    averageResponseTime: 0,
    uptime: 0,
    downtime: 0,
    reconnections: 0,
    lastError: null,
    lastErrorTime: null
  };

  private monitoringInterval: NodeJS.Timeout | null = null;
  private connectionStartTime: number | null = null;
  private lastDisconnectTime: number | null = null;
  private isMonitoring = false;

  // Start monitoring
  startMonitoring(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    AppLogger.info('BRIDGE_MONITOR', 'Starting bridge monitoring service');

    // Monitor every 30 seconds
    this.monitoringInterval = setInterval(() => {
      this.performMonitoringCheck();
    }, 30000);

    // Listen for connection status changes
    onPythonEvent('connection_status', (status: any) => {
      if (status.connected) {
        this.onConnectionEstablished();
      } else {
        this.onConnectionLost();
      }
    });

    // Listen for method call events if available
    onPythonEvent('method_call_start', (data: any) => {
      this.onMethodCallStart(data);
    });

    onPythonEvent('method_call_complete', (data: any) => {
      this.onMethodCallComplete(data);
    });

    onPythonEvent('method_call_error', (data: any) => {
      this.onMethodCallError(data);
    });

    sessionTracker.logAction('BRIDGE_MONITORING_STARTED', {
      timestamp: Date.now()
    });
  }

  // Stop monitoring
  stopMonitoring(): void {
    if (!this.isMonitoring) return;

    this.isMonitoring = false;
    AppLogger.info('BRIDGE_MONITOR', 'Stopping bridge monitoring service');

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    sessionTracker.logAction('BRIDGE_MONITORING_STOPPED', {
      timestamp: Date.now(),
      finalMetrics: this.metrics
    });
  }

  // Connection event handlers
  private onConnectionEstablished(): void {
    this.connectionStartTime = Date.now();
    
    if (this.lastDisconnectTime) {
      this.metrics.downtime += Date.now() - this.lastDisconnectTime;
      this.lastDisconnectTime = null;
    }

    this.metrics.reconnections++;

    AppLogger.success('BRIDGE_MONITOR', 'Bridge connection established', {
      reconnections: this.metrics.reconnections
    });
  }

  private onConnectionLost(): void {
    if (this.connectionStartTime) {
      this.metrics.uptime += Date.now() - this.connectionStartTime;
      this.connectionStartTime = null;
    }

    this.lastDisconnectTime = Date.now();

    AppLogger.warn('BRIDGE_MONITOR', 'Bridge connection lost', {
      totalUptime: this.metrics.uptime,
      reconnections: this.metrics.reconnections
    });
  }

  // Method call event handlers
  private onMethodCallStart(data: { method: string; timestamp: number }): void {
    AppLogger.debug('BRIDGE_MONITOR', `Method call started: ${data.method}`);
  }

  private onMethodCallComplete(data: { method: string; duration: number; timestamp: number }): void {
    this.metrics.totalCalls++;
    this.metrics.successfulCalls++;

    // Update average response time
    this.updateAverageResponseTime(data.duration);

    AppLogger.debug('BRIDGE_MONITOR', `Method call completed: ${data.method}`, {
      duration: data.duration,
      successRate: this.getSuccessRate()
    });
  }

  private onMethodCallError(data: { method: string; error: string; duration: number; timestamp: number }): void {
    this.metrics.totalCalls++;
    this.metrics.failedCalls++;
    this.metrics.lastError = data.error;
    this.metrics.lastErrorTime = data.timestamp;

    AppLogger.warn('BRIDGE_MONITOR', `Method call failed: ${data.method}`, {
      error: data.error,
      duration: data.duration,
      errorRate: this.getErrorRate()
    });
  }

  // Monitoring check
  private async performMonitoringCheck(): Promise<void> {
    try {
      const health = getConnectionHealth();
      const stats = getConnectionStats();
      const queueStatus = getQueueStatus();

      AppLogger.debug('BRIDGE_MONITOR', 'Periodic monitoring check', {
        health: health.quality,
        responseTime: health.responseTime,
        errorCount: health.errorCount,
        queueSize: queueStatus.size
      });

      // Check if bridge is responsive
      if (health.connected) {
        const isHealthy = await checkBridgeHealth();
        if (!isHealthy) {
          AppLogger.warn('BRIDGE_MONITOR', 'Bridge health check failed during monitoring');
        }
      }

      // Check queue size
      if (queueStatus.size > 10) {
        AppLogger.warn('BRIDGE_MONITOR', 'Large method call queue detected', {
          queueSize: queueStatus.size
        });
      }

      // Update uptime if connected
      if (this.connectionStartTime) {
        this.metrics.uptime = Date.now() - this.connectionStartTime;
      }

    } catch (error) {
      AppLogger.error('BRIDGE_MONITOR', 'Monitoring check failed', {
        error: (error as Error).message
      });
    }
  }

  // Metrics calculation
  private updateAverageResponseTime(newResponseTime: number): void {
    if (this.metrics.successfulCalls === 1) {
      this.metrics.averageResponseTime = newResponseTime;
    } else {
      // Calculate running average
      this.metrics.averageResponseTime = 
        (this.metrics.averageResponseTime * (this.metrics.successfulCalls - 1) + newResponseTime) / 
        this.metrics.successfulCalls;
    }
  }

  private getSuccessRate(): number {
    if (this.metrics.totalCalls === 0) return 1;
    return this.metrics.successfulCalls / this.metrics.totalCalls;
  }

  private getErrorRate(): number {
    if (this.metrics.totalCalls === 0) return 0;
    return this.metrics.failedCalls / this.metrics.totalCalls;
  }

  private getUptimeRatio(): number {
    const totalTime = this.metrics.uptime + this.metrics.downtime;
    if (totalTime === 0) return 1;
    return this.metrics.uptime / totalTime;
  }

  // Public API
  getMetrics(): BridgeMetrics {
    return { ...this.metrics };
  }

  resetMetrics(): void {
    this.metrics = {
      totalCalls: 0,
      successfulCalls: 0,
      failedCalls: 0,
      averageResponseTime: 0,
      uptime: 0,
      downtime: 0,
      reconnections: 0,
      lastError: null,
      lastErrorTime: null
    };
    
    AppLogger.info('BRIDGE_MONITOR', 'Bridge metrics reset');
  }

  // Generate report
  generateReport(): any {
    const now = Date.now();
    const uptimeRatio = this.getUptimeRatio();
    const successRate = this.getSuccessRate();
    const errorRate = this.getErrorRate();

    return {
      timestamp: now,
      metrics: this.metrics,
      performance: {
        uptimeRatio,
        successRate,
        errorRate,
        averageResponseTime: this.metrics.averageResponseTime
      },
      status: {
        healthy: successRate > 0.9 && uptimeRatio > 0.95 && errorRate < 0.1,
        degraded: successRate > 0.7 && uptimeRatio > 0.8,
        critical: successRate < 0.7 || uptimeRatio < 0.8
      },
      recommendations: this.generateRecommendations()
    };
  }

  private generateRecommendations(): string[] {
    const recommendations: string[] = [];
    const errorRate = this.getErrorRate();
    const uptimeRatio = this.getUptimeRatio();

    if (errorRate > 0.1) {
      recommendations.push('High error rate detected - check Python backend logs');
    }

    if (uptimeRatio < 0.9) {
      recommendations.push('Low uptime ratio - investigate connection stability');
    }

    if (this.metrics.averageResponseTime > 2000) {
      recommendations.push('High response times - check system performance');
    }

    if (this.metrics.reconnections > 5) {
      recommendations.push('Frequent reconnections - check network stability');
    }

    return recommendations;
  }
}

// Create global bridge monitoring service instance
export const bridgeMonitoringService = new SimpleBridgeMonitoringService();

// Make it globally available for debugging
if (typeof window !== 'undefined') {
  (window as any).bridgeMonitoringService = bridgeMonitoringService;
}