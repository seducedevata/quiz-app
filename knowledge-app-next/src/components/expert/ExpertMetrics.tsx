'use client';

import React, { useState, useEffect } from 'react';
import { AppLogger } from '../../lib/logger';
import { onPythonEvent, offPythonEvent, callPythonMethod } from '../../lib/pythonBridge';

interface MetricData {
  value: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
  history: number[];
}

interface ExpertMetricsData {
  performance: {
    avgThinkingTime: MetricData;
    avgResponseTime: MetricData;
    successRate: MetricData;
    throughput: MetricData;
  };
  quality: {
    complexityScore: MetricData;
    researchDepth: MetricData;
    accuracyScore: MetricData;
    innovationIndex: MetricData;
  };
  efficiency: {
    tokenEfficiency: MetricData;
    memoryUsage: MetricData;
    cacheHitRate: MetricData;
    errorRate: MetricData;
  };
  usage: {
    totalQuestions: number;
    expertQuestions: number;
    documentsProcessed: number;
    avgSessionLength: number;
  };
}

interface ExpertMetricsProps {
  isEnabled: boolean;
  refreshInterval?: number;
}

export const ExpertMetrics: React.FC<ExpertMetricsProps> = ({
  isEnabled,
  refreshInterval = 30000 // 30 seconds
}) => {
  const [metrics, setMetrics] = useState<ExpertMetricsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<'performance' | 'quality' | 'efficiency' | 'usage'>('performance');
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  // Fetch metrics from backend
  const fetchMetrics = async () => {
    if (!isEnabled) return;

    try {
      setIsLoading(true);
      const data = await callPythonMethod('get_expert_metrics', {
        time_range: timeRange,
        include_history: true
      });
      
      setMetrics(data);
      setLastUpdated(new Date());
      AppLogger.info('EXPERT_METRICS', 'Metrics updated successfully', {
        timeRange,
        metricsCount: Object.keys(data).length
      });
    } catch (error) {
      AppLogger.error('EXPERT_METRICS', 'Failed to fetch metrics', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Set up periodic refresh
  useEffect(() => {
    if (!isEnabled) return;

    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);

    return () => clearInterval(interval);
  }, [isEnabled, timeRange, refreshInterval]);

  if (!isEnabled) {
    return (
      <div className="expert-metrics-disabled text-center py-8 text-gray-500">
        <div className="text-4xl mb-2">📊</div>
        <p>Expert metrics are disabled</p>
        <p className="text-sm">Enable advanced metrics in Expert Mode settings</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="expert-metrics-loading text-center py-8">
        <div className="text-4xl mb-2">⏳</div>
        <p className="text-gray-300">Loading expert metrics...</p>
        {isLoading && (
          <div className="mt-4">
            <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto"></div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="expert-metrics bg-gray-900/50 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="text-2xl">📊</div>
          <div>
            <h3 className="text-xl font-bold text-white">Expert Metrics</h3>
            <p className="text-sm text-gray-400">
              {lastUpdated ? `Last updated: ${lastUpdated.toLocaleTimeString()}` : 'Loading...'}
            </p>
          </div>
        </div>
      </div>

      <div className="text-center py-8 text-gray-400">
        <div className="text-4xl mb-2">🚧</div>
        <p>Expert metrics implementation in progress</p>
        <p className="text-sm">Advanced metrics will be available soon</p>
      </div>
    </div>
  );
};

export default ExpertMetrics;