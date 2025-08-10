// Initialize comprehensive logging system matching Qt implementation
import { AppLogger } from './logger';
import { sessionTracker } from './sessionTracker';
import './debugApp'; // This will set up window.debugApp and global error handlers

// Initialize logging system on app startup
export function initializeLoggingSystem(): void {
  // Log system initialization
  AppLogger.info('SYSTEM', '🚀 Initializing Knowledge App Logging System');
  
  // Set up session tracking
  sessionTracker.logAction('APP_INITIALIZATION', {
    timestamp: Date.now(),
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent.slice(0, 100) : '',
    url: typeof window !== 'undefined' ? window.location.href : ''
  });
  
  // Log system capabilities
  AppLogger.info('SYSTEM', 'Logging system capabilities initialized', {
    sessionTracking: true,
    criticalLogStorage: true,
    pythonBridgeIntegration: true,
    debugConsoleTools: true,
    globalErrorHandling: true,
    performanceMonitoring: true
  });
  
  // Set up performance monitoring
  if (typeof window !== 'undefined' && window.performance) {
    // Monitor page load performance
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        AppLogger.performanceMetric('PAGE_LOAD', navigation.loadEventEnd - navigation.fetchStart, true, {
          domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
          domInteractive: navigation.domInteractive - navigation.fetchStart,
          firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
          firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
        });
      }
    });
    
    // Monitor resource loading
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.entryType === 'resource' && entry.duration > 1000) {
          AppLogger.performanceMetric('SLOW_RESOURCE', entry.duration, false, {
            name: entry.name,
            type: (entry as PerformanceResourceTiming).initiatorType
          });
        }
      });
    });
    
    try {
      observer.observe({ entryTypes: ['resource'] });
    } catch (e) {
      // PerformanceObserver not supported in all browsers
      AppLogger.debug('SYSTEM', 'PerformanceObserver not supported');
    }
  }
  
  // Set up memory monitoring (if available)
  if (typeof window !== 'undefined' && (window.performance as any)?.memory) {
    const checkMemory = () => {
      const memory = (window.performance as any).memory;
      const memoryUsage = {
        used: Math.round(memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(memory.jsHeapSizeLimit / 1024 / 1024)
      };
      
      // Log if memory usage is high
      if (memoryUsage.used > 100) { // More than 100MB
        AppLogger.performanceMetric('MEMORY_USAGE', memoryUsage.used, memoryUsage.used < 200, memoryUsage);
      }
    };
    
    // Check memory every 30 seconds
    setInterval(checkMemory, 30000);
  }
  
  AppLogger.success('SYSTEM', 'Comprehensive logging system initialized successfully');
}

// Auto-initialize when module is imported
if (typeof window !== 'undefined') {
  // Initialize immediately if DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLoggingSystem);
  } else {
    initializeLoggingSystem();
  }
}