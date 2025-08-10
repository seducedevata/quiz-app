import { AppLogger } from './logger';

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock as any;

// Mock pythonBridge
jest.mock('./pythonBridge', () => ({
  callPythonMethod: jest.fn().mockResolvedValue({}),
}));

describe('AppLogger', () => {
  let consoleSpy: jest.SpyInstance;

  beforeEach(() => {
    consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    jest.clearAllMocks();
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  it('should log info messages correctly', () => {
    AppLogger.log('INFO', 'TestCategory', 'This is an info message');
    
    expect(consoleSpy).toHaveBeenCalled();
    const logCall = consoleSpy.mock.calls[0];
    expect(logCall[0]).toContain('💡');
    expect(logCall[0]).toContain('INFO');
    expect(logCall[0]).toContain('TestCategory');
    expect(logCall[0]).toContain('This is an info message');
  });

  it('should log error messages correctly', () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation();
    AppLogger.log('ERROR', 'TestCategory', 'This is an error message', { error: 'details' });
    
    expect(errorSpy).toHaveBeenCalled();
    const errorCall = errorSpy.mock.calls[0];
    expect(errorCall[0]).toContain('❌');
    expect(errorCall[0]).toContain('ERROR');
    expect(errorCall[0]).toContain('TestCategory');
    expect(errorCall[0]).toContain('This is an error message');
    expect(errorCall[0]).toContain('Session:');
    expect(errorCall[2]).toEqual({ error: 'details' }); // Data is now the third parameter
    
    errorSpy.mockRestore();
  });

  it('should track user actions', () => {
    AppLogger.trackUserAction('Button Click', { buttonId: 'submitBtn' });
    
    expect(consoleSpy).toHaveBeenCalled();
    const logCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1];
    expect(logCall[0]).toContain('👤'); // USER emoji, not ACTION emoji
    expect(logCall[0]).toContain('USER');
    expect(logCall[0]).toContain('USER_ACTION');
    expect(logCall[0]).toContain('Button Click');
    expect(logCall[0]).toContain('Session:');
  });

  it('should track navigation', () => {
    AppLogger.trackNavigation('/old-page', '/new-page');
    
    expect(consoleSpy).toHaveBeenCalled();
    const logCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1];
    expect(logCall[0]).toContain('🎯'); // ACTION emoji for navigation
    expect(logCall[0]).toContain('ACTION');
    expect(logCall[0]).toContain('NAVIGATION');
    expect(logCall[0]).toContain('Screen Navigation: /old-page → /new-page');
    expect(logCall[0]).toContain('Session:');
  });

  it('should track performance metrics', () => {
    AppLogger.performanceMetric('dataFetch', 123.45, true);
    
    expect(consoleSpy).toHaveBeenCalled();
    const logCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1];
    expect(logCall[0]).toContain('⏱️');
    expect(logCall[0]).toContain('PERF');
    expect(logCall[0]).toContain('METRICS');
    expect(logCall[0]).toContain('dataFetch: 123.45ms ✅'); // Updated format
    expect(logCall[0]).toContain('Session:');
  });

  it('should store critical logs', () => {
    localStorageMock.getItem.mockReturnValue('[]');
    
    // Mock the storeCriticalLog method to test localStorage interaction
    const storeSpy = jest.spyOn(AppLogger, 'storeCriticalLog');
    AppLogger.log('ERROR', 'TestCategory', 'Critical error message');
    
    expect(storeSpy).toHaveBeenCalledWith(
      'ERROR',
      'TestCategory', 
      'Critical error message',
      null, // Data is null when not provided
      expect.any(String)
    );
    
    storeSpy.mockRestore();
  });

  it('should get session summary', () => {
    const summary = AppLogger.getSessionSummary();
    
    expect(summary).toHaveProperty('sessionId');
    expect(summary).toHaveProperty('startTime');
    expect(summary).toHaveProperty('duration');
    expect(summary).toHaveProperty('actionCount');
    expect(summary).toHaveProperty('currentScreen');
  });

  it('should clear critical logs', () => {
    // Test that the method exists and can be called
    expect(() => AppLogger.clearCriticalLogs()).not.toThrow();
    
    // Check that an info log was generated
    expect(consoleSpy).toHaveBeenCalled();
    const logCall = consoleSpy.mock.calls[consoleSpy.mock.calls.length - 1];
    expect(logCall[0]).toContain('Critical logs cleared');
  });

  it('should increment action count for user actions', () => {
    const initialCount = AppLogger.actionCount;
    
    AppLogger.log('ACTION', 'TEST', 'Test action');
    expect(AppLogger.actionCount).toBe(initialCount + 1);
    
    AppLogger.log('USER', 'TEST', 'Test user action');
    expect(AppLogger.actionCount).toBe(initialCount + 2);
    
    AppLogger.log('INFO', 'TEST', 'Test info');
    expect(AppLogger.actionCount).toBe(initialCount + 2); // Should not increment
  });

  it('should set current screen', () => {
    AppLogger.setCurrentScreen('quiz');
    expect(AppLogger.currentScreen).toBe('quiz');
  });
});