import React, { useState, useEffect } from 'react';
import { AppLogger } from '../../lib/logger';

interface LogEntry {
  level: string;
  category: string;
  message: string;
  timestamp: string;
  data?: any;
}

export const LogDisplay: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);

  useEffect(() => {
    // This is a simplified way to capture logs. In a real app, you might
    // want to modify AppLogger to emit events or use a centralized state management.
    const originalLog = AppLogger.log;
    AppLogger.log = (level, category, message, data) => {
      originalLog(level, category, message, data);
      setLogs(prevLogs => [{
        level,
        category,
        message,
        timestamp: new Date().toISOString(),
        data
      }, ...prevLogs.slice(0, 99)]); // Keep last 100 logs
    };

    return () => {
      AppLogger.log = originalLog; // Clean up on unmount
    };
  }, []);

  return (
    <div className="log-display bg-gray-800 text-gray-200 p-4 rounded-md font-mono text-xs max-h-64 overflow-y-auto">
      <h3 className="text-sm font-bold mb-2">Application Logs</h3>
      {logs.length === 0 ? (
        <p>No logs yet.</p>
      ) : (
        <ul>
          {logs.map((log, index) => (
            <li key={index} className="mb-1">
              <span className="text-blue-400">[{log.timestamp}]</span>
              <span className={`font-bold ${log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARN' ? 'text-yellow-400' : 'text-green-400'}`}> {log.level}</span>
              <span className="text-purple-300"> [{log.category}]</span>: {log.message}
              {log.data && <pre className="ml-4 text-gray-400">{JSON.stringify(log.data, null, 2)}</pre>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
