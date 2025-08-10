// Session tracking matching Qt implementation
import { AppLogger } from './logger';

interface UserAction {
  session_id: string;
  action_count: number;
  timestamp: number;
  action_name: string;
  current_screen: string;
  details: any;
}

interface SessionSummary {
  session_id: string;
  duration: number;
  action_count: number;
  current_screen: string;
  start_time: number;
}

class SessionTracker {
  public session_id: string;
  public start_time: number;
  public action_count: number;
  public current_screen: string;
  public user_actions: UserAction[];

  constructor() {
    this.session_id = `session_${Date.now()}`;
    this.start_time = Date.now();
    this.action_count = 0;
    this.current_screen = 'home';
    this.user_actions = [];

    // Load persisted session data if available
    this.loadSessionFromStorage();
    
    AppLogger.info('SESSION', 'SessionTracker initialized', {
      session_id: this.session_id,
      start_time: new Date(this.start_time).toISOString()
    });
  }

  logAction(action_type: string, details: any = null): void {
    this.action_count++;
    const action_data: UserAction = {
      session_id: this.session_id,
      action_count: this.action_count,
      timestamp: Date.now(),
      action_name: action_type,
      current_screen: this.current_screen,
      details: details || {}
    };
    
    this.user_actions.push(action_data);
    
    // Persist to localStorage
    this.saveSessionToStorage();
    
    AppLogger.user('SESSION_ACTION', `Action #${this.action_count}: ${action_type}`, action_data);
  }

  setScreen(screen_name: string): void {
    const old_screen = this.current_screen;
    this.current_screen = screen_name;
    
    // Update AppLogger's current screen (matching Qt implementation)
    AppLogger.setCurrentScreen(screen_name);
    
    // Log navigation using AppLogger's enhanced navigation method
    AppLogger.trackNavigation(old_screen, screen_name);
    
    // Log as user action
    this.logAction('SCREEN_NAVIGATION', {
      from: old_screen,
      to: screen_name,
      timestamp: Date.now()
    });
    
    // Persist to localStorage
    this.saveSessionToStorage();
  }

  getSessionSummary(): SessionSummary {
    const duration = Date.now() - this.start_time;
    return {
      session_id: this.session_id,
      duration: duration,
      action_count: this.action_count,
      current_screen: this.current_screen,
      start_time: this.start_time
    };
  }

  // Persist session data to localStorage
  private saveSessionToStorage(): void {
    try {
      const sessionData = {
        session_id: this.session_id,
        start_time: this.start_time,
        action_count: this.action_count,
        current_screen: this.current_screen,
        user_actions: this.user_actions.slice(-50) // Keep last 50 actions
      };
      
      localStorage.setItem('sessionTracker', JSON.stringify(sessionData));
    } catch (e) {
      console.warn('Failed to save session to localStorage:', e);
    }
  }

  // Load session data from localStorage
  private loadSessionFromStorage(): void {
    try {
      const savedSession = localStorage.getItem('sessionTracker');
      if (savedSession) {
        const sessionData = JSON.parse(savedSession);
        
        // Only restore if session is recent (within 1 hour)
        const sessionAge = Date.now() - sessionData.start_time;
        if (sessionAge < 3600000) { // 1 hour in milliseconds
          this.session_id = sessionData.session_id;
          this.start_time = sessionData.start_time;
          this.action_count = sessionData.action_count || 0;
          this.current_screen = sessionData.current_screen || 'home';
          this.user_actions = sessionData.user_actions || [];
          
          AppLogger.info('SESSION', 'Session restored from localStorage', {
            session_id: this.session_id,
            age_minutes: Math.round(sessionAge / 60000)
          });
        } else {
          // Clear old session data
          localStorage.removeItem('sessionTracker');
          AppLogger.info('SESSION', 'Old session data cleared');
        }
      }
    } catch (e) {
      console.warn('Failed to load session from localStorage:', e);
      localStorage.removeItem('sessionTracker');
    }
  }

  // Get recent user actions
  getRecentActions(limit: number = 10): UserAction[] {
    return this.user_actions.slice(-limit);
  }

  // Clear session data
  clearSession(): void {
    this.user_actions = [];
    this.action_count = 0;
    localStorage.removeItem('sessionTracker');
    AppLogger.info('SESSION', 'Session data cleared');
  }

  // Get session analytics
  getSessionAnalytics(): any {
    const summary = this.getSessionSummary();
    const duration_minutes = summary.duration / 60000;
    
    // Count actions by type
    const actionTypes: { [key: string]: number } = {};
    this.user_actions.forEach(action => {
      actionTypes[action.action_name] = (actionTypes[action.action_name] || 0) + 1;
    });

    // Count screen visits
    const screenVisits: { [key: string]: number } = {};
    this.user_actions.forEach(action => {
      if (action.action_name === 'SCREEN_NAVIGATION') {
        const screen = action.details?.to || action.current_screen;
        screenVisits[screen] = (screenVisits[screen] || 0) + 1;
      }
    });

    return {
      session_summary: summary,
      duration_minutes: Math.round(duration_minutes * 100) / 100,
      actions_per_minute: duration_minutes > 0 ? Math.round((summary.action_count / duration_minutes) * 100) / 100 : 0,
      action_types: actionTypes,
      screen_visits: screenVisits,
      total_actions: this.user_actions.length
    };
  }
}

// Create global session tracker instance
const sessionTracker = new SessionTracker();

// Make it globally available for debugging
if (typeof window !== 'undefined') {
  (window as any).sessionTracker = sessionTracker;
}

export { SessionTracker, sessionTracker };