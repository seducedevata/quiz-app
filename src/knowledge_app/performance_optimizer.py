"""
WebEngine Performance Patch
Reduces redundant operations and improves UI responsiveness
"""

import time
from functools import wraps

class PerformanceOptimizer:
    def __init__(self):
        self.last_injection_time = 0
        self.injection_count = 0
        self.max_injections = 3
        self.min_injection_interval = 30  # seconds
        
    def should_inject_script(self):
        """Determine if script injection should proceed"""
        current_time = time.time()
        
        # Check if we've exceeded max injections
        if self.injection_count >= self.max_injections:
            return False
            
        # Check if enough time has passed since last injection
        if (current_time - self.last_injection_time) < self.min_injection_interval:
            return False
            
        return True
    
    def record_injection(self):
        """Record that a script injection occurred"""
        self.last_injection_time = time.time()
        self.injection_count += 1
        
    def reset_injection_count(self):
        """Reset injection count (call when page loads)"""
        self.injection_count = 0

def debounce(wait_time):
    """Decorator to debounce function calls"""
    def decorator(func):
        last_called = [0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if (current_time - last_called[0]) >= wait_time:
                last_called[0] = current_time
                return func(*args, **kwargs)
            else:
                print(f"🚫 Debounced call to {func.__name__}")
                
        return wrapper
    return decorator

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Optimized script injection function
@debounce(5.0)  # Minimum 5 seconds between injections
def inject_script_optimized(page, script_content, callback=None):
    """Optimized script injection with rate limiting"""
    if not performance_optimizer.should_inject_script():
        print("🚫 Script injection skipped due to rate limiting")
        return
        
    performance_optimizer.record_injection()
    print(f"✅ Injecting script (attempt {performance_optimizer.injection_count}/{performance_optimizer.max_injections})")
    
    page.runJavaScript(script_content, callback)

# Export for use in webengine_app.py
__all__ = ['PerformanceOptimizer', 'performance_optimizer', 'inject_script_optimized', 'debounce']
