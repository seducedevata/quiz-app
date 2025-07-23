#!/usr/bin/env python3
"""
Ultra-Clean Test Runner
Runs tests with absolutely no warnings or verbose output
"""
import sys
import subprocess
import warnings
import os
from pathlib import Path

# Completely suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def run_silent_tests():
    """Run tests with minimal output"""
    test_dir = Path(__file__).parent
    
    print("🧪 Running Test Suite...")
    
    # Run pytest with maximum silence
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "--tb=no",  # No traceback
        "--disable-warnings",  # No warnings
        "-q",  # Quiet mode
        "--no-header",  # No header
        "--no-summary",  # No summary
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count passed tests from output
            lines = result.stdout.strip().split('\n')
            if lines and lines[-1]:
                print(f"✅ {lines[-1]}")
            else:
                print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
            # Only show failures, not warnings
            if result.stdout:
                print(result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_silent_tests()
    sys.exit(0 if success else 1)
