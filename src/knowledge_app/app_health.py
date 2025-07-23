"""
app_health.py: Robustness utilities for dependency checking and logging.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
import importlib
import traceback
import os
from pathlib import Path

print(f"DEBUG: Starting app_health.py. sys.argv: {sys.argv}")
print(f"DEBUG: Current working directory: {os.getcwd()}")

# 🔧 CRITICAL FIX: Use unified dependency list from modern checker
# This prevents inconsistencies between legacy and modern dependency checking
def get_unified_dependencies():
    """Get unified dependency list from modern checker"""
    try:
        from knowledge_app.core.dependency_health_checker import DependencyHealthChecker
        checker = DependencyHealthChecker()

        # Combine critical and optional dependencies with version requirements
        unified_deps = []

        # Add critical dependencies (required)
        for dep in checker.critical_dependencies:
            unified_deps.append((dep, None))  # No specific version requirements for now

        # Add some optional dependencies that are commonly needed
        important_optional = ["haystack", "faiss", "sentence_transformers", "peft"]
        for dep in important_optional:
            if dep in checker.optional_dependencies:
                unified_deps.append((dep, None))

        return unified_deps

    except ImportError:
        # Fallback to legacy list if modern checker not available
        return [
            ("PyQt5", None),
            ("transformers", None),
            ("torch", "2.0.0"),
            ("torchvision", None),
            ("pdfplumber", None),
            ("python-dotenv", None),
            ("six", None),
        ]

# 🔧 CRITICAL FIX: Dynamic dependency list that stays in sync
REQUIRED_PACKAGES = get_unified_dependencies()


def setup_logging():
    """Configure logging for the application

    Returns:
        str: Path to the log file
    """
    # Get the workspace root directory
    workspace_root = Path(__file__).parent.parent.parent

    # Create logs directory
    log_dir = workspace_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Set up log file path
    log_file_path = log_dir / "knowledge_app.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = RotatingFileHandler(
        str(log_file_path), maxBytes=2 * 1024 * 1024, backupCount=3  # 2MB
    )
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging system initialized")
    return str(log_file_path)


def check_dependencies():
    """
    🛡️ CRITICAL FIX: Check dependencies with version pinning enforcement

    Returns:
        tuple: (bool, str) indicating success and message
    """
    logger = logging.getLogger(__name__)
    logger.info("Checking dependencies with modern health checker and version enforcement...")

    try:
        # Use the new dependency health checker
        from knowledge_app.core.dependency_health_checker import check_dependencies as modern_check
        from knowledge_app.core.dependency_health_checker import fix_deprecated_warnings
        from knowledge_app.core.dependency_health_checker import get_health_checker

        # Fix deprecated warnings first
        fix_deprecated_warnings()

        # Perform modern dependency check
        success = modern_check()
        if not success:
            return False, "Critical dependencies missing - check logs for details"

        # 🛡️ CRITICAL FIX: Enforce version pinning
        logger.info("✅ Basic dependencies available, checking version pinning...")

        checker = get_health_checker()
        version_results = checker.check_exact_versions()

        if not version_results["all_exact"]:
            # Generate detailed version mismatch report
            mismatch_report = checker.get_version_mismatch_report()
            logger.warning("⚠️ Version pinning violations detected:")
            logger.warning(mismatch_report)

            # Check if there are critical mismatches
            if version_results["critical_mismatches"]:
                critical_count = len(version_results["critical_mismatches"])
                return False, f"CRITICAL: {critical_count} version mismatches detected. Check logs for details."
            else:
                # Only optional mismatches - warn but don't fail
                optional_count = len(version_results["optional_mismatches"])
                logger.warning(f"⚠️ {optional_count} optional version mismatches detected")
                return True, f"Dependencies available with {optional_count} optional version warnings"

        logger.info("✅ All dependencies match pinned versions exactly")
        return True, "All dependencies available with exact version matching"

    except ImportError:
        # Fallback to legacy checking if new checker not available
        logger.warning("Modern dependency checker not available, using legacy method")
        return _legacy_check_dependencies()


def _legacy_check_dependencies():
    """Legacy dependency checking method"""
    logger = logging.getLogger(__name__)

    for pkg, min_version in REQUIRED_PACKAGES:
        try:
            # Handle special case for python-dotenv
            if pkg == "python-dotenv":
                mod = importlib.import_module("dotenv")
            else:
                mod = importlib.import_module(pkg)

            if min_version:
                version = getattr(mod, "__version__", None)
                if version and version < min_version:
                    msg = f"{pkg} version {version} is less than required {min_version}"
                    logger.error(msg)
                    return False, msg

            logger.debug(f"Found {pkg} {getattr(mod, '__version__', 'unknown version')}")

        except ImportError as e:
            msg = f"Missing dependency: {pkg}"
            logger.error(f"{msg}. Error: {e}")
            return False, msg
        except Exception as e:
            msg = f"Error checking {pkg}: {e}"
            logger.error(msg)
            return False, msg

    logger.info("All dependencies are installed")
    return True, "All dependencies are installed."


def startup_health_check():
    """
    🛡️ CRITICAL FIX: Comprehensive startup health check with version enforcement

    This function should be called during application startup to ensure
    all dependencies are properly installed with correct versions.

    Returns:
        tuple: (bool, str, dict) indicating success, message, and detailed results
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting comprehensive startup health check...")

    health_results = {
        "dependencies_available": False,
        "versions_exact": False,
        "critical_mismatches": [],
        "optional_mismatches": [],
        "warnings": [],
        "recommendations": []
    }

    try:
        # Step 1: Check basic dependency availability
        deps_ok, deps_msg = check_dependencies()
        health_results["dependencies_available"] = deps_ok

        if not deps_ok:
            return False, f"Startup health check failed: {deps_msg}", health_results

        # Step 2: Detailed version analysis
        from knowledge_app.core.dependency_health_checker import get_health_checker
        checker = get_health_checker()
        version_results = checker.check_exact_versions()

        health_results["versions_exact"] = version_results["all_exact"]
        health_results["critical_mismatches"] = version_results["critical_mismatches"]
        health_results["optional_mismatches"] = version_results["optional_mismatches"]

        # Step 3: Generate recommendations
        if version_results["critical_mismatches"]:
            health_results["recommendations"].append(
                "Run 'pip install -r requirements-pinned.txt' to fix critical version mismatches"
            )

        if version_results["optional_mismatches"]:
            health_results["recommendations"].append(
                "Consider updating optional dependencies to pinned versions for best compatibility"
            )

        # Step 4: Final assessment
        if version_results["critical_mismatches"]:
            critical_count = len(version_results["critical_mismatches"])
            return False, f"Startup health check failed: {critical_count} critical version mismatches", health_results

        # Success with possible warnings
        warning_count = len(version_results["optional_mismatches"])
        if warning_count > 0:
            health_results["warnings"].append(f"{warning_count} optional version mismatches detected")
            return True, f"Startup health check passed with {warning_count} warnings", health_results

        logger.info("✅ Startup health check passed completely")
        return True, "Startup health check passed - all dependencies exact", health_results

    except Exception as e:
        logger.error(f"❌ Startup health check failed with exception: {e}")
        health_results["warnings"].append(f"Health check exception: {str(e)}")
        return False, f"Startup health check failed: {str(e)}", health_results

def get_log_file_path():
    """Get the path to the log file

    Returns:
        str: Path to the log file
    """
    workspace_root = Path(__file__).parent.parent.parent
    return str(workspace_root / "logs" / "knowledge_app.log")


if __name__ == "__main__":
    log_file = setup_logging()
    print(f"Logging to: {log_file}")

    # 🛡️ CRITICAL FIX: Use comprehensive startup health check
    print("🔍 Running comprehensive startup health check...")
    ok, msg, results = startup_health_check()

    print(f"Health check result: {msg}")

    if results["warnings"]:
        print("⚠️ Warnings:")
        for warning in results["warnings"]:
            print(f"  • {warning}")

    if results["recommendations"]:
        print("💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  • {rec}")

    if not ok:
        print("❌ Startup health check failed!")
        if results["critical_mismatches"]:
            print("\n🚨 Critical version mismatches:")
            for mismatch in results["critical_mismatches"]:
                print(f"  • {mismatch['package']}: expected {mismatch['expected']}, got {mismatch['installed']}")
        exit(1)
    else:
        print("✅ Startup health check passed!")
        if results["versions_exact"]:
            print("🎯 All versions match pinned requirements exactly")
        else:
            print("⚠️ Some optional versions differ from pinned requirements")