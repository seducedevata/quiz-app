"""
Dependency Health Checker

This module provides comprehensive dependency checking and health monitoring
for the Knowledge App, replacing deprecated packages and ensuring compatibility.

Features:
- Modern dependency checking without deprecated packages
- Health monitoring for critical dependencies
- Fallback mechanisms for missing dependencies
- Performance impact assessment
"""

import logging
import importlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pkg_resources

logger = logging.getLogger(__name__)


class DependencyHealthChecker:
    """
    Modern dependency health checker that replaces deprecated packages
    and provides comprehensive dependency monitoring.
    """

    def __init__(self):
        self.dependency_status = {}

        # 🚀 BUG FIX 30: Define exact version requirements for stability
        self.critical_dependencies_with_versions = {
            "PyQt5": "5.15.10",
            "numpy": "1.26.4",
            "pandas": "2.2.2",
            "torch": "2.1.0",
            "transformers": "4.41.2"
        }

        self.optional_dependencies_with_versions = {
            "haystack": "1.26.4",
            "faiss-cpu": "1.8.0",
            "sentence-transformers": "2.7.0",
            "peft": "0.10.0",
            "bitsandbytes": "0.43.1",
            "accelerate": "0.29.3",
            "aiohttp": "3.9.5",
            "requests": "2.31.0",
            "pydantic": "2.7.4"
        }

        # Legacy lists for backward compatibility
        self.critical_dependencies = list(self.critical_dependencies_with_versions.keys())
        self.optional_dependencies = list(self.optional_dependencies_with_versions.keys())

        self.deprecated_packages = [
            "quantulum3",  # Replace with modern alternatives
            "pkg_resources",  # Use importlib.metadata instead
        ]

    def check_all_dependencies(self) -> Dict[str, Any]:
        """
        Check all dependencies and return comprehensive health report.

        Returns:
            Dictionary with dependency status and recommendations
        """
        report = {
            "critical": {},
            "optional": {},
            "deprecated": {},
            "recommendations": [],
            "overall_health": "unknown",
        }

        # Check critical dependencies
        critical_issues = 0
        for dep in self.critical_dependencies:
            status = self._check_dependency(dep)
            report["critical"][dep] = status
            if not status["available"]:
                critical_issues += 1

        # Check optional dependencies
        for dep in self.optional_dependencies:
            status = self._check_dependency(dep)
            report["optional"][dep] = status

        # Check for deprecated packages
        for dep in self.deprecated_packages:
            status = self._check_deprecated_package(dep)
            report["deprecated"][dep] = status
            if status["found"]:
                report["recommendations"].append(
                    f"Replace deprecated package '{dep}' with modern alternative"
                )

        # Determine overall health
        if critical_issues == 0:
            report["overall_health"] = "good"
        elif critical_issues <= 2:
            report["overall_health"] = "warning"
        else:
            report["overall_health"] = "critical"

        return report

    def _check_dependency(self, package_name: str) -> Dict[str, Any]:
        """Check a single dependency"""
        try:
            # Use importlib.metadata instead of deprecated pkg_resources
            try:
                import importlib.metadata as metadata

                version = metadata.version(package_name)
                available = True
                method = "importlib.metadata"
            except ImportError:
                # Fallback to pkg_resources if importlib.metadata not available
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    version = pkg_resources.get_distribution(package_name).version
                available = True
                method = "pkg_resources"
            except Exception:
                version = None
                available = False
                method = None

            # Try to import the module
            try:
                module = importlib.import_module(package_name)
                importable = True
                module_path = getattr(module, "__file__", "unknown")
            except ImportError:
                importable = False
                module_path = None

            return {
                "available": available,
                "importable": importable,
                "version": version,
                "module_path": module_path,
                "check_method": method,
            }

        except Exception as e:
            logger.debug(f"Error checking dependency {package_name}: {e}")
            return {
                "available": False,
                "importable": False,
                "version": None,
                "module_path": None,
                "error": str(e),
            }

    def _check_deprecated_package(self, package_name: str) -> Dict[str, Any]:
        """Check for deprecated packages"""
        try:
            # Check if package is installed
            try:
                import importlib.metadata as metadata

                version = metadata.version(package_name)
                found = True
            except ImportError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    version = pkg_resources.get_distribution(package_name).version
                found = True
            except Exception:
                found = False
                version = None

            # Check if it's being used in the current process
            in_use = package_name in sys.modules

            return {
                "found": found,
                "version": version,
                "in_use": in_use,
                "recommendation": self._get_replacement_recommendation(package_name),
            }

        except Exception as e:
            return {"found": False, "version": None, "in_use": False, "error": str(e)}

    def _get_replacement_recommendation(self, package_name: str) -> str:
        """Get replacement recommendations for deprecated packages"""
        replacements = {
            "quantulum3": "Use built-in text processing or modern NLP libraries like spaCy",
            "pkg_resources": "Use importlib.metadata for package metadata access",
        }
        return replacements.get(package_name, "Consider finding a modern alternative")

    def get_health_summary(self) -> str:
        """Get a human-readable health summary"""
        report = self.check_all_dependencies()

        summary_lines = []
        summary_lines.append(f"🏥 Dependency Health: {report['overall_health'].upper()}")

        # Critical dependencies
        critical_ok = sum(1 for status in report["critical"].values() if status["available"])
        critical_total = len(report["critical"])
        summary_lines.append(f"🔴 Critical: {critical_ok}/{critical_total} available")

        # Optional dependencies
        optional_ok = sum(1 for status in report["optional"].values() if status["available"])
        optional_total = len(report["optional"])
        summary_lines.append(f"🟡 Optional: {optional_ok}/{optional_total} available")

        # Deprecated packages
        deprecated_found = sum(1 for status in report["deprecated"].values() if status["found"])
        if deprecated_found > 0:
            summary_lines.append(f"⚠️ Deprecated packages found: {deprecated_found}")

        # Recommendations
        if report["recommendations"]:
            summary_lines.append("📋 Recommendations:")
            for rec in report["recommendations"][:3]:  # Show top 3
                summary_lines.append(f"  - {rec}")

        return "\n".join(summary_lines)

    def fix_deprecated_warnings(self):
        """Apply fixes for deprecated package warnings"""
        try:
            # Suppress pkg_resources deprecation warnings
            warnings.filterwarnings(
                "ignore", message=".*pkg_resources.*deprecated.*", category=DeprecationWarning
            )

            # Suppress quantulum3 warnings if present
            warnings.filterwarnings(
                "ignore", message=".*quantulum3.*deprecated.*", category=UserWarning
            )

            logger.debug("✅ Deprecated package warnings suppressed")

        except Exception as e:
            logger.warning(f"Failed to suppress deprecated warnings: {e}")

    def check_exact_versions(self) -> Dict[str, Any]:
        """
        🚀 BUG FIX 30: Check that installed versions exactly match pinned requirements

        This method validates that all critical dependencies are installed with
        the exact versions specified in requirements-pinned.txt to prevent
        compatibility issues and "works on my machine" syndrome.

        Returns:
            Dictionary with version check results
        """
        results = {
            "all_exact": True,
            "critical_mismatches": [],
            "optional_mismatches": [],
            "missing_packages": [],
            "version_report": {}
        }

        # Check critical dependencies
        for package, expected_version in self.critical_dependencies_with_versions.items():
            try:
                # Get installed version
                try:
                    import importlib.metadata as metadata
                    installed_version = metadata.version(package)
                except ImportError:
                    # Fallback to pkg_resources
                    import pkg_resources
                    installed_version = pkg_resources.get_distribution(package).version

                # Compare versions
                if installed_version != expected_version:
                    results["all_exact"] = False
                    results["critical_mismatches"].append({
                        "package": package,
                        "expected": expected_version,
                        "installed": installed_version,
                        "severity": "CRITICAL"
                    })
                    logger.warning(f"🚨 CRITICAL version mismatch: {package} expected {expected_version}, got {installed_version}")
                else:
                    logger.debug(f"✅ {package} version {installed_version} matches exactly")

                results["version_report"][package] = {
                    "expected": expected_version,
                    "installed": installed_version,
                    "matches": installed_version == expected_version,
                    "type": "critical"
                }

            except Exception as e:
                results["all_exact"] = False
                results["missing_packages"].append({
                    "package": package,
                    "expected": expected_version,
                    "error": str(e),
                    "severity": "CRITICAL"
                })
                logger.error(f"❌ CRITICAL package missing: {package} (expected {expected_version})")

        # Check optional dependencies
        for package, expected_version in self.optional_dependencies_with_versions.items():
            try:
                # Get installed version
                try:
                    import importlib.metadata as metadata
                    installed_version = metadata.version(package)
                except ImportError:
                    import pkg_resources
                    installed_version = pkg_resources.get_distribution(package).version

                # Compare versions
                if installed_version != expected_version:
                    results["optional_mismatches"].append({
                        "package": package,
                        "expected": expected_version,
                        "installed": installed_version,
                        "severity": "WARNING"
                    })
                    logger.warning(f"⚠️ Optional version mismatch: {package} expected {expected_version}, got {installed_version}")
                else:
                    logger.debug(f"✅ {package} version {installed_version} matches exactly")

                results["version_report"][package] = {
                    "expected": expected_version,
                    "installed": installed_version,
                    "matches": installed_version == expected_version,
                    "type": "optional"
                }

            except Exception as e:
                # Optional dependencies missing is not critical
                logger.debug(f"Optional package not installed: {package} (expected {expected_version})")
                results["version_report"][package] = {
                    "expected": expected_version,
                    "installed": None,
                    "matches": False,
                    "type": "optional",
                    "error": str(e)
                }

        return results

    def get_version_mismatch_report(self) -> str:
        """
        🚀 BUG FIX 30: Generate a user-friendly report of version mismatches

        Returns:
            Formatted string with version mismatch information and fix instructions
        """
        results = self.check_exact_versions()

        if results["all_exact"] and not results["optional_mismatches"]:
            return "✅ All dependencies match pinned versions exactly"

        report = []
        report.append("🚨 DEPENDENCY VERSION MISMATCHES DETECTED")
        report.append("=" * 50)

        # Critical mismatches
        if results["critical_mismatches"]:
            report.append("\n❌ CRITICAL VERSION MISMATCHES:")
            for mismatch in results["critical_mismatches"]:
                report.append(f"  • {mismatch['package']}: expected {mismatch['expected']}, got {mismatch['installed']}")

        # Missing packages
        if results["missing_packages"]:
            report.append("\n❌ MISSING CRITICAL PACKAGES:")
            for missing in results["missing_packages"]:
                report.append(f"  • {missing['package']}: expected {missing['expected']}")

        # Optional mismatches
        if results["optional_mismatches"]:
            report.append("\n⚠️ OPTIONAL VERSION MISMATCHES:")
            for mismatch in results["optional_mismatches"]:
                report.append(f"  • {mismatch['package']}: expected {mismatch['expected']}, got {mismatch['installed']}")

        # Fix instructions
        report.append("\n🔧 TO FIX THESE ISSUES:")
        report.append("1. Install exact versions using pinned requirements:")
        report.append("   pip install -r requirements-pinned.txt")
        report.append("2. Or upgrade specific packages:")

        all_mismatches = results["critical_mismatches"] + results["optional_mismatches"]
        for mismatch in all_mismatches:
            report.append(f"   pip install {mismatch['package']}=={mismatch['expected']}")

        report.append("\n⚠️ WARNING: Version mismatches can cause:")
        report.append("  • Cryptic CUDA errors")
        report.append("  • AttributeErrors for missing functions")
        report.append("  • TypeErrors for changed function signatures")
        report.append("  • Incompatible model loading")
        report.append("  • Training failures")

        return "\n".join(report)
    
    def enforce_critical_versions(self) -> Tuple[bool, str]:
        """
        CRITICAL FIX: Enforce critical version requirements and prevent startup with mismatches
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        results = self.check_exact_versions()
        
        # Check for critical issues that should prevent startup
        critical_issues = []
        
        # Critical version mismatches
        if results["critical_mismatches"]:
            for mismatch in results["critical_mismatches"]:
                critical_issues.append(
                    f"CRITICAL: {mismatch['package']} version {mismatch['installed']} "
                    f"incompatible (requires {mismatch['expected']})"
                )
        
        # Missing critical packages
        if results["missing_packages"]:
            for missing in results["missing_packages"]:
                critical_issues.append(
                    f"CRITICAL: Required package {missing['package']} {missing['expected']} is missing"
                )
        
        # If we have critical issues, prevent startup
        if critical_issues:
            error_message = (
                "APPLICATION STARTUP BLOCKED - CRITICAL DEPENDENCY ISSUES:\n\n" +
                "\n".join(f"• {issue}" for issue in critical_issues) +
                "\n\nFIX REQUIRED:\n" +
                "Run: pip install -r requirements-pinned.txt\n\n" +
                "These version mismatches can cause:\n" +
                "• CUDA initialization failures\n" +
                "• Model loading crashes\n" +
                "• Cryptic runtime errors\n" +
                "• Data corruption\n\n" +
                "The application will not start until these issues are resolved."
            )
            
            logger.critical("🚨 BLOCKING APPLICATION STARTUP DUE TO CRITICAL VERSION MISMATCHES")
            logger.critical(error_message)
            
            return False, error_message
        
        # Optional mismatches are warnings only
        if results["optional_mismatches"]:
            warning_count = len(results["optional_mismatches"])
            warning_message = (
                f"WARNING: {warning_count} optional dependencies have version mismatches. "
                "Application will start but some features may not work correctly. "
                "Run 'pip install -r requirements-pinned.txt' to fix."
            )
            logger.warning(f"⚠️ {warning_message}")
            return True, warning_message
        
        # All good
        logger.info("✅ All critical dependencies have correct versions")
        return True, "All critical dependencies verified"


# CRITICAL FIX: Thread-safe global health checker instance
import threading
_global_health_checker: Optional[DependencyHealthChecker] = None
_health_checker_lock = threading.RLock()  # Use RLock to allow recursive calls


def get_health_checker() -> DependencyHealthChecker:
    """Get the global dependency health checker - THREAD SAFE"""
    global _global_health_checker
    
    with _health_checker_lock:
        if _global_health_checker is None:
            _global_health_checker = DependencyHealthChecker()
        return _global_health_checker


def check_dependencies() -> bool:
    """
    Check all dependencies and return True if system is healthy.

    Returns:
        True if all critical dependencies are available
    """
    checker = get_health_checker()
    report = checker.check_all_dependencies()

    # Check if all critical dependencies are available
    critical_ok = all(status["available"] for status in report["critical"].values())

    if not critical_ok:
        logger.error("❌ Critical dependencies missing")
        logger.info(checker.get_health_summary())
        return False

    logger.info("✅ All critical dependencies available")
    return True


def fix_deprecated_warnings():
    """Fix deprecated package warnings"""
    checker = get_health_checker()
    checker.fix_deprecated_warnings()


def log_dependency_health():
    """Log dependency health summary"""
    checker = get_health_checker()
    summary = checker.get_health_summary()
    logger.info(f"Dependency Health Report:\n{summary}")
