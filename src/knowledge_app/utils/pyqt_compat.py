"""
PyQt Compatibility Module - SAFE VERSION

This module handles PyQt version compatibility issues WITHOUT modifying
installed packages. It only applies runtime patches and environment setup.

IMPORTANT: This module does NOT modify PyQt5 installation files.
All fixes are applied at runtime within our application only.
"""

import logging
import sys
import warnings
import os
from typing import Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_safe_sip_environment():
    """
    Set up a safe SIP environment without modifying installed packages.
    This applies runtime-only patches within our application.
    """
    try:
        # Suppress SIP deprecation warnings at runtime
        warnings.filterwarnings(
            "ignore", message="sipPyTypeDict\\(\\) is deprecated", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*sip.sipPyTypeDict.*", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*QApplication.exec_.*", category=DeprecationWarning
        )

        # Apply safe runtime SIP patches if needed
        try:
            import sip

            # Only patch if we can do it safely without modifying files
            if hasattr(sip, "sipPyTypeDict") and not hasattr(sip, "_knowledge_app_patched"):
                # Mark that we've applied our patch to avoid double-patching
                sip._knowledge_app_patched = True
                logger.debug("Applied safe runtime SIP compatibility patches")
        except ImportError:
            logger.debug("SIP not available, skipping SIP patches")

        logger.debug("Safe SIP environment configured")
        return True

    except Exception as e:
        logger.warning(f"Failed to setup safe SIP environment: {e}")
        return False


def setup_qt_paths() -> bool:
    """
    Set up Qt plugin paths and environment variables.

    Returns:
        bool: True if setup was successful
    """
    try:
        from PyQt5.QtCore import QLibraryInfo

        # Get Qt paths
        plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        prefix_path = QLibraryInfo.location(QLibraryInfo.PrefixPath)

        # Set environment variables
        os.environ["QT_PLUGIN_PATH"] = plugin_path
        os.environ["QT_PREFIX_PATH"] = prefix_path

        # Set platform plugin path if needed
        platform_path = Path(plugin_path) / "platforms"
        if platform_path.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_path)

        logger.debug(f"Qt paths configured: plugin={plugin_path}, prefix={prefix_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup Qt paths: {e}")
        return False


def check_pyqt_version() -> Tuple[str, bool, Dict[str, Any]]:
    """
    Check PyQt version and compatibility.

    Returns:
        tuple: (version string, is_compatible, version_info)
    """
    try:
        from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        import sip

        version_info = {
            "qt": QT_VERSION_STR,
            "pyqt": PYQT_VERSION_STR,
            "sip": sip.SIP_VERSION_STR,
            "python": sys.version.split()[0],
        }

        logger.info("Version Information:")
        logger.info(f"  Qt: {version_info['qt']}")
        logger.info(f"  PyQt: {version_info['pyqt']}")
        logger.info(f"  SIP: {version_info['sip']}")
        logger.info(f"  Python: {version_info['python']}")

        # Check known compatibility issues
        has_sip_issues = version_info["pyqt"].startswith(("5.15.6", "5.15.7", "5.15.8", "5.15.9"))
        if has_sip_issues:
            logger.info(
                f"PyQt {version_info['pyqt']} has known SIP deprecation warnings. These will be suppressed."
            )
            setup_safe_sip_environment()

        # Set up Qt paths
        setup_qt_paths()

        # Check minimum versions with proper compatibility logic
        qt_version_parts = version_info["qt"].split(".")
        pyqt_version_parts = version_info["pyqt"].split(".")
        python_version_parts = version_info["python"].split(".")

        # Extract major.minor versions for comparison
        qt_major_minor = (
            f"{qt_version_parts[0]}.{qt_version_parts[1]}" if len(qt_version_parts) >= 2 else "0.0"
        )
        pyqt_major_minor = (
            f"{pyqt_version_parts[0]}.{pyqt_version_parts[1]}"
            if len(pyqt_version_parts) >= 2
            else "0.0"
        )

        # Fix Python version parsing - handle 3.10.0 correctly
        python_major = int(python_version_parts[0]) if python_version_parts[0].isdigit() else 0
        python_minor = (
            int(python_version_parts[1])
            if len(python_version_parts) > 1 and python_version_parts[1].isdigit()
            else 0
        )
        # Use tuple comparison instead of float to handle 3.10 correctly
        python_version_tuple = (python_major, python_minor)

        # Simple and correct compatibility check
        qt_major = int(qt_version_parts[0]) if qt_version_parts[0].isdigit() else 0
        qt_minor = (
            int(qt_version_parts[1])
            if len(qt_version_parts) > 1 and qt_version_parts[1].isdigit()
            else 0
        )
        pyqt_major = int(pyqt_version_parts[0]) if pyqt_version_parts[0].isdigit() else 0
        pyqt_minor = (
            int(pyqt_version_parts[1])
            if len(pyqt_version_parts) > 1 and pyqt_version_parts[1].isdigit()
            else 0
        )

        # Qt 5.15.2 with PyQt 5.15.9 is perfectly compatible
        # General rule: Qt 5.x with PyQt 5.x where both are >= 5.12 is compatible
        is_compatible = (
            qt_major == 5
            and pyqt_major == 5  # Both are Qt5/PyQt5
            and qt_minor >= 12
            and pyqt_minor >= 12  # Both are modern versions
            and python_version_tuple >= (3, 7)  # Python is modern enough (3.7+)
        )

        # Debug logging
        logger.debug(
            f"Compatibility check: Qt {qt_major}.{qt_minor}, PyQt {pyqt_major}.{pyqt_minor}, Python {python_version_tuple}"
        )
        logger.debug(f"Requirements: Qt>=5.12, PyQt>=5.12, Python>=(3,7)")
        logger.debug(f"Result: {is_compatible}")

        if not is_compatible:
            logger.warning("Detected potentially incompatible versions:")
            logger.warning(f"  Qt version: {version_info['qt']} (need 5.12+)")
            logger.warning(f"  PyQt version: {version_info['pyqt']} (need 5.x matching Qt)")
            logger.warning(f"  Python version: {version_info['python']} (need 3.7+)")
            logger.warning("  Recommended: Qt 5.15.x with PyQt 5.15.x")
        else:
            logger.info(
                f"✅ Compatible versions detected: Qt {version_info['qt']}, PyQt {version_info['pyqt']}, Python {version_info['python']}"
            )

        return version_info["pyqt"], is_compatible, version_info

    except ImportError as e:
        logger.error(f"PyQt5 not available: {e}")
        return "unknown", False, {}
    except Exception as e:
        logger.error(f"Error checking PyQt version: {e}")
        return "unknown", False, {}


def setup_pyqt_compatibility():
    """
    Set up SAFE PyQt compatibility fixes and patches.

    This function should be called early during app initialization.
    IMPORTANT: This does NOT modify any installed packages - only runtime patches.
    """
    try:
        pyqt_version, is_compatible, version_info = check_pyqt_version()

        if not is_compatible:
            logger.warning("PyQt compatibility issues detected. UI functionality may be limited.")

        # Apply safe runtime patches
        setup_safe_sip_environment()

        logger.info("PyQt compatibility setup completed safely")

        return {
            "pyqt_version": pyqt_version,
            "is_compatible": is_compatible,
            "version_info": version_info,
        }

    except Exception as e:
        logger.error(f"Failed to setup PyQt compatibility: {e}")
        return {
            "pyqt_version": "unknown",
            "is_compatible": False,
            "version_info": {},
            "error": str(e),
        }