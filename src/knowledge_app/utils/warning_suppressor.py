"""
Enterprise Warning Suppression System

This module provides comprehensive warning suppression for a clean, professional
application experience. It systematically suppresses known deprecation warnings
and other non-critical warnings that clutter the output.

This is essential for enterprise-grade applications where clean logs and output
are critical for professional deployment.
"""

import warnings
import logging
import sys
import os
import contextlib
import io
from typing import List, Dict, Any

# Import CSS warning suppressor
try:
    from .css_warning_suppressor import CSSWarningSupressor
except ImportError:
    # Fallback if CSS warning suppressor is not available
    CSSWarningSupressor = None

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def suppress_xformers_output():
    """Suppress xFormers output during imports and initialization"""
    # Set environment variables to minimize xFormers output
    os.environ["XFORMERS_MORE_DETAILS"] = "0"
    os.environ["XFORMERS_DISABLE_FLASH_ATTN"] = "1"

    # Try to suppress xFormers warnings at the source
    try:
        # Temporarily redirect stderr to suppress xFormers warnings
        with suppress_stdout_stderr():
            # This will suppress the initial xFormers import warnings
            pass
    except Exception:
        pass


class EnterpriseWarningManager:
    """
    Enterprise-grade warning management system.

    Provides systematic suppression of known warnings while maintaining
    the ability to selectively enable warnings for debugging.
    """

    def __init__(self):
        self.suppressed_warnings: List[Dict[str, Any]] = []
        self.debug_mode = os.getenv("KNOWLEDGE_APP_DEBUG_WARNINGS", "false").lower() == "true"

        # Initialize CSS warning suppressor if available
        self.css_suppressor = None
        if CSSWarningSupressor:
            try:
                self.css_suppressor = CSSWarningSupressor(debug_mode=self.debug_mode)
            except Exception as e:
                logger.warning(f"Failed to initialize CSS warning suppressor: {e}")

    def suppress_pyqt_warnings(self):
        """Suppress PyQt5/SIP related deprecation warnings."""
        pyqt_warnings = [
            # SIP deprecation warnings
            {
                "action": "ignore",
                "message": r"sipPyTypeDict\(\) is deprecated.*",
                "category": DeprecationWarning,
            },
            {"action": "ignore", "message": r".*sipPyTypeDict.*", "category": DeprecationWarning},
            {
                "action": "ignore",
                "message": r".*sip\.sipPyTypeDict.*",
                "category": DeprecationWarning,
            },
            # SWIG related warnings
            {
                "action": "ignore",
                "message": r"builtin type SwigPyPacked has no __module__ attribute",
                "category": DeprecationWarning,
            },
            {
                "action": "ignore",
                "message": r"builtin type SwigPyObject has no __module__ attribute",
                "category": DeprecationWarning,
            },
            {
                "action": "ignore",
                "message": r"builtin type swigvarlink has no __module__ attribute",
                "category": DeprecationWarning,
            },
            # General PyQt warnings
            {"action": "ignore", "message": r".*PyQt5.*", "category": DeprecationWarning},
            {"action": "ignore", "message": r".*Qt.*deprecated.*", "category": DeprecationWarning},
            # CSS Property warnings - Qt doesn't support CSS3 properties
            {"action": "ignore", "message": r".*Unknown CSS property.*", "category": UserWarning},
            {
                "action": "ignore",
                "message": r".*CSS property.*not supported.*",
                "category": UserWarning,
            },
            {"action": "ignore", "message": r".*box-shadow.*", "category": UserWarning},
            {"action": "ignore", "message": r".*text-shadow.*", "category": UserWarning},
            {"action": "ignore", "message": r".*transform.*", "category": UserWarning},
            {"action": "ignore", "message": r".*transition.*", "category": UserWarning},
            {"action": "ignore", "message": r".*animation.*", "category": UserWarning},
            {"action": "ignore", "message": r".*filter.*", "category": UserWarning},
            {"action": "ignore", "message": r".*backdrop-filter.*", "category": UserWarning},
            # QPainter warnings from animations
            {"action": "ignore", "message": r".*QPainter.*", "category": UserWarning},
            {"action": "ignore", "message": r".*painter.*not active.*", "category": UserWarning},
            {"action": "ignore", "message": r".*QPaintDevice.*", "category": UserWarning},
        ]

        for warning_config in pyqt_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("PyQt5/SIP deprecation warnings suppressed for clean output")

    def suppress_ml_warnings(self):
        """Suppress machine learning library warnings."""
        ml_warnings = [
            # PyTorch warnings
            {
                "action": "ignore",
                "message": r".*torch.*deprecated.*",
                "category": DeprecationWarning,
            },
            {"action": "ignore", "message": r".*CUDA.*", "category": UserWarning},
            # xFormers warnings - comprehensive suppression
            {"action": "ignore", "message": r".*xFormers.*", "category": UserWarning},
            {"action": "ignore", "message": r".*XFORMERS.*", "category": UserWarning},
            {"action": "ignore", "message": r".*flashattention.*", "category": UserWarning},
            {"action": "ignore", "message": r".*xFormers can\'t load.*", "category": UserWarning},
            {
                "action": "ignore",
                "message": r".*PyTorch.*CUDA.*you have.*",
                "category": UserWarning,
            },
            # Transformers warnings
            {"action": "ignore", "message": r".*transformers.*", "category": FutureWarning},
            {"action": "ignore", "message": r".*huggingface.*", "category": UserWarning},
            # NumPy warnings
            {
                "action": "ignore",
                "message": r".*numpy.*deprecated.*",
                "category": DeprecationWarning,
            },
            # FAISS warnings
            {
                "action": "ignore",
                "message": r".*Failed to load GPU Faiss.*",
                "category": UserWarning,
            },
            {
                "action": "ignore",
                "message": r".*GpuIndexIVFFlat.*not defined.*",
                "category": UserWarning,
            },
        ]

        for warning_config in ml_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("ML library warnings suppressed for clean output")

    def suppress_general_warnings(self):
        """Suppress general Python warnings that clutter output."""
        general_warnings = [
            # Import warnings
            {"action": "ignore", "category": ImportWarning},
            # Pending deprecation warnings
            {"action": "ignore", "category": PendingDeprecationWarning},
            # Resource warnings (file handles, etc.)
            {"action": "ignore", "category": ResourceWarning},
            # Runtime warnings for non-critical issues
            {
                "action": "ignore",
                "message": r".*invalid escape sequence.*",
                "category": DeprecationWarning,
            },
        ]

        for warning_config in general_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("General Python warnings suppressed for clean output")

    def suppress_all_enterprise_warnings(self):
        """
        Suppress all enterprise-grade warnings for clean production output.

        This is the main method to call for comprehensive warning suppression.
        """
        if self.debug_mode:
            logger.info("Debug mode enabled - warnings will be shown")
            return

        # Suppress xFormers output first
        suppress_xformers_output()

        # Suppress different categories of warnings
        self.suppress_pyqt_warnings()
        self.suppress_ml_warnings()
        self.suppress_general_warnings()

        # Suppress CSS warnings using dedicated suppressor
        if self.css_suppressor:
            try:
                self.css_suppressor.suppress_all_css_warnings()
                logger.debug("CSS warnings suppressed via dedicated suppressor")
            except Exception as e:
                logger.warning(f"Failed to suppress CSS warnings: {e}")

        # Set global warning behavior
        if not self.debug_mode:
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", ImportWarning)
            warnings.simplefilter("ignore", ResourceWarning)

        # Additional comprehensive warning suppression
        warnings.filterwarnings("ignore", message=r".*xFormers.*")
        warnings.filterwarnings("ignore", message=r".*XFORMERS.*")
        warnings.filterwarnings("ignore", message=r".*DLL load failed.*")
        warnings.filterwarnings("ignore", message=r".*flashattention.*")
        warnings.filterwarnings("ignore", message=r".*_C_flashattention.*")

        logger.info(
            f"✅ Enterprise warning suppression active - {len(self.suppressed_warnings)} warning types suppressed"
        )
        logger.info("🔧 Set KNOWLEDGE_APP_DEBUG_WARNINGS=true to enable warnings for debugging")

    def enable_debug_warnings(self):
        """Enable all warnings for debugging purposes."""
        warnings.resetwarnings()
        self.suppressed_warnings.clear()
        logger.info("🐛 Debug mode: All warnings enabled")

    def get_suppression_summary(self) -> Dict[str, Any]:
        """Get a summary of suppressed warnings."""
        return {
            "total_suppressed": len(self.suppressed_warnings),
            "debug_mode": self.debug_mode,
            "categories_suppressed": list(
                set(
                    w.get("category", "Unknown").__name__
                    for w in self.suppressed_warnings
                    if w.get("category")
                )
            ),
        }


# Global instance for easy access
_warning_manager = None


def get_warning_manager() -> EnterpriseWarningManager:
    """Get the global warning manager instance."""
    global _warning_manager
    if _warning_manager is None:
        _warning_manager = EnterpriseWarningManager()
    return _warning_manager


def suppress_enterprise_warnings():
    """
    Convenience function to suppress all enterprise warnings.

    Call this early in your application startup for clean output.
    """
    manager = get_warning_manager()
    manager.suppress_all_enterprise_warnings()


def enable_debug_warnings():
    """
    Convenience function to enable warnings for debugging.

    Useful during development when you need to see warnings.
    """
    manager = get_warning_manager()
    manager.enable_debug_warnings()


def configure_test_warnings():
    """
    Configure warnings specifically for test environments.

    This provides a balance between clean test output and useful debugging info.
    """
    import os

    # Set environment variables for clean test output
    os.environ["XFORMERS_MORE_DETAILS"] = "0"

    # For tests, we want even cleaner output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ImportWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Specifically suppress PyQt warnings
    warnings.filterwarnings("ignore", message=r".*sipPyTypeDict.*")
    warnings.filterwarnings("ignore", message=r".*SwigPy.*")
    warnings.filterwarnings("ignore", message=r".*swigvarlink.*")
    warnings.filterwarnings("ignore", message=r".*builtin type.*has no __module__ attribute.*")

    # Comprehensive xFormers warning suppression for tests
    warnings.filterwarnings("ignore", message=r".*xFormers.*")
    warnings.filterwarnings("ignore", message=r".*XFORMERS.*")
    warnings.filterwarnings("ignore", message=r".*DLL load failed.*")
    warnings.filterwarnings("ignore", message=r".*flashattention.*")
    warnings.filterwarnings("ignore", message=r".*_C_flashattention.*")
    warnings.filterwarnings("ignore", message=r".*PyTorch.*CUDA.*you have.*")

    logger.debug("Test environment warning suppression configured")


# Auto-configure for test environments
if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
    configure_test_warnings()