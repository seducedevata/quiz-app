"""
CSS Warning Suppressor - Suppress Qt CSS Property Warnings

This module provides utilities to suppress CSS property warnings that occur
when using CSS3 properties that Qt doesn't support. These warnings are
cosmetic and don't affect functionality.

Features:
- Suppress CSS3 property warnings (box-shadow, text-shadow, transform, etc.)
- Suppress QPainter warnings from animations
- Context manager for temporary suppression
- Global suppression setup
"""

import warnings
import logging
import sys
from contextlib import contextmanager
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# CSS3 properties that Qt doesn't support but are commonly used
UNSUPPORTED_CSS_PROPERTIES = [
    "box-shadow",
    "text-shadow",
    "transform",
    "transition",
    "animation",
    "filter",
    "backdrop-filter",
    "clip-path",
    "mask",
    "perspective",
    "transform-origin",
    "animation-delay",
    "animation-duration",
    "animation-fill-mode",
    "animation-iteration-count",
    "animation-name",
    "animation-play-state",
    "animation-timing-function",
    "transition-delay",
    "transition-duration",
    "transition-property",
    "transition-timing-function",
]


class CSSWarningSupressor:
    """Suppress CSS and QPainter warnings that are harmless but noisy"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.suppressed_warnings: List[Dict[str, Any]] = []
        self.original_warning_filters = []

    def suppress_css_warnings(self) -> None:
        """Suppress CSS property warnings"""
        css_warnings = [
            # General CSS property warnings
            {"action": "ignore", "message": r".*Unknown CSS property.*", "category": UserWarning},
            {
                "action": "ignore",
                "message": r".*CSS property.*not supported.*",
                "category": UserWarning,
            },
            {"action": "ignore", "message": r".*Invalid CSS.*", "category": UserWarning},
            {"action": "ignore", "message": r".*CSS.*warning.*", "category": UserWarning},
        ]

        # Add specific warnings for each unsupported CSS property
        for prop in UNSUPPORTED_CSS_PROPERTIES:
            css_warnings.extend(
                [
                    {"action": "ignore", "message": f".*{prop}.*", "category": UserWarning},
                    {
                        "action": "ignore",
                        "message": f'.*{prop.replace("-", "_")}.*',
                        "category": UserWarning,
                    },
                ]
            )

        # Apply all CSS warning filters
        for warning_config in css_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("CSS property warnings suppressed")

    def suppress_qpainter_warnings(self) -> None:
        """Suppress QPainter warnings from animations"""
        qpainter_warnings = [
            {"action": "ignore", "message": r".*QPainter.*", "category": UserWarning},
            {"action": "ignore", "message": r".*painter.*not active.*", "category": UserWarning},
            {"action": "ignore", "message": r".*QPaintDevice.*", "category": UserWarning},
            {"action": "ignore", "message": r".*QPaintEngine.*", "category": UserWarning},
            {"action": "ignore", "message": r".*painting.*", "category": UserWarning},
        ]

        for warning_config in qpainter_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("QPainter warnings suppressed")

    def suppress_qt_style_warnings(self) -> None:
        """Suppress Qt style-related warnings"""
        qt_style_warnings = [
            {"action": "ignore", "message": r".*QStyleSheet.*", "category": UserWarning},
            {"action": "ignore", "message": r".*style.*sheet.*", "category": UserWarning},
            {"action": "ignore", "message": r".*QStyle.*", "category": UserWarning},
        ]

        for warning_config in qt_style_warnings:
            warnings.filterwarnings(**warning_config)
            self.suppressed_warnings.append(warning_config)

        if not self.debug_mode:
            logger.debug("Qt style warnings suppressed")

    def suppress_all_css_warnings(self) -> None:
        """Suppress all CSS and style-related warnings"""
        self.suppress_css_warnings()
        self.suppress_qpainter_warnings()
        self.suppress_qt_style_warnings()

        if not self.debug_mode:
            logger.info("All CSS and style warnings suppressed for clean output")

    def restore_warnings(self) -> None:
        """Restore original warning behavior"""
        # Reset warning filters
        warnings.resetwarnings()
        self.suppressed_warnings.clear()

        if not self.debug_mode:
            logger.debug("Warning filters restored")


@contextmanager
def suppress_css_warnings_context(debug_mode: bool = False):
    """Context manager to temporarily suppress CSS warnings"""
    suppressor = CSSWarningSupressor(debug_mode=debug_mode)
    try:
        suppressor.suppress_all_css_warnings()
        yield suppressor
    finally:
        suppressor.restore_warnings()


def setup_global_css_warning_suppression(debug_mode: bool = False) -> CSSWarningSupressor:
    """
    Set up global CSS warning suppression for the entire application.

    Args:
        debug_mode: If True, warnings will be logged for debugging

    Returns:
        CSSWarningSupressor instance for manual control if needed
    """
    suppressor = CSSWarningSupressor(debug_mode=debug_mode)
    suppressor.suppress_all_css_warnings()

    logger.info("✅ Global CSS warning suppression enabled")
    return suppressor


def is_css_warning(message: str) -> bool:
    """
    Check if a warning message is related to CSS properties.

    Args:
        message: Warning message to check

    Returns:
        True if the message is a CSS-related warning
    """
    css_keywords = [
        "css",
        "stylesheet",
        "property",
        "box-shadow",
        "text-shadow",
        "transform",
        "transition",
        "animation",
        "filter",
        "qpainter",
        "qstyle",
        "painting",
    ]

    message_lower = message.lower()
    return any(keyword in message_lower for keyword in css_keywords)


# Global suppressor instance
_global_suppressor: CSSWarningSupressor = None


def get_global_suppressor() -> CSSWarningSupressor:
    """Get the global CSS warning suppressor instance"""
    global _global_suppressor
    if _global_suppressor is None:
        _global_suppressor = setup_global_css_warning_suppression()
    return _global_suppressor


# Auto-setup when module is imported (can be disabled by setting environment variable)
import os

if os.getenv("DISABLE_AUTO_CSS_WARNING_SUPPRESSION") != "1":
    try:
        setup_global_css_warning_suppression()
    except Exception as e:
        logger.warning(f"Failed to auto-setup CSS warning suppression: {e}")