"""
CSS Cleaner - Remove Unsupported CSS Properties

This utility helps clean up CSS stylesheets by removing CSS3 properties
that Qt doesn't support, preventing warnings while maintaining visual
appearance where possible.

Features:
- Remove unsupported CSS3 properties
- Add comments explaining removals
- Preserve supported properties
- Batch processing of files
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# CSS3 properties that Qt doesn't support
UNSUPPORTED_PROPERTIES = [
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


class CSSCleaner:
    """Clean CSS stylesheets by removing unsupported properties"""

    def __init__(self, add_comments: bool = True):
        self.add_comments = add_comments
        self.removed_properties: List[Tuple[str, str]] = []

    def clean_css_content(self, css_content: str) -> str:
        """
        Clean CSS content by removing unsupported properties.

        Args:
            css_content: CSS content to clean

        Returns:
            Cleaned CSS content
        """
        cleaned_content = css_content
        self.removed_properties.clear()

        for prop in UNSUPPORTED_PROPERTIES:
            # Pattern to match the property and its value
            pattern = rf"\s*{re.escape(prop)}\s*:\s*[^;]+;"

            # Find all matches
            matches = re.findall(pattern, cleaned_content, re.IGNORECASE)

            if matches:
                for match in matches:
                    self.removed_properties.append((prop, match.strip()))

                    # Replace with comment if requested
                    if self.add_comments:
                        comment = f"/* {match.strip()} removed - not supported in Qt */"
                        cleaned_content = re.sub(
                            pattern,
                            f"\n                {comment}",
                            cleaned_content,
                            flags=re.IGNORECASE,
                        )
                    else:
                        cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE)

        return cleaned_content

    def clean_css_file(self, file_path: Path, backup: bool = True) -> bool:
        """
        Clean a CSS file by removing unsupported properties.

        Args:
            file_path: Path to CSS file
            backup: Whether to create a backup

        Returns:
            True if file was cleaned successfully
        """
        try:
            # Read original content
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Clean content
            cleaned_content = self.clean_css_content(original_content)

            # Check if any changes were made
            if cleaned_content == original_content:
                logger.debug(f"No unsupported properties found in {file_path}")
                return True

            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                logger.info(f"Backup created: {backup_path}")

            # Write cleaned content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            logger.info(
                f"Cleaned {file_path}: removed {len(self.removed_properties)} unsupported properties"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {e}")
            return False

    def clean_python_file_stylesheets(self, file_path: Path, backup: bool = True) -> bool:
        """
        Clean CSS stylesheets embedded in Python files.

        Args:
            file_path: Path to Python file
            backup: Whether to create a backup

        Returns:
            True if file was cleaned successfully
        """
        try:
            # Read original content
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            cleaned_content = original_content
            total_removed = 0

            # Find CSS in setStyleSheet calls
            stylesheet_pattern = r'(setStyleSheet\s*\(\s*["\'])(.*?)(["\'])'

            def clean_stylesheet_match(match):
                nonlocal total_removed
                prefix = match.group(1)
                css_content = match.group(2)
                suffix = match.group(3)

                # Clean the CSS content
                cleaned_css = self.clean_css_content(css_content)
                total_removed += len(self.removed_properties)

                return f"{prefix}{cleaned_css}{suffix}"

            # Apply cleaning to all stylesheet matches
            cleaned_content = re.sub(
                stylesheet_pattern, clean_stylesheet_match, cleaned_content, flags=re.DOTALL
            )

            # Check if any changes were made
            if cleaned_content == original_content:
                logger.debug(f"No unsupported CSS properties found in {file_path}")
                return True

            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                # Use async file operations to prevent UI blocking
                from ..core.async_converter import async_file_write
                import asyncio
                asyncio.create_task(async_file_write(backup_path, original_content))
                logger.info(f"Backup created: {backup_path}")

            # Write cleaned content using async operation
            from ..core.async_converter import async_file_write
            import asyncio
            asyncio.create_task(async_file_write(file_path, cleaned_content))

            logger.info(f"Cleaned {file_path}: removed {total_removed} unsupported CSS properties")
            return True

        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {e}")
            return False

    def get_removal_summary(self) -> Dict[str, int]:
        """Get summary of removed properties"""
        summary = {}
        for prop, _ in self.removed_properties:
            summary[prop] = summary.get(prop, 0) + 1
        return summary


def clean_project_css(project_root: Path, file_patterns: List[str] = None) -> Dict[str, int]:
    """
    Clean all CSS in a project directory.

    Args:
        project_root: Root directory of the project
        file_patterns: List of file patterns to search (default: ['*.py', '*.css'])

    Returns:
        Summary of cleaning results
    """
    if file_patterns is None:
        file_patterns = ["*.py", "*.css"]

    cleaner = CSSCleaner()
    results = {"files_processed": 0, "files_cleaned": 0, "properties_removed": 0, "errors": 0}

    for pattern in file_patterns:
        for file_path in project_root.rglob(pattern):
            if file_path.is_file():
                results["files_processed"] += 1

                try:
                    if file_path.suffix == ".css":
                        success = cleaner.clean_css_file(file_path)
                    elif file_path.suffix == ".py":
                        success = cleaner.clean_python_file_stylesheets(file_path)
                    else:
                        continue

                    if success:
                        if cleaner.removed_properties:
                            results["files_cleaned"] += 1
                            results["properties_removed"] += len(cleaner.removed_properties)
                    else:
                        results["errors"] += 1

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results["errors"] += 1

    return results


def create_qt_compatible_stylesheet(original_css: str) -> str:
    """
    Create a Qt-compatible version of a CSS stylesheet.

    Args:
        original_css: Original CSS content

    Returns:
        Qt-compatible CSS content
    """
    cleaner = CSSCleaner(add_comments=True)
    return cleaner.clean_css_content(original_css)


# Convenience function for quick cleaning
def quick_clean_css(css_content: str) -> str:
    """Quickly clean CSS content without comments"""
    cleaner = CSSCleaner(add_comments=False)
    return cleaner.clean_css_content(css_content)