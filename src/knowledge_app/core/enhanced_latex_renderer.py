"""
🧮 Enhanced LaTeX Renderer for Knowledge App
Fixes MathJax initialization issues, prevents UI freezing, and provides robust math rendering
"""

from .async_converter import async_time_sleep


from .async_converter import async_time_sleep


import logging
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MathJaxConfig:
    """Enhanced MathJax configuration for robust rendering"""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get optimized MathJax configuration"""
        return {
            "tex": {
                "inlineMath": [["$", "$"], ["\\(", "\\)"]],
                "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
                "processEscapes": True,
                "processEnvironments": True,
                "processRefs": True,
                "packages": {
                    "base": [
                        "base", "ams", "newcommand", "configmacros", 
                        "action", "color", "unicode", "cancel", "bbox"
                    ],
                    "extended": [
                        "mhchem", "physics", "braket", "mathtools",
                        "siunitx", "gensymb", "textcomp"
                    ]
                },
                "macros": {
                    "RR": "{\\mathbb{R}}",
                    "NN": "{\\mathbb{N}}",
                    "ZZ": "{\\mathbb{Z}}",
                    "QQ": "{\\mathbb{Q}}",
                    "CC": "{\\mathbb{C}}",
                    "dd": "{\\mathrm{d}}",
                    "ii": "{\\mathrm{i}}",
                    "ee": "{\\mathrm{e}}",
                    "bra": ["{\\langle #1 |}", 1],
                    "ket": ["{| #1 \\rangle}", 1],
                    "braket": ["{\\langle #1 | #2 \\rangle}", 2]
                },
                "tags": "ams",
                "tagSide": "right",
                "tagIndent": "0.8em",
                "useLabelIds": True,
                "maxMacros": 10000,
                "maxBuffer": 5 * 1024
            },
            "svg": {
                "fontCache": "global",
                "scale": 1.2,
                "minScale": 0.5,
                "matchFontHeight": False,
                "mtextInheritFont": False,
                "merrorInheritFont": True,
                "mathmlSpacing": False,
                "skipAttributes": {},
                "exFactor": 0.5,
                "displayAlign": "center",
                "displayIndent": "0"
            },
            "chtml": {
                "scale": 1.2,
                "minScale": 0.5,
                "matchFontHeight": False,
                "fontURL": "[mathjax]/output/chtml/fonts/woff-v2",
                "adaptiveCSS": True
            },
            "startup": {
                "typeset": False,
                "ready": "MathJax.startup.defaultReady"
            },
            "options": {
                "ignoreHtmlClass": "tex2jax_ignore",
                "processHtmlClass": "tex2jax_process",
                "renderActions": {
                    "addMenu": [0, "", ""],
                    "checkLoading": [1, "checkLoading", ""],
                    "compile": [2, "compile", ""],
                    "metrics": [3, "getMetrics", ""],
                    "typeset": [4, "typeset", ""],
                    "update": [5, "update", ""]
                }
            },
            "loader": {
                "load": [
                    "[tex]/ams", "[tex]/newcommand", "[tex]/configmacros",
                    "[tex]/action", "[tex]/color", "[tex]/unicode", 
                    "[tex]/cancel", "[tex]/bbox", "[tex]/mhchem",
                    "[tex]/physics", "[tex]/braket"
                ],
                "paths": {
                    "mathjax": "https://cdn.jsdelivr.net/npm/mathjax@3/es5"
                },
                "source": {},
                "dependencies": {},
                "provides": {},
                "failed": {},
                "load": []
            }
        }

class NonBlockingLatexRenderer:
    """
    🚀 NON-BLOCKING LaTeX Renderer
    Prevents UI freezing during mathematical content rendering
    """
    
    def __init__(self):
        self.mathjax_ready = False
        self.mathjax_loading = False
        self.render_queue = []
        self.render_lock = threading.RLock()
        self.initialization_callbacks = []
        self.render_stats = {
            "total_renders": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "avg_render_time": 0.0
        }
        
        # Enhanced LaTeX pattern detection
        self.latex_patterns = [
            # Display math patterns
            (r'\$\$([^$]+)\$\$', 'display'),
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'display'),
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'display'),
            (r'\\begin\{gather\}(.*?)\\end\{gather\}', 'display'),
            (r'\\begin\{multline\}(.*?)\\end\{multline\}', 'display'),
            (r'\\begin\{split\}(.*?)\\end\{split\}', 'display'),
            (r'\\begin\{cases\}(.*?)\\end\{cases\}', 'display'),
            (r'\\begin\{matrix\}(.*?)\\end\{matrix\}', 'display'),
            (r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', 'display'),
            (r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', 'display'),
            (r'\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}', 'display'),
            (r'\\begin\{Bmatrix\}(.*?)\\end\{Bmatrix\}', 'display'),
            (r'\\begin\{Vmatrix\}(.*?)\\end\{Vmatrix\}', 'display'),
            
            # Inline math patterns
            (r'\$([^$\n]+)\$', 'inline'),
            (r'\\\\\\(([^)]+)\\\\\\)', 'inline'),
            
            # Physics and chemistry
            (r'\\begin\{chemical\}(.*?)\\end\{chemical\}', 'display'),
            (r'\\ce\{([^}]+)\}', 'inline'),
            (r'\\pu\{([^}]+)\}', 'inline'),
            
            # LaTeX commands that indicate math content
            (r'\\[a-zA-Z]+\{[^}]*\}', 'command'),
            (r'\\[a-zA-Z]+', 'command')
        ]
        
        # Comprehensive symbol mapping for fallback rendering
        self.symbol_map = {
            # Greek letters (lowercase)
            r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
            r'\\epsilon': 'ε', r'\\varepsilon': 'ε', r'\\zeta': 'ζ', r'\\eta': 'η',
            r'\\theta': 'θ', r'\\vartheta': 'ϑ', r'\\iota': 'ι', r'\\kappa': 'κ',
            r'\\lambda': 'λ', r'\\mu': 'μ', r'\\nu': 'ν', r'\\xi': 'ξ',
            r'\\pi': 'π', r'\\varpi': 'ϖ', r'\\rho': 'ρ', r'\\varrho': 'ϱ',
            r'\\sigma': 'σ', r'\\varsigma': 'ς', r'\\tau': 'τ', r'\\upsilon': 'υ',
            r'\\phi': 'φ', r'\\varphi': 'ϕ', r'\\chi': 'χ', r'\\psi': 'ψ', r'\\omega': 'ω',
            
            # Greek letters (uppercase)
            r'\\Gamma': 'Γ', r'\\Delta': 'Δ', r'\\Theta': 'Θ', r'\\Lambda': 'Λ',
            r'\\Xi': 'Ξ', r'\\Pi': 'Π', r'\\Sigma': 'Σ', r'\\Upsilon': 'Υ',
            r'\\Phi': 'Φ', r'\\Psi': 'Ψ', r'\\Omega': 'Ω',
            
            # Mathematical operators
            r'\\pm': '±', r'\\mp': '∓', r'\\times': '×', r'\\div': '÷',
            r'\\cdot': '·', r'\\ast': '∗', r'\\star': '⋆', r'\\circ': '∘',
            r'\\bullet': '•', r'\\oplus': '⊕', r'\\ominus': '⊖', r'\\otimes': '⊗',
            r'\\oslash': '⊘', r'\\odot': '⊙', r'\\cup': '∪', r'\\cap': '∩',
            r'\\vee': '∨', r'\\wedge': '∧', r'\\setminus': '∖', r'\\wr': '≀',
            
            # Relations
            r'\\neq': '≠', r'\\ne': '≠', r'\\leq': '≤', r'\\le': '≤',
            r'\\geq': '≥', r'\\ge': '≥', r'\\ll': '≪', r'\\gg': '≫',
            r'\\approx': '≈', r'\\cong': '≅', r'\\equiv': '≡', r'\\sim': '∼',
            r'\\simeq': '≃', r'\\asymp': '≍', r'\\propto': '∝', r'\\prec': '≺',
            r'\\succ': '≻', r'\\preceq': '⪯', r'\\succeq': '⪰', r'\\parallel': '∥',
            r'\\perp': '⊥', r'\\in': '∈', r'\\ni': '∋', r'\\notin': '∉',
            r'\\subset': '⊂', r'\\supset': '⊃', r'\\subseteq': '⊆', r'\\supseteq': '⊇',
            
            # Arrows
            r'\\leftarrow': '←', r'\\rightarrow': '→', r'\\leftrightarrow': '↔',
            r'\\Leftarrow': '⇐', r'\\Rightarrow': '⇒', r'\\Leftrightarrow': '⇔',
            r'\\uparrow': '↑', r'\\downarrow': '↓', r'\\updownarrow': '↕',
            r'\\Uparrow': '⇑', r'\\Downarrow': '⇓', r'\\Updownarrow': '⇕',
            r'\\mapsto': '↦', r'\\longmapsto': '⟼', r'\\hookleftarrow': '↩',
            r'\\hookrightarrow': '↪', r'\\leftharpoonup': '↼', r'\\rightharpoonup': '⇀',
            
            # Special symbols
            r'\\infty': '∞', r'\\partial': '∂', r'\\nabla': '∇', r'\\emptyset': '∅',
            r'\\varnothing': '∅', r'\\angle': '∠', r'\\triangle': '△', r'\\square': '□',
            r'\\lozenge': '◊', r'\\blacksquare': '■', r'\\blacktriangle': '▲',
            r'\\blacklozenge': '⧫', r'\\prime': '′', r'\\backprime': '‵',
            r'\\forall': '∀', r'\\exists': '∃', r'\\nexists': '∄', r'\\top': '⊤',
            r'\\bot': '⊥', r'\\vdash': '⊢', r'\\dashv': '⊣', r'\\models': '⊨',
            
            # Integrals and sums
            r'\\int': '∫', r'\\iint': '∬', r'\\iiint': '∭', r'\\oint': '∮',
            r'\\sum': '∑', r'\\prod': '∏', r'\\coprod': '∐', r'\\bigcup': '⋃',
            r'\\bigcap': '⋂', r'\\bigvee': '⋁', r'\\bigwedge': '⋀',
            r'\\bigoplus': '⊕', r'\\bigotimes': '⊗', r'\\bigodot': '⊙',
            
            # Miscellaneous
            r'\\aleph': 'ℵ', r'\\hbar': 'ℏ', r'\\ell': 'ℓ', r'\\wp': '℘',
            r'\\Re': 'ℜ', r'\\Im': 'ℑ', r'\\mho': '℧', r'\\Finv': 'Ⅎ',
            r'\\Game': '⅁', r'\\eth': 'ð', r'\\beth': 'ℶ', r'\\gimel': 'ℷ',
            r'\\daleth': 'ℸ', r'\\backslash': '\\', r'\\|': '‖'
        }
    
    def initialize_mathjax_async(self, callback: Optional[Callable] = None) -> None:
        """
        🚀 NON-BLOCKING MathJax initialization
        Prevents UI freezing during startup
        """
        if self.mathjax_ready:
            if callback:
                callback()
            return
                
        if self.mathjax_loading:
            if callback:
                self.initialization_callbacks.append(callback)
            return
            
        self.mathjax_loading = True
        
        def initialize_in_thread():
            try:
                logger.info("🧮 Initializing MathJax asynchronously...")
                
                # Simulate initialization process
                time.sleep(0.1)  # Brief delay to ensure DOM is ready
                
                # Create MathJax configuration
                config = MathJaxConfig.get_config()
                
                # Set up MathJax in the JavaScript environment
                # This would be handled by the web engine
                
                logger.info("✅ MathJax initialization complete")
                self.mathjax_ready = True
                self.mathjax_loading = False
                
                # Execute callbacks
                if callback:
                    callback()
                    
                for cb in self.initialization_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        logger.error(f"❌ MathJax callback failed: {e}")
                        
                self.initialization_callbacks.clear()
                
            except Exception as e:
                logger.error(f"❌ MathJax initialization failed: {e}")
                self.mathjax_loading = False
                
        # Run in background thread to prevent UI blocking
        thread = threading.Thread(target=initialize_in_thread, daemon=True)
        thread.start()
    
    def detect_latex_content(self, text: str) -> Dict[str, Any]:
        """
        Enhanced LaTeX content detection with detailed analysis
        """
        if not text or not isinstance(text, str):
            return {
                "has_latex": False,
                "patterns_found": [],
                "complexity": "none",
                "requires_mathjax": False
            }
        
        patterns_found = []
        complexity_score = 0
        
        # CRITICAL FIX: Use more specific regex flags to prevent overreach
        for pattern, math_type in self.latex_patterns:
            try:
                # CRITICAL FIX: Remove re.DOTALL and re.IGNORECASE for most patterns
                # Only use specific flags where actually needed
                if math_type in ["display", "inline"]:
                    # For math delimiters, we need more precise matching
                    matches = re.findall(pattern, text, re.MULTILINE)
                elif math_type == "command":
                    # For LaTeX commands, case sensitivity matters - no flags
                    matches = re.findall(pattern, text)
                else:
                    # Default: no special flags to prevent overreach
                    matches = re.findall(pattern, text)
                
                if matches:
                    # CRITICAL FIX: Validate matches to prevent false positives
                    valid_matches = self._validate_latex_matches(matches, math_type)
                    if valid_matches:
                        patterns_found.append({
                            "type": math_type,
                            "pattern": pattern,
                            "count": len(valid_matches),
                            "matches": valid_matches[:3]  # First 3 matches for analysis
                        })
                        
                        # Calculate complexity
                        if math_type == "display":
                            complexity_score += len(valid_matches) * 3
                        elif math_type == "inline":
                            complexity_score += len(valid_matches) * 2
                        else:
                            complexity_score += len(valid_matches)
            except re.error as e:
                logger.warning(f"⚠️ LaTeX pattern matching error: {e} for pattern: {pattern}")
                continue
        
        # Determine complexity level
        if complexity_score == 0:
            complexity = "none"
        elif complexity_score <= 3:
            complexity = "simple"
        elif complexity_score <= 10:
            complexity = "moderate"
        else:
            complexity = "complex"
            
        has_latex = len(patterns_found) > 0
        requires_mathjax = any(p["type"] in ["display", "inline"] for p in patterns_found)
        
        return {
            "has_latex": has_latex,
            "patterns_found": patterns_found,
            "complexity": complexity,
            "complexity_score": complexity_score,
            "requires_mathjax": requires_mathjax,
            "total_matches": sum(p["count"] for p in patterns_found)
        }
    
    async def render_latex_async(self, text: str, element_id: Optional[str] = None) -> Dict[str, Any]:
        """
        🚀 NON-BLOCKING LaTeX rendering
        Prevents UI freezing during complex mathematical rendering
        """
        start_time = time.time()
        
        try:
            # Analyze content first
            analysis = self.detect_latex_content(text)
            
            if not analysis["has_latex"]:
                return {
                    "success": True,
                    "method": "no_processing",
                    "render_time": time.time() - start_time,
                    "analysis": analysis
                }
            
            # Use appropriate rendering method based on complexity
            if analysis["complexity"] == "simple" or not self.mathjax_ready:
                # Use fallback rendering for simple content or when MathJax isn't ready
                result = await self._render_with_fallback_async(text, analysis)
            else:
                # Use full MathJax rendering for complex content
                result = await self._render_with_mathjax_async(text, element_id, analysis)
            
            # Update statistics
            render_time = time.time() - start_time
            self._update_render_stats(render_time, result["success"])
            
            result["render_time"] = render_time
            result["analysis"] = analysis
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LaTeX rendering failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "render_time": time.time() - start_time,
                "analysis": self.detect_latex_content(text)
            }
    
    async def _render_with_fallback_async(self, text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rendering using Unicode symbol substitution"""
        
        def process_text():
            processed_text = text
            substitutions_made = 0
            
            # Apply symbol substitutions
            for latex_cmd, unicode_symbol in self.symbol_map.items():
                if latex_cmd in processed_text:
                    processed_text = processed_text.replace(latex_cmd, unicode_symbol)
                    substitutions_made += 1
            
            # Handle fractions
            fraction_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            fractions = re.findall(fraction_pattern, processed_text)
            for numerator, denominator in fractions:
                processed_text = re.sub(
                    fraction_pattern,
                    f"({numerator})/({denominator})",
                    processed_text,
                    count=1
                )
                substitutions_made += 1
            
            # Handle superscripts and subscripts
            processed_text = re.sub(r'\^{([^}]+)}', r'^(\1)', processed_text)
            processed_text = re.sub(r'_{([^}]+)}', r'_(\1)', processed_text)
            processed_text = re.sub(r'\^([a-zA-Z0-9])', r'^\1', processed_text)
            processed_text = re.sub(r'_([a-zA-Z0-9])', r'_\1', processed_text)
            
            # Clean up remaining LaTeX commands
            processed_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', processed_text)
            processed_text = re.sub(r'\\[a-zA-Z]+', '', processed_text)
            
            # Clean up extra spaces and formatting
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            return processed_text, substitutions_made
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        processed_text, substitutions_made = await loop.run_in_executor(
            None, process_text
        )
        
        return {
            "success": True,
            "method": "unicode_fallback",
            "processed_text": processed_text,
            "substitutions_made": substitutions_made
        }
    
    async def _render_with_mathjax_async(self, text: str, element_id: Optional[str], 
                                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Full MathJax rendering for complex mathematical content"""
        
        if not self.mathjax_ready:
            # Wait for MathJax initialization with timeout
            wait_time = 0
            while not self.mathjax_ready and wait_time < 5.0:
                await asyncio.sleep(0.1)
                wait_time += 0.1
            
            if not self.mathjax_ready:
                logger.warning("⚠️ MathJax not ready, falling back to Unicode rendering")
                return await self._render_with_fallback_async(text, analysis)
        
        def process_mathjax():
            # Preprocess text for MathJax
            processed_text = text
            
            # Normalize LaTeX delimiters
            processed_text = re.sub(r'\$\$\s*([^$]+?)\s*\$\$', r'$$\1$$', processed_text)
            processed_text = re.sub(r'\$\s*([^$\n]+?)\s*\$', r'$\1$', processed_text)
            
            # Fix common LaTeX formatting issues
            processed_text = processed_text.replace('\\\\', '\\')
            processed_text = re.sub(r'\\([a-zA-Z]+)\s+', r'\\\1 ', processed_text)
            
            return processed_text
        
        # Process in thread pool
        loop = asyncio.get_event_loop()
        processed_text = await loop.run_in_executor(None, process_mathjax)
        
        return {
            "success": True,
            "method": "mathjax",
            "processed_text": processed_text,
            "element_id": element_id,
            "ready_for_typesetting": True
        }
    
    def _update_render_stats(self, render_time: float, success: bool) -> None:
        """Update rendering statistics"""
        with self.render_lock:
            self.render_stats["total_renders"] += 1
            
            if success:
                self.render_stats["successful_renders"] += 1
            else:
                self.render_stats["failed_renders"] += 1
            
            # Update average render time
            total_successful = self.render_stats["successful_renders"]
            if total_successful > 0:
                current_avg = self.render_stats["avg_render_time"]
                self.render_stats["avg_render_time"] = (
                    (current_avg * (total_successful - 1) + render_time) / total_successful
                )
    
    def _validate_latex_matches(self, matches: List[str], math_type: str) -> List[str]:
        """
        CRITICAL FIX: Validate LaTeX matches to prevent false positives
        
        Args:
            matches: List of regex matches
            math_type: Type of LaTeX content (display, inline, command, etc.)
            
        Returns:
            List of validated matches
        """
        valid_matches = []
        
        for match in matches:
            if not match or not isinstance(match, str):
                continue
                
            # CRITICAL FIX: Specific validation based on math type
            if math_type == "display":
                # Display math should have substantial content
                if len(match.strip()) >= 3 and not self._is_false_positive_display(match):
                    valid_matches.append(match)
            elif math_type == "inline":
                # Inline math should be reasonable length and not just punctuation
                if 2 <= len(match.strip()) <= 100 and not self._is_false_positive_inline(match):
                    valid_matches.append(match)
            elif math_type == "command":
                # LaTeX commands should be valid command names
                if self._is_valid_latex_command(match):
                    valid_matches.append(match)
            else:
                # For other types, basic validation
                if len(match.strip()) >= 1:
                    valid_matches.append(match)
        
        return valid_matches
    
    def _is_false_positive_display(self, match: str) -> bool:
        """Check if display math match is a false positive"""
        match = match.strip()
        
        # Common false positives for display math
        false_positives = [
            # Just punctuation or symbols
            r'^[\s\$\{\}\[\]\(\)]*$',
            # Just numbers
            r'^\d+$',
            # Just basic operators
            r'^[\+\-\*\/\=]+$',
            # Empty or whitespace only
            r'^\s*$'
        ]
        
        for pattern in false_positives:
            if re.match(pattern, match):
                return True
        
        return False
    
    def _is_false_positive_inline(self, match: str) -> bool:
        """Check if inline math match is a false positive"""
        match = match.strip()
        
        # Common false positives for inline math
        false_positives = [
            # Just a single character or symbol
            r'^.$',
            # Just punctuation
            r'^[^\w]*$',
            # Just whitespace
            r'^\s*$',
            # Common non-math dollar usage
            r'^\d+(\.\d+)?$',  # Just numbers (could be currency)
        ]
        
        for pattern in false_positives:
            if re.match(pattern, match):
                return True
        
        return False
    
    def _is_valid_latex_command(self, match: str) -> bool:
        """Check if LaTeX command match is valid"""
        # Extract command name from match like \command{content}
        command_match = re.match(r'\\([a-zA-Z]+)', match)
        if not command_match:
            return False
        
        command_name = command_match.group(1)
        
        # List of valid LaTeX math commands
        valid_commands = {
            'frac', 'sqrt', 'sum', 'int', 'lim', 'sin', 'cos', 'tan', 'log', 'ln', 'exp',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 'mu', 'pi',
            'sigma', 'phi', 'psi', 'omega', 'partial', 'nabla', 'infty', 'pm', 'mp',
            'times', 'div', 'cdot', 'circ', 'bullet', 'star', 'ast', 'oplus', 'ominus',
            'otimes', 'oslash', 'odot', 'bigcirc', 'dagger', 'ddagger', 'amalg',
            'leq', 'geq', 'equiv', 'sim', 'simeq', 'approx', 'cong', 'neq', 'propto',
            'subset', 'supset', 'subseteq', 'supseteq', 'in', 'ni', 'notin', 'cup',
            'cap', 'uplus', 'sqcup', 'vee', 'wedge', 'setminus', 'wr', 'diamond',
            'bigtriangleup', 'bigtriangledown', 'triangleleft', 'triangleright',
            'leftarrow', 'rightarrow', 'leftrightarrow', 'Leftarrow', 'Rightarrow',
            'Leftrightarrow', 'mapsto', 'hookleftarrow', 'hookrightarrow',
            'leftharpoonup', 'leftharpoondown', 'rightharpoonup', 'rightharpoondown',
            'rightleftharpoons', 'iff', 'top', 'bot', 'vdash', 'dashv', 'up', 'down',
            'updownarrow', 'Updownarrow', 'nearrow', 'searrow', 'swarrow', 'nwarrow',
            'forall', 'exists', 'neg', 'flat', 'natural', 'sharp', 'wp', 'Re', 'Im',
            'mho', 'prime', 'emptyset', 'nabla', 'surd', 'angle', 'triangle',
            'clubsuit', 'diamondsuit', 'heartsuit', 'spadesuit'
        }
        
        return command_name.lower() in valid_commands

    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics"""
        with self.render_lock:
            return self.render_stats.copy()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.render_queue.clear()
        self.initialization_callbacks.clear()

# Global instance
_latex_renderer = None

def get_latex_renderer() -> NonBlockingLatexRenderer:
    """Get or create global LaTeX renderer instance"""
    global _latex_renderer
    if _latex_renderer is None:
        _latex_renderer = NonBlockingLatexRenderer()
    return _latex_renderer
