#!/usr/bin/env python3
"""
Intelligent Topic Resolver - Makes ANY input generate meaningful questions
Handles abbreviations, random text, typos, and ambiguous inputs
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import requests
import json

logger = logging.getLogger(__name__)

class IntelligentTopicResolver:
    """
    🧠 INTELLIGENT TOPIC RESOLVER
    Converts ANY input into meaningful educational content
    """
    
    def __init__(self):
        self.acronym_database = {
            # Computer Science
            "dfs": ["Depth-First Search", "Distributed File System", "Dynamic File System"],
            "bfs": ["Breadth-First Search", "Best-First Search"],
            "ai": ["Artificial Intelligence", "Adobe Illustrator"],
            "ml": ["Machine Learning", "Markup Language"],
            "api": ["Application Programming Interface"],
            "sql": ["Structured Query Language"],
            "css": ["Cascading Style Sheets"],
            "html": ["HyperText Markup Language"],
            "http": ["HyperText Transfer Protocol"],
            "tcp": ["Transmission Control Protocol"],
            "udp": ["User Datagram Protocol"],
            "dns": ["Domain Name System"],
            "url": ["Uniform Resource Locator"],
            "gui": ["Graphical User Interface"],
            "cli": ["Command Line Interface"],
            "ide": ["Integrated Development Environment"],
            "sdk": ["Software Development Kit"],
            "orm": ["Object-Relational Mapping"],
            "mvc": ["Model-View-Controller"],
            "crud": ["Create, Read, Update, Delete"],
            
            # Science & Math
            "dna": ["Deoxyribonucleic Acid"],
            "rna": ["Ribonucleic Acid"],
            "atp": ["Adenosine Triphosphate"],
            "gps": ["Global Positioning System"],
            "led": ["Light Emitting Diode"],
            "lcd": ["Liquid Crystal Display"],
            "cpu": ["Central Processing Unit"],
            "gpu": ["Graphics Processing Unit"],
            "ram": ["Random Access Memory"],
            "ssd": ["Solid State Drive"],
            "hdd": ["Hard Disk Drive"],
            "usb": ["Universal Serial Bus"],
            "wifi": ["Wireless Fidelity"],
            "bluetooth": ["Bluetooth wireless technology"],
            
            # Business & Finance
            "roi": ["Return on Investment"],
            "kpi": ["Key Performance Indicator"],
            "ceo": ["Chief Executive Officer"],
            "cto": ["Chief Technology Officer"],
            "hr": ["Human Resources"],
            "pr": ["Public Relations"],
            "b2b": ["Business to Business"],
            "b2c": ["Business to Consumer"],
            "saas": ["Software as a Service"],
            "crm": ["Customer Relationship Management"],
            "erp": ["Enterprise Resource Planning"],
            
            # General
            "usa": ["United States of America"],
            "uk": ["United Kingdom"],
            "eu": ["European Union"],
            "un": ["United Nations"],
            "who": ["World Health Organization"],
            "nasa": ["National Aeronautics and Space Administration"],
            "fbi": ["Federal Bureau of Investigation"],
            "cia": ["Central Intelligence Agency"],
        }
        
        self.topic_patterns = {
            # Math patterns
            r'\d+\s*[\+\-\*/]\s*\d+': 'mathematics',
            r'equation|formula|calculate|solve': 'mathematics',
            r'algebra|geometry|calculus|trigonometry': 'mathematics',
            
            # Science patterns
            r'atom|molecule|element|chemical': 'chemistry',
            r'cell|organism|biology|evolution': 'biology',
            r'force|energy|motion|physics': 'physics',
            r'planet|star|galaxy|astronomy': 'astronomy',
            
            # Technology patterns
            r'computer|software|programming|code': 'computer science',
            r'algorithm|data structure|database': 'computer science',
            r'network|internet|web|server': 'networking',
            
            # History patterns
            r'war|battle|empire|civilization': 'history',
            r'ancient|medieval|renaissance': 'history',
            
            # Language patterns
            r'grammar|syntax|language|literature': 'linguistics',
            r'novel|poem|author|writing': 'literature',
        }
    
    def resolve_topic(self, raw_input: str, content_filter_callback=None) -> Dict[str, any]:
        """
        🚀 BUG FIX 32: INTELLIGENT RESOLUTION with content filtering integration

        This method now respects content filtering and will NOT provide fallbacks
        for inappropriate content, preventing content filter bypass attacks.

        Args:
            raw_input: Any user input (abbreviation, typo, random text, etc.)
            content_filter_callback: Optional callback to check content safety

        Returns:
            Dict with resolved topic info and context, or error for inappropriate content
        """
        if not raw_input or not raw_input.strip():
            return self._create_general_topic()

        # 🚀 BUG FIX 32: Check content safety BEFORE any resolution attempts
        if content_filter_callback:
            filter_result = content_filter_callback(raw_input)
            if not filter_result.get("is_safe", True):
                # Return error result instead of fallback for inappropriate content
                return {
                    "original_input": raw_input,
                    "resolved_topic": None,
                    "subject_area": None,
                    "context": None,
                    "resolution_method": "content_filter_blocked",
                    "confidence": 0.0,
                    "error": f"Content policy violation: {filter_result.get('reason', 'inappropriate content')}",
                    "is_safe": False
                }

        input_clean = raw_input.strip().lower()

        # Step 1: Check if it's a known acronym
        acronym_result = self._resolve_acronym(input_clean)
        if acronym_result:
            return acronym_result

        # Step 2: Pattern matching for topic detection
        pattern_result = self._detect_topic_patterns(input_clean)
        if pattern_result:
            return pattern_result

        # Step 3: Fuzzy matching for common topics
        fuzzy_result = self._fuzzy_topic_match(input_clean)
        if fuzzy_result:
            return fuzzy_result

        # Step 4: Intelligent interpretation for random text
        interpretation_result = self._intelligent_interpretation(raw_input)
        if interpretation_result:
            return interpretation_result

        # 🚀 BUG FIX 32: Modified fallback behavior
        # Instead of always falling back to "general knowledge", be more explicit
        # about what we're doing to prevent inappropriate content laundering
        return self._create_explicit_fallback_topic(raw_input)
    
    def _resolve_acronym(self, input_text: str) -> Optional[Dict[str, any]]:
        """Resolve known acronyms"""
        if input_text in self.acronym_database:
            meanings = self.acronym_database[input_text]
            primary_meaning = meanings[0]
            
            # Determine the subject area
            subject = self._determine_subject_area(primary_meaning)
            
            return {
                "original_input": input_text.upper(),
                "resolved_topic": primary_meaning,
                "subject_area": subject,
                "alternative_meanings": meanings[1:] if len(meanings) > 1 else [],
                "context": f"The acronym '{input_text.upper()}' commonly refers to {primary_meaning}",
                "resolution_method": "acronym_database",
                "confidence": 0.9
            }
        return None
    
    def _detect_topic_patterns(self, input_text: str) -> Optional[Dict[str, any]]:
        """Detect topics using regex patterns with rich context generation"""
        for pattern, topic in self.topic_patterns.items():
            if re.search(pattern, input_text, re.IGNORECASE):
                # Generate rich, domain-specific context
                rich_context = self._generate_rich_context(input_text, topic)

                return {
                    "original_input": input_text,
                    "resolved_topic": topic,
                    "subject_area": topic,
                    "context": rich_context,
                    "resolution_method": "pattern_matching",
                    "confidence": 0.8
                }
        return None

    def _generate_rich_context(self, input_text: str, topic: str) -> str:
        """Generate rich, domain-specific context for better question generation"""

        # Chemistry-specific rich contexts
        if topic == "chemistry":
            if "atom" in input_text.lower():
                return """Atomic structure and properties: electron configuration, quantum mechanical model, orbital theory, periodic trends, atomic radius, ionization energy, electronegativity, chemical bonding theories (VSEPR, molecular orbital theory), intermolecular forces, and their applications in predicting chemical behavior and reactivity patterns."""
            elif "molecule" in input_text.lower():
                return """Molecular structure and bonding: covalent bonding, molecular geometry, hybridization, resonance structures, polarity, intermolecular forces, molecular orbital theory, spectroscopic analysis, and structure-property relationships in organic and inorganic compounds."""
            elif "element" in input_text.lower():
                return """Periodic properties and trends: atomic structure, electron configuration, periodic law, group and period trends, metallic character, oxidation states, chemical reactivity patterns, and applications in predicting chemical behavior and compound formation."""
            else:
                return """General chemistry principles: atomic theory, chemical bonding, molecular structure, thermodynamics, kinetics, equilibrium, acid-base chemistry, electrochemistry, and their applications in understanding chemical processes and reactions."""

        # Physics-specific rich contexts
        elif topic == "physics":
            return """Physics principles: classical mechanics, thermodynamics, electromagnetism, quantum mechanics, relativity, wave phenomena, optics, and their mathematical formulations and real-world applications."""

        # Mathematics-specific rich contexts
        elif topic == "mathematics":
            return """Mathematical concepts: algebra, calculus, geometry, statistics, discrete mathematics, mathematical proofs, problem-solving techniques, and their applications across various fields."""

        # Biology-specific rich contexts
        elif topic == "biology":
            return """Biological systems: cell biology, genetics, evolution, ecology, physiology, molecular biology, biochemistry, and their interconnections in living organisms."""

        # Default rich context
        else:
            return f"""Advanced {topic} concepts: theoretical foundations, practical applications, current research developments, interdisciplinary connections, and critical thinking approaches in {topic}."""
    
    def _fuzzy_topic_match(self, input_text: str) -> Optional[Dict[str, any]]:
        """Fuzzy matching for common topics with typos"""
        common_topics = {
            "physics": ["physic", "phisics", "fisics", "physics"],
            "mathematics": ["math", "maths", "mathmatics", "matematics"],
            "chemistry": ["chem", "chemestry", "kemistry"],
            "biology": ["bio", "biologi", "biolgy"],
            "history": ["histori", "historie", "histry"],
            "geography": ["geo", "geografi", "geografy"],
            "computer science": ["cs", "comp sci", "computing", "computers"],
            "programming": ["coding", "code", "prog", "programing"],
            "literature": ["lit", "english", "writing"],
            "economics": ["econ", "economy", "economic"],
            "psychology": ["psych", "psycology", "psychology"],
        }
        
        for topic, variations in common_topics.items():
            for variation in variations:
                if self._similarity_score(input_text, variation) > 0.7:
                    return {
                        "original_input": input_text,
                        "resolved_topic": topic,
                        "subject_area": topic,
                        "context": f"Interpreted '{input_text}' as {topic} (fuzzy match)",
                        "resolution_method": "fuzzy_matching",
                        "confidence": 0.7
                    }
        return None
    
    def _intelligent_interpretation(self, input_text: str) -> Optional[Dict[str, any]]:
        """Intelligent interpretation of random or unclear input"""
        input_clean = input_text.strip().lower()
        
        # Check for single letters or very short inputs
        if len(input_clean) <= 3:
            interpretations = {
                "a": "alphabet and linguistics",
                "b": "biology and life sciences", 
                "c": "chemistry and chemical processes",
                "d": "data structures and algorithms",
                "e": "energy and physics",
                "f": "functions and mathematics",
                "g": "geometry and spatial reasoning",
                "h": "history and human civilization",
                "i": "information technology",
                "j": "journalism and communication",
                "k": "knowledge management",
                "l": "logic and reasoning",
                "m": "mathematics and modeling",
                "n": "natural sciences",
                "o": "operations and optimization",
                "p": "physics and natural phenomena",
                "q": "quantum mechanics and advanced physics",
                "r": "research methodology",
                "s": "science and scientific method",
                "t": "technology and innovation",
                "u": "universe and cosmology",
                "v": "variables and algebra",
                "w": "world geography and cultures",
                "x": "unknown variables and problem solving",
                "y": "systems analysis",
                "z": "zoology and animal sciences"
            }
            
            if input_clean in interpretations:
                topic = interpretations[input_clean]
                return {
                    "original_input": input_text,
                    "resolved_topic": topic,
                    "subject_area": topic.split(" and ")[0],
                    "context": f"Interpreted single character '{input_text}' as {topic}",
                    "resolution_method": "intelligent_interpretation",
                    "confidence": 0.6
                }
        
        # Check for number patterns
        if input_clean.isdigit():
            number = int(input_clean)
            if number <= 12:
                return {
                    "original_input": input_text,
                    "resolved_topic": "number theory and mathematics",
                    "subject_area": "mathematics",
                    "context": f"Number {number} - exploring mathematical properties and applications",
                    "resolution_method": "number_interpretation",
                    "confidence": 0.7
                }
        
        return None
    
    def _create_adaptive_topic(self, input_text: str) -> Dict[str, any]:
        """🔧 FIX: Create educational content with transparent fallback explanation"""
        # Analyze the input characteristics
        has_numbers = bool(re.search(r'\d', input_text))
        has_special_chars = bool(re.search(r'[^a-zA-Z0-9\s]', input_text))
        length = len(input_text.strip())

        # 🔧 FIX: Make fallback behavior transparent to user
        fallback_explanation = f"Couldn't understand '{input_text}', so here's a question about"

        # Determine the best educational angle
        if has_numbers and has_special_chars:
            topic = "data analysis and pattern recognition"
            context = f"Analyzing the structure and patterns in complex data"
            explanation = f"{fallback_explanation} data analysis instead!"
        elif has_numbers:
            topic = "numerical analysis and mathematics"
            context = f"Exploring mathematical concepts and numerical methods"
            explanation = f"{fallback_explanation} mathematics instead!"
        elif length <= 5:
            topic = "linguistics and communication"
            explanation = f"{fallback_explanation} language and communication instead!"
            context = f"Studying the linguistic properties of '{input_text}'"
        else:
            topic = "critical thinking and problem solving"
            context = f"Using '{input_text}' as a case study in analytical thinking"
        
        return {
            "original_input": input_text,
            "resolved_topic": topic,
            "subject_area": topic.split(" and ")[0],
            "context": context,
            "resolution_method": "adaptive_creation",
            "confidence": 0.5,
            "note": "Generated educational content from user input",
            "fallback_explanation": explanation  # 🔧 FIX: Transparent fallback explanation
        }
    
    def _create_explicit_fallback_topic(self, input_text: str) -> Dict[str, any]:
        """
        🚀 BUG FIX 32: Create explicit fallback that makes it clear we couldn't understand the input

        This method replaces the generic "general knowledge" fallback to prevent
        inappropriate content from being laundered into safe topics.
        """
        return {
            "original_input": input_text,
            "resolved_topic": None,  # Explicitly no topic resolved
            "subject_area": None,
            "context": None,
            "resolution_method": "failed_to_resolve",
            "confidence": 0.0,
            "error": f"Could not understand or resolve the input: '{input_text}'. Please provide a clearer topic.",
            "suggestion": "Try using specific subject names like 'Physics', 'Chemistry', 'Biology', 'Mathematics', etc."
        }

    def _create_general_topic(self) -> Dict[str, any]:
        """Create general knowledge topic for empty input only"""
        return {
            "original_input": "",
            "resolved_topic": "general knowledge and critical thinking",
            "subject_area": "general knowledge",
            "context": "Exploring fundamental concepts across multiple disciplines",
            "resolution_method": "default_general",
            "confidence": 0.8
        }
    
    def _determine_subject_area(self, topic: str) -> str:
        """Determine the subject area for a resolved topic"""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["computer", "software", "algorithm", "data", "programming"]):
            return "computer science"
        elif any(word in topic_lower for word in ["math", "equation", "number", "calculate"]):
            return "mathematics"
        elif any(word in topic_lower for word in ["physics", "force", "energy", "motion"]):
            return "physics"
        elif any(word in topic_lower for word in ["chemistry", "chemical", "molecule", "atom"]):
            return "chemistry"
        elif any(word in topic_lower for word in ["biology", "cell", "organism", "life"]):
            return "biology"
        elif any(word in topic_lower for word in ["history", "historical", "ancient", "civilization"]):
            return "history"
        else:
            return "general knowledge"
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein-like similarity
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str2 if len(str1) > len(str2) else str1
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters
        matches = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matches / len(longer)

# Global instance
_topic_resolver = None

def get_intelligent_topic_resolver() -> IntelligentTopicResolver:
    """Get the global topic resolver instance"""
    global _topic_resolver
    if _topic_resolver is None:
        _topic_resolver = IntelligentTopicResolver()
    return _topic_resolver
