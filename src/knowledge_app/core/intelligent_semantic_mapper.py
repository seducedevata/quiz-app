#!/usr/bin/env python3
"""
🧠 Intelligent Semantic Keyword Mapper
Uses a small Ollama model to intelligently map topics/abbreviations to question types
"""

import json
import logging
import re
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

def _sanitize_cache_input(user_input: str) -> str:
    """
    🛡️ SECURITY FIX: Sanitize user input before caching to prevent cache poisoning

    Args:
        user_input: Raw user input that needs sanitization

    Returns:
        Sanitized input safe for caching
    """
    if not user_input or not isinstance(user_input, str):
        return ""

    # Remove dangerous characters and normalize
    sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', user_input.strip())

    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)

    # Limit length to prevent cache bloat
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."
        logger.warning("Input truncated for cache safety")

    return sanitized.lower()

@dataclass
class SemanticMapping:
    """Result of semantic mapping analysis"""
    original_input: str
    expanded_topic: str
    question_type: str  # "numerical", "conceptual", "mixed"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    is_abbreviation: bool
    full_form: Optional[str] = None

class IntelligentSemanticMapper:
    """
    Uses a small, fast Ollama model to intelligently map any input to appropriate question types.
    Handles abbreviations, acronyms, misspellings, and context understanding.
    """
    
    def __init__(self, model_name: str = "phi"):
        """
        Initialize with a small, fast model for semantic mapping

        Args:
            model_name: Small Ollama model for fast semantic analysis (default: phi - super small and fast)
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.cache = {}  # Simple cache for repeated queries

        # 🚀 PERFORMANCE REVOLUTION: Comprehensive keyword database for instant matching
        self._init_keyword_database()

        # Test model availability (only for edge cases now)
        self._test_model_availability()

    def _init_keyword_database(self):
        """🚀 Initialize comprehensive keyword database for instant semantic analysis"""

        # NUMERICAL TOPICS - Topics that typically require calculations, formulas, numbers
        self.numerical_keywords = {
            # Mathematics
            "math", "mathematics", "algebra", "calculus", "geometry", "trigonometry",
            "statistics", "probability", "arithmetic", "equation", "formula", "derivative",
            "integral", "matrix", "vector", "polynomial", "logarithm", "exponential",
            "factorial", "permutation", "combination", "regression", "correlation",

            # Physics - Calculations
            "physics", "mechanics", "thermodynamics", "electromagnetism", "optics",
            "quantum mechanics", "relativity", "kinematics", "dynamics", "energy",
            "force", "momentum", "acceleration", "velocity", "frequency", "wavelength",
            "amplitude", "voltage", "current", "resistance", "capacitance", "inductance",

            # Chemistry - Calculations
            "stoichiometry", "molarity", "molality", "concentration", "ph", "equilibrium",
            "kinetics", "thermochemistry", "electrochemistry", "gas laws", "ideal gas",
            "reaction rate", "activation energy", "enthalpy", "entropy", "gibbs energy",

            # Engineering
            "engineering", "circuit analysis", "signal processing", "control systems",
            "fluid mechanics", "heat transfer", "mass transfer", "structural analysis",

            # Computer Science - Algorithms with complexity
            "algorithm complexity", "big o", "time complexity", "space complexity",
            "sorting algorithms", "graph algorithms", "dynamic programming",
            "computational complexity", "asymptotic analysis",

            # Economics/Finance
            "economics", "finance", "interest rate", "compound interest", "present value",
            "future value", "roi", "npv", "irr", "break even", "cost analysis",
        }

        # CONCEPTUAL TOPICS - Topics that are theory-based, explanatory, descriptive
        self.conceptual_keywords = {
            # Biology - Concepts
            "biology", "anatomy", "physiology", "genetics", "evolution", "ecology",
            "cell biology", "molecular biology", "biochemistry", "microbiology",
            "immunology", "neuroscience", "botany", "zoology", "taxonomy",
            "photosynthesis", "respiration", "metabolism", "homeostasis",

            # Psychology
            "psychology", "cognitive psychology", "behavioral psychology", "social psychology",
            "developmental psychology", "abnormal psychology", "personality", "learning",
            "memory", "perception", "emotion", "motivation", "consciousness",
            "cbt", "cognitive behavioral therapy", "therapy", "counseling",

            # Medicine
            "medicine", "pathology", "pharmacology", "epidemiology", "public health",
            "clinical medicine", "diagnosis", "treatment", "disease", "syndrome",
            "symptoms", "medical ethics", "healthcare", "nursing",

            # History
            "history", "ancient history", "modern history", "world war", "civilization",
            "historical events", "historical figures", "timeline", "era", "period",

            # Literature
            "literature", "poetry", "novel", "drama", "literary analysis", "author",
            "theme", "symbolism", "metaphor", "narrative", "character development",

            # Philosophy
            "philosophy", "ethics", "logic", "metaphysics", "epistemology", "aesthetics",
            "political philosophy", "moral philosophy", "philosophical theory",

            # Social Sciences
            "sociology", "anthropology", "political science", "international relations",
            "social theory", "culture", "society", "social structure", "social change",

            # Law
            "law", "legal theory", "constitutional law", "criminal law", "civil law",
            "contract law", "tort law", "legal ethics", "jurisprudence",
        }

        # MIXED TOPICS - Topics that can be both numerical and conceptual
        self.mixed_keywords = {
            # Computer Science - Can be both theoretical and practical
            "computer science", "programming", "software engineering", "data structures",
            "databases", "machine learning", "artificial intelligence", "ai", "ml",
            "deep learning", "neural networks", "algorithms", "operating systems",
            "computer networks", "cybersecurity", "web development", "mobile development",

            # Chemistry - Can be both calculations and concepts
            "chemistry", "organic chemistry", "inorganic chemistry", "physical chemistry",
            "analytical chemistry", "chemical bonding", "molecular structure",
            "periodic table", "chemical reactions", "atoms", "molecules", "compounds",

            # General Science
            "science", "scientific method", "research methodology", "data analysis",
            "experimental design", "hypothesis testing", "scientific theory",
        }

        # COMMON ABBREVIATIONS - Instant expansion
        self.abbreviation_map = {
            "ai": ("artificial intelligence", "mixed"),
            "ml": ("machine learning", "mixed"),
            "dl": ("deep learning", "mixed"),
            "nlp": ("natural language processing", "mixed"),
            "cv": ("computer vision", "mixed"),
            "cnn": ("convolutional neural network", "mixed"),
            "rnn": ("recurrent neural network", "mixed"),
            "lstm": ("long short-term memory", "mixed"),
            "gpt": ("generative pre-trained transformer", "mixed"),
            "api": ("application programming interface", "mixed"),
            "sql": ("structured query language", "mixed"),
            "html": ("hypertext markup language", "mixed"),
            "css": ("cascading style sheets", "mixed"),
            "js": ("javascript", "mixed"),
            "ui": ("user interface", "mixed"),
            "ux": ("user experience", "conceptual"),
            "cbt": ("cognitive behavioral therapy", "conceptual"),
            "dbt": ("dialectical behavior therapy", "conceptual"),
            "ptsd": ("post-traumatic stress disorder", "conceptual"),
            "adhd": ("attention deficit hyperactivity disorder", "conceptual"),
            "ocd": ("obsessive compulsive disorder", "conceptual"),
            "dfs": ("depth-first search", "mixed"),
            "bfs": ("breadth-first search", "mixed"),
            "bst": ("binary search tree", "mixed"),
            "avl": ("adelson-velsky and landis tree", "mixed"),
            "tcp": ("transmission control protocol", "mixed"),
            "udp": ("user datagram protocol", "mixed"),
            "http": ("hypertext transfer protocol", "mixed"),
            "https": ("hypertext transfer protocol secure", "mixed"),
            "dns": ("domain name system", "mixed"),
            "ip": ("internet protocol", "mixed"),
            "cpu": ("central processing unit", "mixed"),
            "gpu": ("graphics processing unit", "mixed"),
            "ram": ("random access memory", "mixed"),
            "ssd": ("solid state drive", "mixed"),
            "hdd": ("hard disk drive", "mixed"),
            "os": ("operating system", "mixed"),
            "ide": ("integrated development environment", "mixed"),
            "sdk": ("software development kit", "mixed"),
        }

        logger.info(f"🚀 Keyword database initialized: {len(self.numerical_keywords)} numerical, {len(self.conceptual_keywords)} conceptual, {len(self.mixed_keywords)} mixed, {len(self.abbreviation_map)} abbreviations")

    def _test_model_availability(self) -> bool:
        """Test if the semantic mapping model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model_name in models:
                    logger.info(f"✅ Semantic mapper using model: {self.model_name}")
                    return True
                else:
                    # Fallback to any available small model (prefer phi, then others)
                    available = [m for m in models if any(x in m.lower() for x in ['phi', 'llama', 'mistral', 'gemma'])]
                    if available:
                        # Prefer phi models for speed
                        phi_models = [m for m in available if 'phi' in m.lower()]
                        if phi_models:
                            self.model_name = phi_models[0]
                        else:
                            self.model_name = available[0]
                        logger.info(f"🔄 Semantic mapper fallback to: {self.model_name}")
                        return True
            
            logger.warning("⚠️ No suitable model found for semantic mapping - using fallback")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic mapper model test failed: {e}")
            return False
    
    def map_topic_semantically(self, user_input: str) -> SemanticMapping:
        """
        🚀 LIGHTNING-FAST semantic mapping using database lookup + AI fallback

        Args:
            user_input: Any user input (CBT, quantum mechanics, etc.)

        Returns:
            SemanticMapping with intelligent analysis
        """
        logger.info(f"🔍 SEMANTIC MAPPING REQUEST: '{user_input}'")

        if not user_input or not user_input.strip():
            logger.info("❌ Empty input - returning fallback")
            return self._create_fallback_mapping(user_input, "Empty input")

        # 🛡️ SECURITY FIX: Sanitize input before using as cache key to prevent cache poisoning
        cache_key = _sanitize_cache_input(user_input)
        if cache_key in self.cache:
            logger.info(f"🎯 Cache hit for: {user_input}")
            return self.cache[cache_key]

        # 🚀 STEP 1: Try instant database lookup (99% of cases)
        db_result = self._analyze_with_database(user_input)
        if db_result:
            logger.info(f"⚡ Database hit: '{user_input}' → {db_result.question_type} ({db_result.confidence:.2f}) [INSTANT]")
            self.cache[cache_key] = db_result
            return db_result

        # 🚀 STEP 2: Only use AI for true edge cases (1% of cases)
        try:
            logger.info(f"🧠 Database miss - using AI fallback for: '{user_input}'")
            result = self._analyze_with_ai(user_input)

            # 🛡️ SECURITY FIX: Cache the result with sanitized key
            self.cache[cache_key] = result

            logger.info(f"🧠 AI semantic mapping: '{user_input}' → {result.question_type} ({result.confidence:.2f})")
            return result

        except Exception as e:
            logger.error(f"❌ Semantic mapping failed for '{user_input}': {e}")
            return self._create_default_mapping(user_input, str(e))

    def _analyze_with_database(self, user_input: str) -> SemanticMapping:
        """🚀 INSTANT database lookup for semantic analysis"""

        input_lower = user_input.lower().strip()

        # Step 1: Check exact abbreviation matches
        if input_lower in self.abbreviation_map:
            expanded_topic, question_type = self.abbreviation_map[input_lower]
            logger.info(f"⚡ Abbreviation match: '{user_input}' → '{expanded_topic}' ({question_type})")
            return SemanticMapping(
                original_input=user_input,
                expanded_topic=expanded_topic,
                question_type=question_type,
                confidence=0.95,
                reasoning=f"Database abbreviation match",
                is_abbreviation=True,
                full_form=expanded_topic
            )

        # Step 2: Check keyword matches with fuzzy matching
        best_match = None
        best_score = 0
        best_type = None

        # Check all keyword sets
        keyword_sets = [
            (self.numerical_keywords, "numerical"),
            (self.conceptual_keywords, "conceptual"),
            (self.mixed_keywords, "mixed")
        ]

        for keywords, qtype in keyword_sets:
            for keyword in keywords:
                # Exact match
                if keyword == input_lower:
                    return SemanticMapping(
                        original_input=user_input,
                        expanded_topic=keyword,
                        question_type=qtype,
                        confidence=0.95,
                        reasoning=f"Database exact keyword match",
                        is_abbreviation=False
                    )

                # Partial match (contains)
                if keyword in input_lower or input_lower in keyword:
                    score = len(keyword) / max(len(input_lower), len(keyword))
                    if score > best_score:
                        best_score = score
                        best_match = keyword
                        best_type = qtype

        # Return best partial match if good enough
        if best_match and best_score > 0.6:
            return SemanticMapping(
                original_input=user_input,
                expanded_topic=best_match,
                question_type=best_type,
                confidence=0.80 + (best_score * 0.15),  # 0.80-0.95 range
                reasoning=f"Database partial keyword match (score: {best_score:.2f})",
                is_abbreviation=False
            )

        # No database match found
        return None

    def _analyze_with_ai(self, user_input: str) -> SemanticMapping:
        """Use AI model to analyze the input semantically and generate instruction prompts"""

        prompt = f"""ROBUST TOPIC ANALYSIS: "{user_input}"

MISSION: Transform ANY input (clear topics, abbreviations, random words, typos) into actionable MCQ generation instructions.

HANDLING STRATEGIES:
🔤 ABBREVIATIONS: Expand to full form (AI→Artificial Intelligence, CBT→Cognitive Behavioral Therapy)
🔀 RANDOM WORDS: Find educational angle (xyz→algebra variables, random→probability theory)
✏️ TYPOS: Correct and interpret (phisics→physics, mathmatics→mathematics)
🎯 UNCLEAR: Create meaningful educational context from any input

QUESTION TYPES:
- NUMERICAL: Math, calculations, formulas, physics problems, statistics, data analysis
- CONCEPTUAL: Theory, psychology, history, philosophy, explanations, definitions
- MIXED: Both numerical calculations AND conceptual understanding

ROBUST EXAMPLES:
- "CBT" → "Generate questions about cognitive behavioral therapy techniques, treatment protocols, and therapeutic interventions"
- "AI" → "Create questions covering machine learning algorithms, neural networks, and AI applications"
- "xyz" → "Generate algebra problems involving variables x, y, z and coordinate systems"
- "random" → "Create questions about probability theory, random variables, and statistical distributions"
- "phisics" → "Generate physics questions covering mechanics, thermodynamics, and electromagnetic theory"
- "123" → "Create mathematical questions about number theory, sequences, and numerical patterns"
- "dfs" → "Generate computer science questions about depth-first search algorithms and graph traversal"
- "weird input" → "Create critical thinking questions about problem-solving and analytical reasoning"

ALWAYS SUCCEED: Even for nonsensical input, find an educational angle and create useful instructions.

JSON RESPONSE (ALWAYS VALID):
{{
  "expanded_topic": "clear, educational topic name",
  "question_type": "NUMERICAL|CONCEPTUAL|MIXED",
  "confidence": 0.7-0.9,
  "reasoning": "specific generation instructions for MCQ models",
  "is_abbreviation": true/false,
  "full_form": "full expansion or null"
}}"""

        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for consistent analysis
                        "num_predict": 200,  # Short response
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_response = response.json().get('response', '')
                return self._parse_ai_response(user_input, ai_response)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            raise
    
    def _parse_ai_response(self, original_input: str, ai_response: str) -> SemanticMapping:
        """Parse the AI response into a SemanticMapping object"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                return SemanticMapping(
                    original_input=original_input,
                    expanded_topic=data.get('expanded_topic', original_input),
                    question_type=data.get('question_type', 'MIXED').lower(),
                    confidence=float(data.get('confidence', 0.7)),
                    reasoning=data.get('reasoning', 'AI analysis'),
                    is_abbreviation=bool(data.get('is_abbreviation', False)),
                    full_form=data.get('full_form')
                )
            else:
                raise Exception("No JSON found in AI response")
                
        except Exception as e:
            logger.error(f"❌ Failed to parse AI response: {e}")
            # Fallback parsing
            response_lower = ai_response.lower()
            
            if 'numerical' in response_lower:
                question_type = 'numerical'
            elif 'conceptual' in response_lower:
                question_type = 'conceptual'
            else:
                question_type = 'mixed'
            
            return SemanticMapping(
                original_input=original_input,
                expanded_topic=original_input,
                question_type=question_type,
                confidence=0.6,
                reasoning="Fallback parsing",
                is_abbreviation=False
            )
    
    def _create_default_mapping(self, user_input: str, error_reason: str) -> SemanticMapping:
        """Create a ROBUST default mapping when AI analysis fails - handles ANY input gracefully"""

        input_lower = user_input.lower() if user_input else ""

        # ROBUST FALLBACK STRATEGIES

        # 1. Handle common abbreviations manually
        common_abbrevs = {
            'ai': ('artificial intelligence', 'mixed'),
            'ml': ('machine learning', 'mixed'),
            'cbt': ('cognitive behavioral therapy', 'conceptual'),
            'dfs': ('depth-first search algorithms', 'mixed'),
            'bfs': ('breadth-first search algorithms', 'mixed'),
            'cpu': ('computer processor architecture', 'mixed'),
            'gpu': ('graphics processing units', 'mixed'),
            'api': ('application programming interfaces', 'mixed'),
            'sql': ('database query languages', 'mixed'),
            'html': ('web markup languages', 'mixed'),
            'css': ('web styling technologies', 'mixed'),
            'js': ('javascript programming', 'mixed'),
            'ui': ('user interface design', 'conceptual'),
            'ux': ('user experience design', 'conceptual')
        }

        if input_lower in common_abbrevs:
            expanded, qtype = common_abbrevs[input_lower]
            return SemanticMapping(
                original_input=user_input,
                expanded_topic=expanded,
                question_type=qtype,
                confidence=0.8,
                reasoning=f"Generate questions about {expanded} covering key concepts and applications",
                is_abbreviation=True,
                full_form=expanded
            )

        # 2. Handle single characters intelligently
        if len(input_lower) == 1:
            char_mappings = {
                'x': ('algebra and variables', 'numerical'),
                'y': ('coordinate systems', 'numerical'),
                'z': ('complex numbers', 'numerical'),
                'a': ('basic mathematics', 'numerical'),
                'b': ('biology fundamentals', 'conceptual'),
                'c': ('chemistry basics', 'mixed'),
                'd': ('data structures', 'mixed'),
                'e': ('energy and physics', 'numerical'),
                'f': ('functions and calculus', 'numerical'),
                'g': ('geometry', 'numerical'),
                'h': ('history', 'conceptual'),
                'i': ('information technology', 'mixed'),
                'j': ('journalism', 'conceptual'),
                'k': ('kinematics', 'numerical'),
                'l': ('logic', 'mixed'),
                'm': ('mathematics', 'numerical'),
                'n': ('natural sciences', 'mixed'),
                'o': ('optimization', 'numerical'),
                'p': ('physics', 'numerical'),
                'q': ('quantum mechanics', 'numerical'),
                'r': ('research methods', 'conceptual'),
                's': ('statistics', 'numerical'),
                't': ('technology', 'mixed'),
                'u': ('universe and cosmology', 'mixed'),
                'v': ('vectors', 'numerical'),
                'w': ('waves and oscillations', 'numerical')
            }

            if input_lower in char_mappings:
                topic, qtype = char_mappings[input_lower]
                return SemanticMapping(
                    original_input=user_input,
                    expanded_topic=topic,
                    question_type=qtype,
                    confidence=0.7,
                    reasoning=f"Generate questions about {topic} with appropriate difficulty level",
                    is_abbreviation=False
                )

        # 3. Handle numbers
        if input_lower.isdigit():
            return SemanticMapping(
                original_input=user_input,
                expanded_topic="number theory and mathematics",
                question_type='numerical',
                confidence=0.7,
                reasoning=f"Generate mathematical questions involving the number {user_input} and related concepts",
                is_abbreviation=False
            )

        # 4. Smart pattern detection
        if any(indicator in input_lower for indicator in ['math', 'calc', 'physics', 'chem', 'stat', 'data', 'number', 'formula', 'equation']):
            question_type = 'numerical'
            topic = f"mathematical and scientific concepts related to {user_input}"
        elif any(indicator in input_lower for indicator in ['psych', 'history', 'art', 'phil', 'soc', 'theory', 'concept', 'idea']):
            question_type = 'conceptual'
            topic = f"theoretical and conceptual aspects of {user_input}"
        else:
            question_type = 'mixed'
            topic = f"comprehensive study of {user_input}"

        return SemanticMapping(
            original_input=user_input or "",
            expanded_topic=topic,
            question_type=question_type,
            confidence=0.6,
            reasoning=f"Generate educational questions about {topic} covering both theoretical understanding and practical applications",
            is_abbreviation=False
        )
    
    def get_enhanced_topic_profile(self, user_input: str) -> Dict[str, any]:
        """
        Get enhanced topic profile using semantic mapping
        
        Returns:
            Enhanced profile with AI-powered semantic analysis
        """
        mapping = self.map_topic_semantically(user_input)
        
        # Convert to topic analyzer format
        profile = {
            "original_input": mapping.original_input,
            "expanded_topic": mapping.expanded_topic,
            "detected_type": mapping.question_type,
            "confidence": "high" if mapping.confidence > 0.8 else "medium" if mapping.confidence > 0.6 else "low",
            "reasoning": mapping.reasoning,
            "is_abbreviation": mapping.is_abbreviation,
            "full_form": mapping.full_form,
            "semantic_analysis": True,
            
            # Compatibility with existing system
            "is_conceptual_possible": True,  # Always allow conceptual
            "is_numerical_possible": mapping.question_type in ['numerical', 'mixed'],
            "optimal_question_type": mapping.question_type,
            
            # UI recommendations
            "ui_recommendations": {
                "highlight_conceptual": mapping.question_type == "conceptual",
                "highlight_numerical": mapping.question_type == "numerical",
                "highlight_mixed": mapping.question_type == "mixed",
                "disable_numerical": mapping.question_type == "conceptual",
                "show_expansion": mapping.is_abbreviation and mapping.full_form
            }
        }
        
        return profile

# Global instance for easy access
_semantic_mapper = None

def get_semantic_mapper() -> IntelligentSemanticMapper:
    """Get global semantic mapper instance"""
    global _semantic_mapper
    if _semantic_mapper is None:
        _semantic_mapper = IntelligentSemanticMapper()
    return _semantic_mapper
