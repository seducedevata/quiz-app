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

    def verify_and_enhance_prompt(self, prompt: str, topic: str, difficulty: str, question_type: str) -> Dict[str, any]:
        """
        🔍 STAGE 2 PHI VERIFICATION: Review and enhance the Inquisitor's Mandate prompt
        
        This method takes the generated prompt and uses Phi model to:
        1. Verify the prompt quality and clarity
        2. Suggest improvements for better question generation
        3. Enhance domain-specific instructions
        4. Optimize for the target difficulty and question type
        
        Args:
            prompt: The Inquisitor's Mandate prompt to verify
            topic: The topic for context
            difficulty: Target difficulty level
            question_type: Target question type
            
        Returns:
            Dict with verification results and enhanced prompt
        """
        try:
            logger.info(f"🔍 PHI VERIFICATION: Reviewing prompt for '{topic}' ({difficulty}, {question_type})")
            
            # Create verification prompt for Phi model - SIMPLIFIED for better JSON reliability
            verification_prompt = f"""You are an expert educational prompt engineer. Analyze and enhance the following quiz generation prompt for maximum educational value.

Topic: {topic}
Difficulty: {difficulty}
Question Type: {question_type}

PROMPT TO ANALYZE:
{prompt[:1000]}...

INSTRUCTIONS:
1. Rate the prompt quality (1-10 scale)
2. Identify strengths and areas for improvement  
3. Create an enhanced version that is more specific, engaging, and pedagogically sound
4. Provide your analysis in STRICT JSON format

CRITICAL: Your response must be ONLY valid JSON. No explanations before or after. Start with {{ and end with }}.

REQUIRED JSON FORMAT:
{{
  "quality_score": 8.5,
  "strengths": ["specific strength 1", "specific strength 2"],
  "improvements": ["specific improvement 1", "specific improvement 2"],
  "enhanced_prompt": "Your enhanced prompt text here...",
  "confidence": 0.85,
  "reasoning": "Brief explanation of changes made"
}}"""

            # Call Phi model for verification using a dedicated method
            verification_result = self._verify_prompt_with_ai(verification_prompt)
            
            if verification_result and verification_result.get("enhanced_prompt"):
                logger.info(f"✅ PHI VERIFICATION: Enhanced prompt (quality score: {verification_result.get('quality_score', 'N/A')})")
                return {
                    "verification_successful": True,
                    "original_prompt": prompt,
                    "enhanced_prompt": verification_result["enhanced_prompt"],
                    "quality_score": verification_result.get("quality_score", 7),
                    "strengths": verification_result.get("strengths", []),
                    "improvements": verification_result.get("improvements", []),
                    "confidence": verification_result.get("confidence", 0.8),
                    "reasoning": verification_result.get("reasoning", "Prompt enhanced by Phi model"),
                    "verification_method": "phi_ai"
                }
            else:
                logger.warning("⚠️ PHI VERIFICATION: AI verification failed, using fallback analysis")
                return self._fallback_prompt_verification(prompt, topic, difficulty, question_type)
                
        except Exception as e:
            logger.error(f"❌ PHI VERIFICATION: Error during prompt verification: {e}")
            return self._fallback_prompt_verification(prompt, topic, difficulty, question_type)

    def _verify_prompt_with_ai(self, verification_prompt: str) -> Dict[str, any]:
        """
        Dedicated method for prompt verification using Phi model
        """
        try:
            # Call Ollama API for prompt verification
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": verification_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temperature for consistent verification
                        "num_predict": 1000,  # Longer response for detailed verification
                        "top_p": 0.9
                    }
                },
                timeout=60  # Longer timeout for verification
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Try to parse JSON response with multiple robust fallback strategies
                if response_text:
                    return self._parse_json_with_advanced_fallbacks(response_text)
                else:
                    logger.warning("⚠️ PHI VERIFICATION: Empty response from Phi model")
                    return None
            else:
                logger.error(f"❌ PHI VERIFICATION: API call failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ PHI VERIFICATION: Error calling Phi model: {e}")
            return None

    def _parse_json_with_advanced_fallbacks(self, response_text: str) -> Dict[str, any]:
        """
        Advanced JSON parsing with multiple robust fallback strategies
        Based on best practices for handling malformed JSON from AI models
        """
        import re
        
        # Strategy 1: Direct parsing with preprocessing
        try:
            # Clean the response text first
            cleaned_text = response_text.strip()
            
            # Remove any leading/trailing non-JSON content
            if not cleaned_text.startswith('{') and '{' in cleaned_text:
                cleaned_text = cleaned_text[cleaned_text.find('{'):]
            if not cleaned_text.endswith('}') and '}' in cleaned_text:
                cleaned_text = cleaned_text[:cleaned_text.rfind('}') + 1]
            
            verification_data = json.loads(cleaned_text)
            logger.info(f"✅ PHI VERIFICATION: Successfully parsed (direct)")
            return verification_data
            
        except json.JSONDecodeError as e:
            logger.debug(f"🔧 Direct parsing failed: {e}")
            
        # Strategy 2: Extract JSON using multiple methods
        json_candidates = []
        
        # Method A: Find complete JSON objects
        brace_count = 0
        start_pos = -1
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos >= 0:
                    json_candidates.append(response_text[start_pos:i+1])
        
        # Method B: Regex patterns for JSON extraction
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\{.*?\}',  # Simple objects
            r'```json\s*(\{.*?\})\s*```',  # Markdown code blocks
            r'```\s*(\{.*?\})\s*```',  # Generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            json_candidates.extend(matches)
        
        # Strategy 3: Try parsing each candidate with progressive fixing
        for candidate in json_candidates:
            if not candidate.strip():
                continue
                
            # Try direct parsing first
            try:
                verification_data = json.loads(candidate)
                logger.info(f"✅ PHI VERIFICATION: Successfully parsed (extracted)")
                return verification_data
            except json.JSONDecodeError:
                pass
            
            # Apply JSON fixes progressively
            fixed_candidate = self._fix_json_progressively(candidate)
            if fixed_candidate:
                try:
                    verification_data = json.loads(fixed_candidate)
                    logger.info(f"✅ PHI VERIFICATION: Successfully parsed (fixed)")
                    return verification_data
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Last resort - manual JSON construction
        logger.warning("⚠️ PHI VERIFICATION: Attempting manual JSON reconstruction")
        manual_json = self._reconstruct_json_manually(response_text)
        if manual_json:
            try:
                verification_data = json.loads(manual_json)
                logger.info(f"✅ PHI VERIFICATION: Successfully parsed (reconstructed)")
                return verification_data
            except json.JSONDecodeError:
                pass
        
        logger.error("❌ PHI VERIFICATION: All advanced JSON parsing strategies failed")
        logger.debug(f"🔍 Raw response text: {response_text[:500]}...")
        return None
    
    def _fix_json_progressively(self, json_text: str) -> str:
        """
        Apply progressive JSON fixes based on common AI model JSON errors
        """
        import re
        
        fixed = json_text.strip()
        
        # Fix 1: Remove trailing commas (most common issue)
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Fix 2: Handle unescaped quotes in string values - more robust approach
        def fix_unescaped_quotes(text):
            # Split by lines and fix each line that contains string values
            lines = text.split('\n')
            fixed_lines = []
            
            for line in lines:
                # If line contains a key-value pair with string value
                if ':' in line and '"' in line:
                    # Find the key and value parts
                    colon_pos = line.find(':')
                    key_part = line[:colon_pos]
                    value_part = line[colon_pos:]
                    
                    # Fix unescaped quotes in the value part only
                    # Look for patterns like "some "quoted" text"
                    value_part = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\\"\\2\\"\\3"', value_part)
                    
                    fixed_lines.append(key_part + value_part)
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        
        fixed = fix_unescaped_quotes(fixed)
        
        # Fix 3: Handle missing quotes around keys
        fixed = re.sub(r'(\w+)(\s*):', r'"\1"\2:', fixed)
        
        # Fix 4: Fix single quotes to double quotes (be careful with content)
        fixed = re.sub(r"'([^']*)'(\s*):", r'"\1"\2:', fixed)  # Keys only
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # Values only
        
        # Fix 5: Remove extra spaces around colons and commas
        fixed = re.sub(r'\s*:\s*', ': ', fixed)
        fixed = re.sub(r'\s*,\s*', ', ', fixed)
        
        # Fix 6: Handle boolean and null values
        fixed = re.sub(r'\btrue\b', 'true', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\bfalse\b', 'false', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\bnull\b', 'null', fixed, flags=re.IGNORECASE)
        
        # Fix 7: Remove JavaScript-style comments
        fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        # Fix 8: Handle multiple JSON objects - take the first complete one
        if fixed.count('{') > 1:
            brace_count = 0
            for i, char in enumerate(fixed):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        fixed = fixed[:i+1]
                        break
        
        # Fix 9: Clean up any remaining formatting issues
        fixed = re.sub(r',\s*}', '}', fixed)  # Remove trailing commas before }
        fixed = re.sub(r',\s*]', ']', fixed)  # Remove trailing commas before ]
        
        return fixed if fixed != json_text else None
    
    def _reconstruct_json_manually(self, response_text: str) -> str:
        """
        Last resort: manually reconstruct JSON from response text
        This handles cases where the AI model outputs semi-structured text
        """
        import re
        
        # Look for key-value patterns in the text
        patterns = {
            'enhanced_prompt': r'enhanced[^:]*:\s*([^,\n}]+)',
            'quality_score': r'quality[^:]*:\s*(\d+(?:\.\d+)?)',
            'issues_found': r'issues[^:]*:\s*(\[.*?\]|\d+)',
            'suggestions': r'suggestions[^:]*:\s*(\[.*?\]|[^,\n}]+)',
            'is_acceptable': r'acceptable[^:]*:\s*(true|false|yes|no)',
            'strengths': r'strengths[^:]*:\s*(\[.*?\]|[^,\n}]+)',
            'improvements': r'improvements[^:]*:\s*(\[.*?\]|[^,\n}]+)',
            'confidence': r'confidence[^:]*:\s*(\d+(?:\.\d+)?)',
            'reasoning': r'reasoning[^:]*:\s*([^,\n}]+)',
        }
        
        reconstructed = {}
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                value = matches[0].strip()
                
                # Convert to appropriate types
                if key in ['quality_score', 'confidence']:
                    try:
                        reconstructed[key] = float(value)
                    except ValueError:
                        reconstructed[key] = 7.0 if key == 'quality_score' else 0.8
                elif key == 'is_acceptable':
                    reconstructed[key] = value.lower() in ['true', 'yes', '1']
                elif key in ['issues_found', 'suggestions', 'strengths', 'improvements'] and value.startswith('['):
                    try:
                        reconstructed[key] = json.loads(value)
                    except:
                        reconstructed[key] = [value.strip('[]')]
                elif key in ['strengths', 'improvements', 'suggestions']:
                    # Convert string to array if needed
                    if not value.startswith('['):
                        # Split by common delimiters
                        items = re.split(r'[,;]|and\s+', value)
                        reconstructed[key] = [item.strip().strip('"\'') for item in items if item.strip()]
                    else:
                        reconstructed[key] = [value.strip('[]"\'')]
                else:
                    reconstructed[key] = value.strip('"\'')
        
        # Map any mis-named fields to the correct expected names
        field_mappings = {
            'enhanced': 'enhanced_prompt',
            'issues_found': 'improvements',  # Map issues to improvements
            'suggestions': 'improvements',   # Map suggestions to improvements if improvements not found
        }
        
        for old_key, new_key in field_mappings.items():
            if old_key in reconstructed and new_key not in reconstructed:
                reconstructed[new_key] = reconstructed[old_key]
                del reconstructed[old_key]
        
        # Add defaults for missing required fields
        defaults = {
            'enhanced_prompt': response_text[:200] + "...",
            'quality_score': 7.0,
            'strengths': ["Partial analysis recovered"],
            'improvements': ["AI model response was malformed"],
            'confidence': 0.7,
            'reasoning': "Reconstructed from partial response",
            'is_acceptable': True
        }
        
        for key, default_value in defaults.items():
            if key not in reconstructed:
                reconstructed[key] = default_value
        
        try:
            return json.dumps(reconstructed)
        except:
            logger.error("❌ PHI VERIFICATION: Manual JSON reconstruction failed")
            return None

    def _fallback_prompt_verification(self, prompt: str, topic: str, difficulty: str, question_type: str) -> Dict[str, any]:
        """
        Fallback prompt verification using rule-based analysis when AI is unavailable
        """
        try:
            logger.info("🔄 Using fallback rule-based prompt verification")
            
            quality_score = 7  # Default baseline
            strengths = []
            improvements = []
            
            # Rule-based quality assessment
            if "JSON" in prompt.upper():
                strengths.append("Enforces structured JSON output")
                quality_score += 0.5
            
            if difficulty.lower() in prompt.lower():
                strengths.append("Includes difficulty targeting")
                quality_score += 0.5
            
            if any(keyword in prompt.lower() for keyword in ["question", "options", "correct", "explanation"]):
                strengths.append("Includes all required MCQ components")
                quality_score += 0.5
            
            if len(prompt) < 500:
                improvements.append("Consider adding more specific instructions for better guidance")
                quality_score -= 0.5
            
            if question_type == "numerical" and "calculation" not in prompt.lower():
                improvements.append("Add stronger numerical/calculation requirements")
                quality_score -= 0.3
            
            # Create enhanced prompt with rule-based improvements
            enhanced_prompt = prompt
            
            # Add domain-specific enhancements based on topic
            if any(math_term in topic.lower() for math_term in ["math", "physics", "chemistry", "engineering"]):
                if "formula" not in prompt.lower():
                    enhanced_prompt += "\n\nEMPHASIZE: Include relevant formulas, equations, and quantitative analysis where appropriate."
            
            if difficulty == "expert":
                enhanced_prompt += "\n\nEXPERT MODE: Ensure research-level complexity and domain expertise requirements."
            
            return {
                "verification_successful": True,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "quality_score": min(10, max(1, quality_score)),
                "strengths": strengths,
                "improvements": improvements,
                "confidence": 0.7,
                "reasoning": "Rule-based prompt enhancement applied",
                "verification_method": "rule_based"
            }
            
        except Exception as e:
            logger.error(f"❌ Even fallback verification failed: {e}")
            return {
                "verification_successful": False,
                "original_prompt": prompt,
                "enhanced_prompt": prompt,  # Return original if all else fails
                "quality_score": 5,
                "strengths": [],
                "improvements": ["Could not analyze prompt quality"],
                "confidence": 0.3,
                "reasoning": f"Verification failed: {str(e)}",
                "verification_method": "error_fallback"
            }

# Global instance for easy access
_semantic_mapper = None

def get_semantic_mapper() -> IntelligentSemanticMapper:
    """Get global semantic mapper instance"""
    global _semantic_mapper
    if _semantic_mapper is None:
        _semantic_mapper = IntelligentSemanticMapper()
    return _semantic_mapper
