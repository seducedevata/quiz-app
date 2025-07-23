"""
LM Studio MCQ Generator
High-performance MCQ generation using LM Studio's superior JSON-capable models
Specifically optimized for Qwen2.5, Hermes, and other structured output models
"""

import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .mcq_generator import MCQGenerator
from .dynamic_timeout_detector import (
    start_response_monitoring,
    update_response_content,
    check_response_timeout,
    stop_response_monitoring
)

logger = logging.getLogger(__name__)


class LMStudioMCQGenerator(MCQGenerator):
    """
    High-performance MCQ generator using LM Studio's OpenAI-compatible API
    Optimized for structured JSON output with superior models like Qwen2.5 and Hermes
    """
    
    # Class-level cache to prevent repeated model detection across instances
    _active_model_cache = None
    _model_cache_timestamp = 0
    _cache_duration = 30  # Cache active model for 30 seconds
    
    # Class-level connection cache to prevent race conditions
    _connection_cache = None
    _connection_cache_timestamp = 0
    _connection_cache_duration = 10  # Cache connection status for 10 seconds

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # LM Studio API Configuration - Dynamic model detection
        if hasattr(self.config, 'get_value'):
            self.base_url = self.config.get_value('models.lm_studio.url', 'http://127.0.0.1:1234')
            self.generation_config = self.config.get_value('models.lm_studio.generation', {})
        elif isinstance(self.config, dict):
            lm_studio_config = self.config.get('models', {}).get('lm_studio', {})
            self.base_url = lm_studio_config.get('url', 'http://127.0.0.1:1234')
            self.generation_config = lm_studio_config.get('generation', {})
        else:
            self.base_url = 'http://127.0.0.1:1234'
            self.generation_config = {}
        
        logger.info(f"🔧 LM Studio URL: {self.base_url}")
        logger.info("🎯 Dynamic model detection enabled - will use whatever model is loaded")
        
        # API endpoints
        self.models_url = f"{self.base_url}/v1/models"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        # Available models and active model
        self.available_models = []
        self.active_model = None
        
        # HTTP session with optimized connection pooling and retries
        self.session = requests.Session()
        try:
            # Try new parameter name first (urllib3 >= 1.26.0)
            retry_strategy = Retry(
                total=2,  # Reduced retries to prevent timeout cascade
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
                backoff_factor=0.5  # Faster backoff
            )
        except TypeError:
            # Fallback to old parameter name for older urllib3 versions
            retry_strategy = Retry(
                total=2,  # Reduced retries to prevent timeout cascade
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "POST", "OPTIONS"],
                backoff_factor=0.5  # Faster backoff
            )
        
        # Optimized adapter with better connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool size
            pool_maxsize=20,      # Max connections per pool
            pool_block=False      # Don't block when pool is full
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Generation settings optimized for JSON output
        self.generation_params = {
            "temperature": self.generation_config.get("temperature", 0.1),  # Low temp for consistency
            "max_tokens": self.generation_config.get("max_tokens", 800),
            "top_p": self.generation_config.get("top_p", 0.9),
            "frequency_penalty": self.generation_config.get("repetition_penalty", 1.05) - 1.0,  # Convert to OpenAI format
            "presence_penalty": 0.0,
            "stop": self.generation_config.get("stop_sequences", ["\n\n", "###", "---"])
        }
        
        # Statistics
        self.generation_stats = {
            "total_generated": 0,
            "avg_time": 0,
            "active_model": None,
            "json_success_rate": 0
        }

    def initialize(self) -> bool:
        """Initialize the LM Studio MCQ generator with comprehensive diagnostics"""
        try:
            logger.info("🚀 Initializing LM Studio MCQ Generator...")
            logger.info(f"🔗 Connecting to: {self.base_url}")
            
            # Check if LM Studio is running
            if not self._check_connection():
                logger.error("❌ Cannot connect to LM Studio server")
                logger.error("💡 Make sure LM Studio is running and accessible on port 1234")
                return False
            
            logger.info("✅ Successfully connected to LM Studio")
            
            # Get available models
            self.available_models = self._get_available_models()
            if not self.available_models:
                logger.error("❌ No models available in LM Studio")
                logger.error("💡 Load a model in LM Studio and ensure it's ready")
                return False
            
            logger.info(f"📋 Found {len(self.available_models)} models: {', '.join(self.available_models)}")
            
            # Select best model (with running model detection)
            self.active_model = self._select_best_model()
            if not self.active_model:
                logger.error("❌ No suitable models found or running")
                logger.error("💡 Ensure a model is loaded and running in LM Studio")
                return False
            
            # Test the selected model
            logger.info(f"🧪 Testing selected model: {self.active_model}")
            if not self._test_model_availability(self.active_model):
                logger.warning(f"⚠️ Selected model {self.active_model} is not responding properly")
                logger.warning("💡 The model might be loading. Try again in a moment.")
            
            logger.info(f"✅ LM Studio initialized successfully!")
            logger.info(f"🎯 Active model: {self.active_model}")
            logger.info(f"📊 Generation settings: temp={self.generation_params.get('temperature', 0.1)}, max_tokens={self.generation_params.get('max_tokens', 800)}")
            
            self.is_initialized = True
            self.generation_stats["active_model"] = self.active_model
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio generator: {e}")
            import traceback
            logger.debug(f"🔍 Full error traceback: {traceback.format_exc()}")
            return False

    def _check_connection(self) -> bool:
        """Check if LM Studio server is accessible with caching"""
        current_time = time.time()
        
        # Check cache first to avoid excessive connection tests
        if (LMStudioMCQGenerator._connection_cache is not None and 
            current_time - LMStudioMCQGenerator._connection_cache_timestamp < LMStudioMCQGenerator._connection_cache_duration):
            logger.debug(f"🚀 Using cached connection status: {LMStudioMCQGenerator._connection_cache}")
            return LMStudioMCQGenerator._connection_cache
        
        try:
            response = self.session.get(self.models_url, timeout=0.5)  # REDUCED FROM 3 to 0.5 seconds for much faster fallback
            is_connected = response.status_code == 200
            
            # Cache the result
            LMStudioMCQGenerator._connection_cache = is_connected
            LMStudioMCQGenerator._connection_cache_timestamp = current_time
            
            if is_connected:
                logger.debug("✅ LM Studio connection confirmed")
            else:
                logger.warning(f"⚠️ LM Studio connection failed with status: {response.status_code}")
            
            return is_connected
        except Exception as e:
            logger.debug(f"❌ LM Studio connection failed: {e}")
            # Cache the failure for a shorter duration to retry sooner
            LMStudioMCQGenerator._connection_cache = False
            LMStudioMCQGenerator._connection_cache_timestamp = current_time
            return False

    def _get_available_models(self) -> List[str]:
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(self.models_url, timeout=2)  # REDUCED FROM 10 to 2 seconds
            response.raise_for_status()
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            # Filter out embedding models
            text_models = [m for m in models if not any(
                keyword in m.lower() for keyword in ['embed', 'embedding', 'retrieval']
            )]
            
            logger.info(f"📋 Found {len(text_models)} text generation models in LM Studio")
            return text_models
            
        except Exception as e:
            logger.error(f"❌ Failed to get models from LM Studio: {e}")
            return []

    def _select_best_model(self) -> Optional[str]:
        """Dynamically detect and select the currently active/loaded model with caching"""
        
        # Check cache first to avoid redundant model testing
        current_time = time.time()
        if (LMStudioMCQGenerator._active_model_cache and 
            current_time - LMStudioMCQGenerator._model_cache_timestamp < LMStudioMCQGenerator._cache_duration):
            logger.info(f"🚀 Using cached active model: {LMStudioMCQGenerator._active_model_cache}")
            return LMStudioMCQGenerator._active_model_cache
        
        logger.info("🔍 Auto-detecting currently active model in LM Studio...")
        logger.info(f"📋 Scanning {len(self.available_models)} available models...")
        
        # Test each model to find the one that's actually loaded/responding
        active_models = []
        for i, model in enumerate(self.available_models, 1):
            logger.info(f"  🧪 Testing model {i}/{len(self.available_models)}: {model}")
            
            if self._test_model_availability(model):
                active_models.append(model)
                logger.info(f"  ✅ Model {model} is ACTIVE and responding!")
                # Cache the first active model found for efficiency
                LMStudioMCQGenerator._active_model_cache = model
                LMStudioMCQGenerator._model_cache_timestamp = current_time
                break  # Exit early once we find an active model
            else:
                logger.debug(f"  💤 Model {model} is not active/loaded")
        
        # Return the first active model found
        if active_models:
            chosen_model = active_models[0]
            logger.info(f"🎯 DYNAMIC SELECTION: Using active model '{chosen_model}'")
            return chosen_model
        
        # If no models are responding, return first available as fallback
        if self.available_models:
            fallback_model = self.available_models[0]
            logger.warning(f"⚠️ No models responding - trying fallback: {fallback_model}")
            logger.warning("💡 Make sure a model is loaded and ready in LM Studio")
            # Don't cache fallback models
            return fallback_model
        
        logger.error("❌ No models available at all")
        return None
    
    def _test_model_availability(self, model_name: str) -> bool:
        """Test if a specific model is actually loaded and responding quickly"""
        try:
            # Use minimal payload for fastest possible test
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,          # Minimal response
                "temperature": 0.0,       # Deterministic
                "stream": False           # No streaming
            }
            
            # Quick test with very short timeout to prevent connection overload
            response = self.session.post(
                self.chat_url,
                json=test_payload,
                timeout=3,  # Reduced timeout to prevent connection overload
                headers={"Content-Type": "application/json"}
            )
            
            # Check response validity
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Verify it has expected structure
                    if "choices" in data and len(data["choices"]) > 0:
                        logger.debug(f"✅ Model {model_name} is loaded and responding")
                        return True
                    else:
                        logger.debug(f"⚠️ Model {model_name} gave invalid response structure")
                        return False
                except json.JSONDecodeError:
                    logger.debug(f"⚠️ Model {model_name} gave non-JSON response")
                    return False
            else:
                logger.debug(f"⚠️ Model {model_name} returned HTTP {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logger.debug(f"⏱️ Model {model_name} response timeout (likely not loaded)")
            return False
        except requests.exceptions.RequestException as e:
            logger.debug(f"🔌 Model {model_name} connection error: {e}")
            return False
        except Exception as e:
            logger.debug(f"❌ Model {model_name} test error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if LM Studio generation is available with optimized caching"""
        # Quick checks first (no network calls)
        if not self.is_initialized:
            logger.debug("❌ LM Studio not initialized")
            return False
            
        if self.active_model is None:
            logger.debug("❌ LM Studio no active model")
            return False
        
        # Network check with caching (only when needed)
        connection_ok = self._check_connection()
        if not connection_ok:
            logger.debug("❌ LM Studio connection check failed")
            return False
            
        logger.debug("✅ LM Studio fully available")
        return True

    def generate_mcq(self, topic: str, context: str = "", num_questions: int = 1, difficulty: str = "medium", game_mode: str = "casual") -> List[Dict[str, Any]]:
        """Generate MCQ questions using LM Studio"""
        if not self.is_available():
            logger.error("❌ LM Studio generator not available")
            return []
        
        results = []
        start_time = time.time()
        successful_generations = 0
        
        logger.info(f"🧠 Generating {num_questions} MCQ(s) about '{topic}' using {self.active_model}")
        
        for i in range(num_questions):
            try:
                question = self._generate_single_question(topic, context, i, difficulty, game_mode)
                if question:
                    results.append(question)
                    successful_generations += 1
                    logger.info(f"✅ Question {i+1}/{num_questions} generated successfully")
                else:
                    logger.warning(f"⚠️ Question {i+1}/{num_questions} generation failed")
            except Exception as e:
                logger.error(f"❌ Question {i+1}/{num_questions} generation error: {e}")
        
        # Update statistics
        total_time = time.time() - start_time
        self.generation_stats["total_generated"] += successful_generations
        self.generation_stats["avg_time"] = total_time / max(successful_generations, 1)
        self.generation_stats["json_success_rate"] = successful_generations / num_questions if num_questions > 0 else 0
        
        logger.info(f"🏁 Generated {successful_generations}/{num_questions} questions in {total_time:.1f}s")
        logger.info(f"📊 JSON success rate: {self.generation_stats['json_success_rate']:.1%}")
        
        return results

    def _generate_single_question(self, topic: str, context: str, question_index: int, difficulty: str = "medium", game_mode: str = "casual") -> Optional[Dict[str, Any]]:
        """Generate a single MCQ question using LM Studio with enhanced error handling"""
        
        prompt = self._create_structured_prompt(topic, context, question_index, difficulty, game_mode)
        
        # Enhanced payload for LM Studio OpenAI-compatible API
        payload = {
            "model": self.active_model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert MCQ generator that responds with valid JSON. Always format your response as a proper JSON object."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistency
            "max_tokens": 400,   # Reduced for faster generation
            "top_p": 0.95,
            "stream": False,     # Ensure non-streaming response
            "stop": ["\n\n", "---", "###"]  # Stop tokens to prevent over-generation
        }
        
        # Remove response_format for compatibility (not all models support it)
        logger.debug(f"🧠 Generating question with model: {self.active_model}")
        
        try:
            # Enhanced request with better error handling
            response = self.session.post(
                self.chat_url, 
                json=payload, 
                timeout=30,  # Reasonable timeout for generation
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            # Log response details for debugging
            logger.debug(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"❌ API returned status {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            
            # Validate response structure
            if "choices" not in data or len(data["choices"]) == 0:
                logger.error(f"❌ Invalid response structure: {data}")
                return None
            
            content = data["choices"][0]["message"]["content"]
            logger.debug(f"🤖 Model response: {content[:200]}...")
            
            # Parse and validate JSON response
            result = self._parse_json_response(content)
            if result:
                logger.info(f"✅ Successfully generated MCQ for topic: {topic}")
                return result
            else:
                logger.warning(f"⚠️ Failed to parse MCQ response for topic: {topic}")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LM Studio API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse API response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in API call: {e}")
            return None

    def _create_structured_prompt(self, topic: str, context: str, question_index: int, difficulty: str = "medium", game_mode: str = "casual") -> str:
        """Create optimized prompt for structured JSON output with proper difficulty control and game mode adaptation"""
        
        # Different question types based on game mode
        if game_mode == "serious":
            question_types = [
                "analytical reasoning",
                "problem-solving",
                "theoretical analysis", 
                "critical evaluation",
                "complex synthesis",
                "advanced application",
                "mathematical derivation",
                "systematic comparison"
            ]
        else:  # casual mode
            question_types = [
                "conceptual understanding",
                "practical application", 
                "fundamental principles",
                "real-world scenarios",
                "cause-and-effect relationships",
                "basic analysis",
                "everyday examples",
                "visual concepts"
            ]
        
        focus = question_types[question_index % len(question_types)]
        
        # Define difficulty-specific requirements
        difficulty_requirements = {
            "easy": {
                "level": "basic",
                "description": "fundamental concepts, simple recall, basic definitions",
                "examples": "simple definitions, basic facts, elementary concepts"
            },
            "medium": {
                "level": "intermediate", 
                "description": "understanding relationships, applying concepts, moderate analysis",
                "examples": "connecting ideas, practical applications, cause-and-effect"
            },
            "hard": {
                "level": "advanced",
                "description": "complex analysis, synthesis, evaluation, expert-level reasoning, specific mechanisms and pathways",
                "examples": "multi-step problem solving, critical evaluation, advanced synthesis, edge cases, molecular processes"
            }
        }
        
        diff_config = difficulty_requirements.get(difficulty.lower(), difficulty_requirements["medium"])
        
        # AGGRESSIVE anti-vague question enforcement for hard mode
        if difficulty.lower() == "hard":
            anti_vague_section = """
🚨 HARD MODE - ZERO TOLERANCE FOR VAGUE QUESTIONS:
❌ ABSOLUTELY FORBIDDEN: "What is the primary function of..."
❌ ABSOLUTELY FORBIDDEN: "What is the main purpose of..."
❌ ABSOLUTELY FORBIDDEN: "What does X do?"
❌ ABSOLUTELY FORBIDDEN: Basic definition questions
❌ ABSOLUTELY FORBIDDEN: General overview questions
✅ MANDATORY: Specific mechanisms, pathways, processes
✅ MANDATORY: Multi-step reasoning and analysis
✅ MANDATORY: Expert-level detail and precision
✅ MANDATORY: Questions requiring deep understanding

BIOLOGY/REPRODUCTIVE SYSTEM HARD MODE RULES:
If topic relates to reproduction/reproductive systems:
- Ask about SPECIFIC hormonal pathways (FSH, LH, testosterone, estrogen regulation)
- Focus on MOLECULAR mechanisms (meiosis stages, fertilization processes)
- Include BIOCHEMICAL processes (steroidogenesis, gametogenesis)
- Test understanding of REGULATORY feedback loops
- Ask about SPECIFIC anatomical structures and their precise functions
- Include PATHOLOGICAL conditions and their molecular basis

EXAMPLES OF BANNED vs REQUIRED QUESTIONS:
❌ BANNED: "What is the primary function of the male/female reproductive system?"
✅ REQUIRED: "During which phase of spermatogenesis do primary spermatocytes undergo the first meiotic division?"

❌ BANNED: "What is the main purpose of the female reproductive system?"
✅ REQUIRED: "Which hormone surge triggers ovulation and what cellular mechanism initiates this process?"
"""
        else:
            anti_vague_section = ""
        
        # Get topic-specific guidance based on the actual topic
        topic_guidance = self._get_topic_specific_guidance(topic, difficulty)
        
        # Enhanced prompt structure with comprehensive anti-vague enforcement
        domain_keywords = {
            "physics": "force, energy, momentum, wave, particle, field, quantum, electromagnetic",
            "chemistry": "molecule, atom, bond, reaction, compound, solution, acid, base",
            "mathematics": "equation, function, derivative, integral, matrix, variable, theorem, proof"
        }

        # Determine domain keywords
        topic_lower = topic.lower()
        required_keywords = ""
        if any(term in topic_lower for term in ["physics", "mechanics", "quantum", "electromagnetic"]):
            required_keywords = domain_keywords["physics"]
        elif any(term in topic_lower for term in ["chemistry", "chemical", "organic", "inorganic"]):
            required_keywords = domain_keywords["chemistry"]
        elif any(term in topic_lower for term in ["mathematics", "math", "calculus", "algebra"]):
            required_keywords = domain_keywords["mathematics"]

        # Length requirements based on difficulty
        min_length = 120 if difficulty.lower() == "expert" else 80 if difficulty.lower() in ["hard", "medium"] else 50

        prompt = f"""You are an expert educational content creator. Generate a {diff_config['level']} difficulty multiple choice question about "{topic}".

MANDATORY REQUIREMENTS:
- Question must be at least {min_length} characters long
- Question MUST end with a question mark (?)
- Include domain-specific terms: {required_keywords}
- All options must be substantive and non-empty
- Expert questions must demonstrate cutting-edge understanding

{anti_vague_section}

REQUIREMENTS:
- Topic: {topic}
- Difficulty: {difficulty.upper()} ({diff_config['description']})
- Focus: {focus}
- Context: {context if context else "Use educational knowledge"}
- Game Mode: {game_mode.upper()}

{topic_guidance}

QUALITY STANDARDS:
✅ Questions must require {diff_config['description']}
✅ Use specific, technical terminology appropriate for {difficulty} level
✅ Avoid vague generalizations - be precise and specific
✅ Include challenging but fair distractors
✅ Focus on understanding mechanisms rather than simple recall
✅ Examples: {diff_config['examples']}

Response format - return ONLY valid JSON:
{{
  "question": "Your specific {difficulty} level question about {topic}",
  "options": {{
    "A": "Specific option A",
    "B": "Specific option B", 
    "C": "Specific option C",
    "D": "Specific option D"
  }},
  "correct": "A",
  "explanation": "Detailed explanation appropriate for {difficulty} level focusing on specific mechanisms"
}}"""

        return prompt

    def _get_topic_specific_guidance(self, topic: str, difficulty: str = "medium") -> str:
        """Get specific guidance based on the topic to improve question quality and prevent vague questions"""
        
        # Convert topic to lowercase for matching
        topic_lower = topic.lower()
        
        # Biology/Health topics - ENHANCED for reproductive system questions
        if any(word in topic_lower for word in ['biology', 'cell', 'dna', 'protein', 'anatomy', 'physiology', 'health', 'medical', 'reproduction', 'reproductive', 'sex', 'sexual', 'hormone', 'sperm', 'egg', 'ovary', 'testes']):
            if difficulty.lower() == "hard":
                return """
🧬 BIOLOGY/REPRODUCTIVE SYSTEM - HARD MODE GUIDANCE:
🚨 ABSOLUTELY BANNED QUESTIONS:
- "What is the primary function of the male/female reproductive system?"
- "What is the main purpose of..."
- "What does [organ] do?"
- Any basic definition or overview questions

✅ HARD MODE REQUIREMENTS:
- Ask about SPECIFIC hormonal pathways (FSH, LH, testosterone, estrogen cycles)
- Focus on MOLECULAR mechanisms (meiosis, fertilization, implantation details)
- Include BIOCHEMICAL processes (steroidogenesis, gametogenesis)
- Test understanding of REGULATORY feedback loops
- Ask about SPECIFIC anatomical structures and their precise functions
- Include PATHOLOGICAL conditions and their mechanisms

EXAMPLE HARD QUESTIONS:
✅ "Which phase of the menstrual cycle is characterized by peak LH levels and what triggers this surge?"
✅ "During spermatogenesis, at which stage do primary spermatocytes undergo the first meiotic division?"
✅ "What role does inhibin B play in the negative feedback regulation of FSH?"
✅ "Which cellular mechanism prevents polyspermy during fertilization?"

Focus on: hormone regulation, cellular processes, molecular biology, specific anatomical functions, pathophysiology
"""
            else:
                return """
BIOLOGY/HEALTH TOPIC GUIDANCE:
- Focus on biological processes, mechanisms, and structures
- Include questions about cellular functions, molecular biology, organ systems  
- Ask about specific pathways, reactions, or anatomical relationships
- Use proper scientific terminology
- Questions should test understanding of HOW and WHY, not just WHAT
- Example: Instead of "What do kidneys do?" ask "Which process in the nephron is primarily responsible for concentrating urine?"
"""
        
        # Science topics
        elif any(word in topic_lower for word in ['physics', 'chemistry', 'science', 'scientific', 'formula', 'equation', 'element', 'atom', 'molecule']):
            if difficulty.lower() == "hard":
                return """
🔬 SCIENCE - HARD MODE GUIDANCE:
- Focus on COMPLEX scientific principles and advanced applications
- Include multi-step calculations and problem-solving scenarios
- Ask about THEORETICAL frameworks and their limitations
- Test understanding of ADVANCED relationships between variables
- Include questions about EXPERIMENTAL design and analysis
- Focus on QUANTITATIVE analysis and precise measurements
- Example: "Given a collision where momentum is conserved but kinetic energy decreases by 40%, what type of collision occurred and what factors determine the energy loss?"
"""
            else:
                return """
SCIENCE TOPIC GUIDANCE:
- Focus on scientific principles, laws, and applications
- Include calculation-based or problem-solving questions
- Ask about relationships between variables and concepts
- Use specific scientific units and measurements
- Test understanding of cause-and-effect relationships
- Example: Instead of "What is gravity?" ask "If an object's mass doubles while distance remains constant, how does gravitational force change?"
"""
        
        # Default guidance for other topics
        elif difficulty.lower() == "hard":
            return f"""
🎯 HARD MODE GUIDANCE FOR {topic.upper()}:
- Avoid simple "What is..." questions
- Focus on complex analysis and synthesis
- Include multi-step reasoning requirements
- Ask about specific mechanisms and processes
- Test advanced understanding and application
- Use precise, technical terminology
- Example: Instead of "What is {topic}?" ask "What specific mechanism in {topic} explains [complex scenario]?"
"""
        
        return f"""
TOPIC GUIDANCE FOR {topic.upper()}:
- Focus on specific concepts and applications within {topic}
- Avoid overly general questions
- Include practical examples and scenarios
- Test understanding of relationships and principles
- Use appropriate terminology for the {difficulty} level
"""

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON response from LM Studio"""
        try:
            # Clean the response
            content = content.strip()
            
            # Remove any markdown formatting
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            # Validate required fields
            required_fields = ['question', 'options', 'correct', 'explanation']
            if not all(field in data for field in required_fields):
                logger.error("❌ Missing required fields in response")
                logger.debug(f"Response data: {data}")
                return None
            
            # Validate options format
            options = data['options']
            if not isinstance(options, dict) or len(options) != 4:
                logger.error("❌ Invalid options format")
                return None
            
            # Ensure options have correct keys
            expected_keys = ['A', 'B', 'C', 'D']
            if list(options.keys()) != expected_keys:
                logger.error("❌ Options must have keys A, B, C, D")
                return None
            
            # Validate correct answer
            correct_key = data['correct']
            if correct_key not in options:
                logger.error("❌ Correct answer key not found in options")
                return None
            
            # Convert to standardized format
            return {
                "question": data['question'],
                "options": list(options.values()),
                "correct_answer": options[correct_key],
                "explanation": data['explanation'],
                "correct_index": expected_keys.index(correct_key)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.debug(f"Raw content: {content}")
            return None
        except Exception as e:
            logger.error(f"❌ Response validation failed: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "base_url": self.base_url,
            "active_model": self.active_model,
            "available_models": self.available_models,
            "total_models": len(self.available_models),
            "generation_stats": self.generation_stats,
            "status": "ready" if self.is_available() else "not_available",
            "connection": self._check_connection(),
            "dynamic_selection": True
        }

    async def generate_quiz_async(self, context: str, topic: str, difficulty: str = "medium", game_mode: str = "casual") -> Dict[str, Any]:
        """
        Generate a single MCQ question async - compatible with MCQ Manager
        """
        try:
            import asyncio
            
            logger.info(f"🧠 LM Studio generating MCQ for topic: '{topic}', difficulty: {difficulty}")
            
            # Generate single question using sync method in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.generate_mcq, topic, context, 1, difficulty, game_mode)
            
            if results and len(results) > 0:
                mcq = results[0]
                # Convert to expected format for MCQ Manager
                return {
                    "question": mcq.get("question", ""),
                    "options": {
                        "A": mcq["options"][0] if len(mcq.get("options", [])) > 0 else "",
                        "B": mcq["options"][1] if len(mcq.get("options", [])) > 1 else "",
                        "C": mcq["options"][2] if len(mcq.get("options", [])) > 2 else "",
                        "D": mcq["options"][3] if len(mcq.get("options", [])) > 3 else ""
                    },
                    "correct": ["A", "B", "C", "D"][mcq.get("correct_index", 0)],
                    "explanation": mcq.get("explanation", "")
                }
            else:
                logger.error("❌ LM Studio: No MCQ generated")
                return None
                
        except Exception as e:
            logger.error(f"❌ LM Studio async MCQ generation failed: {e}")
            return None

    async def generate_mcq_async(self, topic: str, context: str = "", num_questions: int = 1, difficulty: str = "medium") -> List[Dict[str, Any]]:
        """
        Async version of MCQ generation for better performance
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_mcq, topic, context, num_questions, difficulty)

    def cleanup(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        logger.info("🧹 LM Studio MCQ generator cleaned up")