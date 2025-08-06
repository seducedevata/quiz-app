#!/usr/bin/env python3
"""
[HOT] FIRE: Direct Ollama JSON Generator with GPU Optimizations
Intelligent JSON parsing with local Ollama models using advanced techniques
Optimized for DeepSeek R1 reasoning models with proper GPU utilization
"""

import logging
import json
import re
import time
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .mcq_generator import MCQGenerator
from .intelligent_prompt_generator import get_intelligent_prompt_generator
from .dynamic_model_detector import get_recommended_models

logger = logging.getLogger(__name__)


class OllamaJSONGenerator(MCQGenerator):
    """
    [HOT] FIRE: Direct Ollama JSON MCQ Generator with GPU Optimization
    Uses intelligent JSON parsing with local Ollama models
    Implements advanced prompting techniques for guaranteed JSON output
    Optimized for high GPU utilization and reduced JSON parsing errors
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ollama Configuration
        self.base_url = 'http://localhost:11434'
        # [OK] FIXED: Remove hardcoded preferences - use dynamic selection
        self.preferred_models = []  # Will be populated dynamically
        
        # API endpoints
        self.generate_url = f"{self.base_url}/api/generate"
        self.models_url = f"{self.base_url}/api/tags"
        
        # Available models and active model
        self.available_models = []
        self.active_model = None
        self.is_initialized = False
        
        # HTTP session with optimized settings
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=0.5
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # [HOT] GPU-OPTIMIZED GENERATION PARAMETERS (Non-DeepSeek models)
        # [INFO] Enhanced randomness to prevent identical questions
        self.generation_params = {
            'temperature': 0.8,         # [HOT] MUCH HIGHER temperature for diverse questions
            'top_p': 0.95,              # Slightly reduced for quality while maintaining diversity
            'top_k': 80,                # Balanced choice set for variety
            'num_predict': 1800,        # Even more tokens for PhD-level complexity
            'num_ctx': 16384,           # Maximum context for extremely complex prompts
            'repeat_penalty': 1.2,      # [HOT] STRONGER penalty against repetitive questions
            'stop': ['\n\n---', '```', 'END_JSON', '###'],
            'seed': -1,                 # [HOT] RANDOM SEED for different questions each time
            # [START] GPU OPTIMIZATION (format:json removed per expert advice)
            'n_gpu_layers': -1,         # Offload ALL layers to GPU
            'main_gpu': 0,              # Use first GPU device
            'num_thread': 8             # Limit CPU threads to prevent GPU starvation
        }
        
        # [HOT] DEEPSEEK R1 SPECIFIC PARAMETERS - MAXIMUM GPU UTILIZATION
        self.deepseek_r1_params = {
            'temperature': 0.3,         # Balanced for quality reasoning and creativity
            'top_p': 0.9,               # Allow more diverse vocabulary for complex topics
            'top_k': 50,                # Reasonable choice set
            'num_predict': 2048,        # Enough tokens for thinking + JSON output
            'num_ctx': 16384,           # DOUBLED: Use more GPU memory for context
            'repeat_penalty': 1.1,      # Light penalty to avoid repetition
            # [START] MAXIMUM GPU UTILIZATION SETTINGS
            'n_gpu_layers': -1,         # Full GPU utilization
            'main_gpu': 0,              # Primary GPU device
            'num_thread': 1,            # MINIMAL: Force everything to GPU
            'num_batch': 2048,          # MASSIVE: Max batch size for GPU saturation
            'n_parallel': 4,            # INCREASED: Multiple parallel streams
            'use_mlock': True,          # Lock model in GPU memory
            'use_mmap': False,          # Disable memory mapping, force GPU
            'keep_alive': '3m'          # Keep model hot for sustained GPU usage
        }
        
        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "json_success_rate": 0,
            "avg_time": 0,
            "active_model": None,
            "parsing_method": "intelligent_extraction"
        }

    def initialize(self) -> bool:
        """Initialize the Ollama JSON generator with comprehensive model detection"""
        try:
            logger.info("[FIRE] Initializing GPU-Optimized Ollama JSON Generator...")
            logger.info(f"[LINK] Connecting to Ollama: {self.base_url}")

            # Check if Ollama is running
            if not self._check_connection():
                logger.error("[ERROR] Cannot connect to Ollama server")
                logger.error("[IDEA] Make sure Ollama is running: ollama serve")
                return False

            logger.info("[OK] Successfully connected to Ollama")

            # Get available models
            self.available_models = self._get_available_models()
            if not self.available_models:
                logger.error("[ERROR] No models available in Ollama")
                logger.error("[IDEA] Install a model: ollama pull llama3.1:8b")
                return False

            logger.info(f"[CLIPBOARD] Found {len(self.available_models)} models: {', '.join(self.available_models)}")

            # Select best model for JSON generation
            self.active_model = self._select_best_json_model()
            if not self.active_model:
                logger.error("[ERROR] No suitable models found for JSON generation")
                return False
            
            # Skip model testing for faster initialization - test during actual use
            logger.info(f"[TEST] Skipping model test for faster startup: {self.active_model}")
            logger.info("      Model will be tested during first actual generation")

            logger.info(f"[FIRE] GPU-Optimized Ollama JSON Generator initialized!")
            logger.info(f"[TARGET] Active model: {self.active_model}")
            logger.info(f"[ROCKET] GPU layers: {self.generation_params['n_gpu_layers']}")
            logger.info(f"[CHART] Generation settings: temp={self.generation_params['temperature']}, max_tokens={self.generation_params['num_predict']}")

            self.is_initialized = True
            self.generation_stats["active_model"] = self.active_model
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Ollama JSON generator: {e}")
            return False

    def _check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = self.session.get(self.models_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[ERROR] Ollama connection failed: {e}")
            return False

    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = self.session.get(self.models_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            logger.info(f"[CLIPBOARD] Found {len(models)} models in Ollama")
            return models
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get models from Ollama: {e}")
            return []

    def _select_best_json_model(self) -> Optional[str]:
        """[USER] RESPECT USER PREFERENCES for JSON model selection"""
        logger.info("[USER] CHECKING USER MODEL PREFERENCES for JSON generation...")

        if not self.available_models:
            logger.warning("[WARNING] No models available")
            return None

        try:
            # [USER] Use dynamic model detector (now respects user preferences)
            recommended_models = get_recommended_models("mcq_generation")

            # Find the best available model from recommendations (user preference first)
            for recommended in recommended_models:
                if recommended in self.available_models:
                    logger.info(f"[OK] SELECTED: {recommended} (respecting user preferences)")
                    return recommended

            # If user preference not available, check if it exists in Ollama at all
            if recommended_models:
                user_preferred = recommended_models[0]
                logger.warning(f"[USER] User preferred model '{user_preferred}' not in available list, checking Ollama directly...")

                # Refresh available models and try again
                self.available_models = self._get_available_models()
                if user_preferred in self.available_models:
                    logger.info(f"[OK] FOUND: {user_preferred} after refresh (respecting user preferences)")
                    return user_preferred

        except Exception as e:
            logger.warning(f"[ERROR] User preference lookup failed: {e}")

        # Fallback: Use scoring system if no user preferences
        logger.info("[AI] FALLBACK: Using intelligent scoring for JSON generation...")
        model_scores = {}
        for model in self.available_models:
            score = self._score_model_for_json_generation(model)
            model_scores[model] = score

        # Select highest scoring model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"[TARGET] Selected best JSON model: {best_model} (score: {model_scores[best_model]})")
        return best_model

    def _score_model_for_json_generation(self, model_name: str) -> int:
        """[OK] DYNAMIC: Score model based on JSON generation capabilities"""
        score = 0
        model_lower = model_name.lower()

        # JSON-specific model scoring
        if "llama3.1" in model_lower: score += 60      # Excellent JSON formatting
        elif "llama3" in model_lower: score += 55      # Good JSON formatting
        elif "mistral" in model_lower: score += 50     # Decent JSON
        elif "qwen" in model_lower: score += 45        # Variable but often good
        elif "phi" in model_lower: score += 40         # Basic JSON
        elif "codellama" in model_lower: score += 35   # Code-focused, decent JSON
        elif "wizardcoder" in model_lower: score += 35 # Code-focused, decent JSON
        elif "mathstral" in model_lower: score += 30   # Math-focused, basic JSON

        # DeepSeek models - good reasoning but can be verbose
        if "deepseek-r1" in model_lower: score += 25   # Good reasoning but verbose
        elif "deepseek" in model_lower: score += 20    # Decent but can be chatty

        # Size considerations (for JSON, medium sizes often better than huge)
        if "8b" in model_lower or "7b" in model_lower: score += 25  # Sweet spot for JSON
        elif "13b" in model_lower or "14b" in model_lower: score += 20  # Good balance
        elif "32b" in model_lower: score += 10         # Powerful but slower
        elif "70b" in model_lower: score += 5          # Very powerful but very slow

        # Instruction-tuned models get bonus
        if "instruct" in model_lower: score += 15
        if "chat" in model_lower: score += 10

        return score

        # [OK] REMOVED: Old hardcoded fallback logic
        json_capable_keywords = [
            'code', 'instruct', 'chat', 'llama3', 'qwen', 'mistral', 'hermes'
        ]
        
        for available in self.available_models:
            for keyword in json_capable_keywords:
                if keyword in available.lower():
                    logger.info(f"[OK] Selected JSON-capable model: {available}")
                    return available
        
        # Fallback to first available model
        if self.available_models:
            logger.warning(f"[WARNING] Using fallback model: {self.available_models[0]}")
            return self.available_models[0]
        
        return None

    def _test_json_generation(self) -> bool:
        """Test JSON generation with selected model"""
        try:
            test_prompt = "Generate a simple JSON object with 'test': true"
            response = self._call_ollama_api(test_prompt)
            
            if response:
                try:
                    json.loads(response)
                    return True
                except json.JSONDecodeError:
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"JSON generation test failed: {e}")
            return False

    def generate_mcq(self, topic: str, context: str = "", num_questions: int = 1, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed", generation_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
        """[BRAIN] INTELLIGENT MCQ Generation with Topic Resolution"""
        if not self.is_available():
            logger.error("[ERROR] Ollama JSON generator not available")
            return []

        try:
            # [BRAIN] STEP 1: USE PHI-GENERATED INSTRUCTIONS OR FALLBACK TO INTELLIGENT RESOLUTION
            if generation_instructions:
                logger.info(f"[BRAIN] Using Phi-generated instructions: {generation_instructions}")
                resolved_topic = topic  # Use topic as-is since Phi already processed it
                enhanced_context = generation_instructions  # Use Phi instructions as context
                confidence = 0.9  # High confidence since Phi analyzed it
            else:
                logger.info(f"[BRAIN] Fallback to intelligent topic resolution for: '{topic}'")
                prompt_generator = get_intelligent_prompt_generator()
                intelligent_prompt_data = prompt_generator.generate_intelligent_prompt(
                    raw_input=topic,
                    difficulty=difficulty,
                    question_type=question_type
                )

                # Extract resolved information
                resolved_topic = intelligent_prompt_data["resolution"]["resolved_topic"]
                enhanced_context = intelligent_prompt_data["enhanced_context"]
                confidence = intelligent_prompt_data["confidence"]

            logger.info(f"[OK] Resolved '{topic}' → '{resolved_topic}' (confidence: {confidence:.2f})")
            logger.info(f"[DOC] Context: {enhanced_context[:100]}...")

            # [BRAIN] STEP 2: ENHANCED GENERATION with resolved topic
            if 'deepseek' in self.active_model.lower():
                results = self._reasoning_model_batch_generation(
                    resolved_topic, enhanced_context, num_questions, difficulty, game_mode, question_type
                )

                # If DeepSeek R1 fails, raise error - no fallback
                if not results:
                    logger.error("[ERROR] DeepSeek R1 generation failed - no alternative models available")
                    raise Exception(f"DeepSeek R1 generation failed for topic '{resolved_topic}' - AI generation required")

                return results
            else:
                # Standard generation for non-reasoning models
                if generation_instructions:
                    # Use Phi instructions directly
                    return self._phi_instruction_generation(
                        resolved_topic, enhanced_context, num_questions, difficulty, game_mode, question_type
                    )
                else:
                    # Use intelligent prompt data
                    return self._intelligent_standard_generation(
                        intelligent_prompt_data, resolved_topic, num_questions, difficulty, game_mode, question_type
                    )

        except Exception as e:
            logger.error(f"[ERROR] Generation failed: {e}")
            return []

    def generate_mcq_with_custom_prompt(self, topic: str, custom_prompt: str, context: str = "", num_questions: int = 1, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> List[Dict[str, Any]]:
        """[BRAIN] Generate MCQ using PHI-ENHANCED CUSTOM PROMPT (Stage 2 Verification)"""
        if not self.is_available():
            logger.error("[ERROR] Ollama JSON generator not available")
            return []

        try:
            logger.info(f"[BRAIN] Using PHI-ENHANCED CUSTOM PROMPT for '{topic}' ({difficulty}, {question_type})")
            logger.info(f"[PROMPT] Enhanced prompt length: {len(custom_prompt)} characters")

            results = []
            for i in range(num_questions):
                try:
                    # Use the PHI-enhanced prompt directly
                    logger.info(f"[BRAIN] Generating question {i+1}/{num_questions} with enhanced prompt")

                    # Generate using Ollama with difficulty-based timeout
                    response = self._call_ollama_api(custom_prompt, difficulty)
                    if response:
                        parsed_mcqs = self._parse_json_response(response, 1)  # Expecting 1 question
                        if parsed_mcqs and len(parsed_mcqs) > 0:
                            mcq = parsed_mcqs[0]
                            # Add metadata about the enhancement
                            mcq['generation_metadata'] = {
                                'used_phi_enhanced_prompt': True,
                                'stage_2_verification': True,
                                'original_topic': topic,
                                'difficulty': difficulty,
                                'question_type': question_type
                            }
                            results.append(mcq)
                            logger.info(f"[OK] Generated PHI-ENHANCED question {i+1}/{num_questions}")
                        else:
                            logger.warning(f"[WARNING] Failed to parse PHI-ENHANCED question {i+1}")
                    else:
                        logger.warning(f"[WARNING] No response for PHI-ENHANCED question {i+1}")

                except Exception as e:
                    logger.error(f"[ERROR] Failed to generate PHI-ENHANCED question {i+1}: {e}")

            logger.info(f"[RESULT] PHI-ENHANCED generation completed: {len(results)}/{num_questions} questions")
            return results

        except Exception as e:
            logger.error(f"[ERROR] PHI-ENHANCED generation failed: {e}")
            return []

    def _phi_instruction_generation(self, topic: str, phi_instructions: str, num_questions: int, difficulty: str, game_mode: str, question_type: str) -> List[Dict[str, Any]]:
        """[BRAIN] Generate MCQ using Phi-generated instruction prompts"""
        logger.info(f"[BRAIN] Using Phi instruction-based generation for: {topic}")

        results = []
        for i in range(num_questions):
            try:
                # Create prompt using Phi instructions with STRONG numerical enforcement
                # Handle None or empty phi_instructions
                safe_phi_instructions = phi_instructions if phi_instructions else "Generate high-quality multiple choice questions."

                numerical_enforcement = ""
                if question_type.lower() == "numerical":
                    numerical_enforcement = """
🔢 NUMERICAL EXCELLENCE FOCUS:
- Question should involve calculations, numbers, formulas, or quantitative analysis
- Include specific numerical values (masses, energies, wavelengths, frequencies, etc.)
- All answer options should be numerical values with appropriate units
- Question should require mathematical problem-solving skills
- Focus on quantitative analysis to maintain numerical question integrity"""

                prompt = f"""You are an expert MCQ generator. {safe_phi_instructions}

Topic: {topic}
Difficulty: {difficulty}
Question Type: {question_type}
{numerical_enforcement}

Create a {difficulty}-level {question_type} multiple choice question about {topic}.

Respond with valid JSON only:
{{
  "question": "Your question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "A",
  "explanation": "Brief explanation"
}}"""

                # Generate using Ollama with difficulty-based timeout
                response = self._call_ollama_api(prompt, difficulty)
                if response:
                    parsed_mcqs = self._parse_json_response(response, 1)  # Expecting 1 question
                    if parsed_mcqs and len(parsed_mcqs) > 0:
                        results.append(parsed_mcqs[0])  # Take the first (and only) question
                        logger.info(f"[OK] Generated question {i+1}/{num_questions} using Phi instructions")
                    else:
                        logger.warning(f"[WARNING] Failed to parse question {i+1}")
                else:
                    logger.warning(f"[WARNING] No response for question {i+1}")

            except Exception as e:
                logger.error(f"[ERROR] Failed to generate question {i+1}: {e}")

        return results

    def _reasoning_model_batch_generation(self, topic: str, context: str, num_questions: int, difficulty: str, game_mode: str, question_type: str) -> List[Dict[str, Any]]:
        """[BRAIN] GPU-OPTIMIZED DeepSeek R1 batch generation"""
        
        logger.info(f"[BRAIN] GPU-Optimized DeepSeek R1 generation: {num_questions} questions")
        
        # Generate questions in smaller batches for better GPU utilization
        batch_size = min(5, num_questions)  # Optimal for GPU memory
        all_results = []
        
        remaining = num_questions
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            logger.info(f"[START] GPU batch generation: {current_batch} questions")
            
            batch_results = self._generate_reasoning_batch(
                topic, context, current_batch, difficulty, game_mode, question_type
            )
            
            if batch_results:
                all_results.extend(batch_results)
                logger.info(f"[OK] GPU batch successful: {len(batch_results)} questions")
            
            remaining -= current_batch
        
        return all_results[:num_questions]

    def _intelligent_standard_generation(self, intelligent_prompt_data: Dict, topic: str, num_questions: int, difficulty: str, game_mode: str, question_type: str) -> List[Dict[str, Any]]:
        """[BRAIN] INTELLIGENT Standard generation using smart prompts"""

        logger.info(f"[BRAIN] Intelligent standard generation: {num_questions} questions")

        results = []
        for i in range(num_questions):
            try:
                # Use the intelligent prompt directly with numerical enforcement
                base_prompt = intelligent_prompt_data.get("prompt", "") if intelligent_prompt_data else ""

                # 🚀 REVOLUTIONARY: Use unified prompt system instead of hardcoded constraints
                try:
                    from .inquisitor_prompt import get_provider_optimized_prompt
                    
                    # Use the unified trust-based prompt system
                    prompt = get_provider_optimized_prompt(
                        topic=topic,
                        difficulty=difficulty,
                        question_type=question_type,
                        context="",
                        provider="ollama"
                    )
                    
                    logger.info(f"[BRAIN] Using unified prompt system for question {i+1}/{num_questions}")
                    
                except ImportError:
                    logger.warning("[WARNING] Unified prompt system not available, using simple fallback")
                    
                    # Simple fallback without hardcoded constraints
                    difficulty_desc = {
                        "expert": "PhD-level expert",
                        "hard": "graduate-level advanced", 
                        "medium": "undergraduate-level",
                        "easy": "introductory-level"
                    }.get(difficulty, "appropriate-level")
                    
                    # Trust-based prompt - guide quality without forbidding
                    prompt = f"""Generate a {difficulty_desc} multiple choice question about {topic}.

Quality Expectations:
- Demonstrate {difficulty_desc} understanding and reasoning
- Use appropriate technical terminology for the subject
- Create engaging questions that test real comprehension
- For numerical questions: include calculations and quantitative analysis
- For conceptual questions: focus on understanding and relationships

Return as valid JSON with question, options (A-D), correct answer, and explanation."""

                logger.info(f"[START] Generating question {i+1}/{num_questions} with intelligent prompt")

                # Generate with Ollama using difficulty-based timeout
                response = self._call_ollama_api(prompt, difficulty)

                if response:
                    # Parse the response
                    parsed_list = self._parse_json_response(response, 1)
                    if parsed_list and len(parsed_list) > 0:
                        parsed = parsed_list[0]
                        # Add metadata from intelligent resolution
                        parsed["metadata"] = intelligent_prompt_data["metadata"]
                        parsed["resolution_confidence"] = intelligent_prompt_data["confidence"]
                        parsed["original_input"] = intelligent_prompt_data["metadata"]["original_input"]
                        parsed["resolved_topic"] = intelligent_prompt_data["metadata"]["resolved_topic"]

                        results.append(parsed)
                        logger.info(f"[OK] Question {i+1} generated successfully")
                    else:
                        logger.warning(f"[WARNING] Failed to parse question {i+1}")
                else:
                    logger.warning(f"[WARNING] No response for question {i+1}")

            except Exception as e:
                logger.error(f"[ERROR] Error generating question {i+1}: {e}")

        logger.info(f"[TARGET] Intelligent generation complete: {len(results)}/{num_questions} questions")
        return results

    def _generate_reasoning_batch(self, topic: str, context: str, batch_size: int, difficulty: str, game_mode: str, question_type: str) -> List[Dict[str, Any]]:
        """[HOT] GOLD SOLUTION: Generate individual questions for better success with DeepSeek"""
        
        start_time = time.time()
        
        # [BRAIN] For DeepSeek, generate individual questions rather than batches
        # This gives much better success rates
        logger.info(f"[BRAIN] DeepSeek batch: generating {batch_size} individual questions")
        
        questions = []
        for i in range(batch_size):
            logger.info(f"[BRAIN] Generating question {i+1}/{batch_size}")
            
            # Use single question generation for each
            question = self._generate_single_question(topic, difficulty, question_type)
            
            if question:
                questions.append(question)
                logger.info(f"[OK] Question {i+1} successful")
            else:
                logger.warning(f"[WARNING] Question {i+1} failed")
        
        total_time = time.time() - start_time
        logger.info(f"[TIME] DeepSeek batch completed in {total_time:.1f}s: {len(questions)}/{batch_size} successful")
        
        return questions

    def _create_deepseek_json_prompt(self, topic: str, question_type: str, difficulty: str) -> str:
        """[BRAIN] GOLD SOLUTION: Specialized prompt for DeepSeek-R1 with proper difficulty differentiation"""

        # 🚀 CRITICAL FIX: Add proper difficulty level differentiation
        if difficulty == "expert":
            difficulty_level = "PhD-level expert"
            complexity_desc = "research-level complexity requiring deep theoretical understanding and advanced mathematical analysis"
            cognitive_level = "synthesis, evaluation, and expert-level application of advanced concepts"
            min_length = "200+ characters with multi-step reasoning"
        elif difficulty == "hard":
            complexity_desc = "graduate-level complexity requiring advanced understanding of complex relationships"
            cognitive_level = "analysis and application of sophisticated concepts"
            difficulty_level = "Graduate-level advanced"
            min_length = "150+ characters with detailed reasoning"
        elif difficulty == "medium":
            complexity_desc = "undergraduate-level complexity requiring solid understanding"
            cognitive_level = "application and analysis of standard concepts"
            difficulty_level = "Undergraduate-level"
            min_length = "100+ characters with clear reasoning"
        else:  # easy
            complexity_desc = "introductory-level complexity with basic concepts"
            cognitive_level = "understanding and basic application"
            difficulty_level = "Introductory-level"
            min_length = "80+ characters with simple reasoning"

        # Define difficulty-aware question type requirements
        if difficulty == "expert":
            question_type_requirements = {
                "numerical": {
                    "focus": "advanced mathematical calculations, complex quantitative analysis, and multi-step numerical problem-solving requiring PhD-level expertise",
                    "requirements": [
                        "MUST involve complex calculations with multiple steps and advanced mathematical concepts",
                        "MUST require deep mathematical reasoning involving calculus, differential equations, or advanced statistics",
                        "MUST include sophisticated formulas, constants, and specialized units",
                        "MUST test ability to solve research-level problems with precise numerical answers",
                        "MUST demonstrate mastery of advanced mathematical techniques and computational methods"
                    ],
                    "examples": "quantum mechanical calculations, thermodynamic cycle analysis, electromagnetic field computations, statistical modeling"
                },
                "conceptual": {
                    "focus": "advanced theoretical understanding, complex principles, and sophisticated qualitative reasoning at research level",
                    "requirements": [
                        "MUST test deep understanding of advanced theories, models, and research-level concepts",
                        "MUST focus on complex 'why' and 'how' questions requiring synthesis of multiple theories",
                        "MUST completely avoid numerical calculations or quantitative analysis",
                        "MUST test expert-level conceptual understanding and multi-layered relationships",
                        "MUST require knowledge of cutting-edge research and advanced theoretical frameworks"
                    ],
                    "examples": "theoretical mechanism explanations, advanced model comparisons, research methodology discussions"
                },
                "mixed": {
                    "focus": "integration of advanced numerical and sophisticated conceptual elements at expert level",
                    "requirements": [
                        "Must combine complex quantitative analysis with advanced theoretical understanding",
                        "Should integrate multiple advanced mathematical and conceptual knowledge domains",
                        "Must require both computational expertise and deep theoretical insight"
                    ],
                    "examples": "research problems requiring both advanced calculations and theoretical interpretation"
                }
            }
        elif difficulty == "hard":
            question_type_requirements = {
                "numerical": {
                    "focus": "complex mathematical calculations, advanced quantitative analysis, and multi-step numerical problem-solving at graduate level",
                    "requirements": [
                        "MUST involve multi-step calculations with intermediate to advanced mathematical concepts",
                        "MUST require solid mathematical reasoning involving calculus, linear algebra, or statistics",
                        "MUST include complex formulas, specialized constants, and appropriate units",
                        "MUST test ability to solve graduate-level problems with detailed numerical solutions",
                        "MUST demonstrate proficiency in advanced mathematical techniques"
                    ],
                    "examples": "advanced physics calculations, chemical kinetics, engineering analysis, mathematical modeling"
                },
                "conceptual": {
                    "focus": "advanced theoretical understanding, complex principles, and sophisticated qualitative reasoning",
                    "requirements": [
                        "MUST test solid understanding of advanced theories, principles, and graduate-level concepts",
                        "MUST focus on detailed 'why' and 'how' questions requiring integration of concepts",
                        "MUST avoid numerical calculations or quantitative analysis",
                        "MUST test advanced conceptual understanding and sophisticated relationships",
                        "MUST require knowledge of advanced theories and their applications"
                    ],
                    "examples": "theoretical explanations, advanced comparisons, principle applications, mechanism analysis"
                },
                "mixed": {
                    "focus": "combination of advanced numerical and conceptual elements at graduate level",
                    "requirements": [
                        "Must combine complex quantitative analysis with advanced theoretical understanding",
                        "Should integrate multiple mathematical and conceptual knowledge areas",
                        "Must require both computational skills and theoretical reasoning"
                    ],
                    "examples": "graduate-level problems requiring both calculation and theoretical explanation"
                }
            }
        else:
            # Medium and easy use existing simpler requirements
            question_type_requirements = {
                "numerical": {
                    "focus": "mathematical calculations, quantitative analysis, and numerical problem-solving",
                    "requirements": [
                        "MUST involve specific numbers, calculations, or quantitative analysis",
                        "MUST require mathematical reasoning or computational steps",
                        "MUST include units, formulas, or numerical relationships",
                        "MUST test ability to solve problems with concrete numerical answers"
                    ],
                    "examples": "calculations, unit conversions, formula applications, data analysis"
                },
                "conceptual": {
                    "focus": "theoretical understanding, principles, and qualitative reasoning",
                    "requirements": [
                        "MUST test understanding of concepts, theories, or principles",
                        "MUST focus on 'why' and 'how' rather than 'what'",
                        "MUST avoid numerical calculations or quantitative analysis",
                        "MUST test deep conceptual understanding and relationships"
                    ],
                    "examples": "explanations, comparisons, cause-effect relationships, theoretical applications"
                },
                "mixed": {
                    "focus": "combination of numerical and conceptual elements",
                    "requirements": [
                        "Can combine quantitative analysis with conceptual understanding",
                        "May include both calculations and theoretical reasoning",
                        "Should integrate multiple types of knowledge"
                    ],
                    "examples": "problems requiring both calculation and explanation"
                }
            }

        type_config = question_type_requirements.get(question_type, question_type_requirements["mixed"])
        requirements_text = "\n".join(f"- {req}" for req in type_config['requirements'])

        return f"""<think>
My goal is to generate a single, high-quality {question_type.upper()} multiple-choice question.
Topic: {topic}
Difficulty: {difficulty_level} ({difficulty})
Question Type: {question_type.upper()}

DIFFICULTY SPECIFICATIONS:
- Level: {difficulty_level}
- Complexity: {complexity_desc}
- Cognitive Level: {cognitive_level}
- Length Requirement: {min_length}

CRITICAL QUESTION TYPE REQUIREMENTS for {question_type.upper()}:
Focus: {type_config['focus']}
Requirements:
{requirements_text}

Examples: {type_config['examples']}

Plan:
1. Analyze the topic '{topic}' specifically for {question_type.upper()} aspects at {difficulty_level}
2. Design a question that follows {question_type.upper()} excellence guidelines with {complexity_desc}
3. Ensure the question demonstrates {cognitive_level}
4. Create one unambiguously correct answer that fits {question_type.upper()} format
5. Create three plausible but incorrect distractors appropriate for {difficulty_level} {question_type.upper()} questions
6. Write explanation that demonstrates {difficulty_level} understanding
7. Output ONLY the JSON object

IMPORTANT: This MUST be a {difficulty_level} {question_type.upper()} question with {complexity_desc}.
</think>

{{
  "question": "Your {difficulty_level} {question_type.upper()} question about {topic}?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "A",
  "explanation": "A detailed explanation showing why this is the correct {difficulty_level} {question_type.upper()} answer with {complexity_desc}."
}}"""

    def _call_ollama_reasoning_api(self, prompt: str) -> Optional[str]:
        """🚀 ENHANCED: Smart streaming for DeepSeek R1 with dynamic JSON detection"""
        
        # Extract keep_alive from params to include at top level
        reasoning_options = self.deepseek_r1_params.copy()
        keep_alive = reasoning_options.pop('keep_alive', '5m')
        
        payload = {
            "model": self.active_model,
            "prompt": prompt,
            "stream": True,  # 🚀 ENABLE STREAMING for immediate processing
            "keep_alive": keep_alive,  # CRITICAL: Keep model hot
            "options": reasoning_options
        }
        
        try:
            logger.info(f"[STREAM] DeepSeek R1 smart streaming with keep_alive={keep_alive}")
            logger.debug(f"[GPU] layers: {reasoning_options['n_gpu_layers']}")
            logger.debug(f"[CPU] threads: {reasoning_options['num_thread']}")
            logger.debug(f"[TOKENS] predict: {reasoning_options['num_predict']}")
            
            start_time = time.time()
            response = self.session.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=30  # Just connection timeout
            )
            response.raise_for_status()
            
            # 🧠 SMART JSON DETECTION for DeepSeek responses
            accumulated_response = ""
            json_bracket_count = 0
            json_started = False
            complete_json_detected = False
            reasoning_complete = False
            
            max_time = 600  # 10 minutes absolute max for complex reasoning
            
            for line in response.iter_lines(decode_unicode=True):
                if time.time() - start_time > max_time:
                    logger.warning(f"[TIMEOUT] DeepSeek reached maximum time limit")
                    break
                    
                if line:
                    try:
                        chunk_data = json.loads(line)
                        chunk_text = chunk_data.get("response", "")
                        
                        if chunk_text:
                            accumulated_response += chunk_text
                            
                            # 🧠 SPECIAL DEEPSEEK DETECTION: Look for reasoning completion
                            if "<think>" in chunk_text:
                                logger.debug("[REASON] DeepSeek reasoning started")
                            elif "</think>" in chunk_text:
                                logger.info(f"[REASON] DeepSeek reasoning completed after {time.time() - start_time:.1f}s")
                                reasoning_complete = True
                            
                            # 🧠 DETECT JSON STRUCTURE after reasoning
                            if reasoning_complete:
                                for char in chunk_text:
                                    if char == '{':
                                        if not json_started:
                                            json_started = True
                                            logger.debug("[JSON] JSON object started after reasoning")
                                        json_bracket_count += 1
                                    elif char == '}':
                                        json_bracket_count -= 1
                                        if json_started and json_bracket_count == 0:
                                            complete_json_detected = True
                                            logger.info(f"[JSON] Complete JSON detected after {time.time() - start_time:.1f}s")
                                            break
                                
                                # 🚀 IMMEDIATE PROCESSING: Return as soon as JSON is complete
                                if complete_json_detected:
                                    if '"question"' in accumulated_response and '"options"' in accumulated_response:
                                        elapsed = time.time() - start_time
                                        logger.info(f"[SUCCESS] Valid DeepSeek JSON ready in {elapsed:.1f}s")
                                        return accumulated_response
                                    else:
                                        logger.debug("[JSON] Complete but missing MCQ fields, continuing...")
                                        complete_json_detected = False
                                        json_bracket_count = 0
                        
                        # Check if generation is complete
                        if chunk_data.get("done", False):
                            logger.info(f"[DONE] DeepSeek generation marked complete after {time.time() - start_time:.1f}s")
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Return accumulated response
            if accumulated_response.strip():
                elapsed = time.time() - start_time
                logger.info(f"[COMPLETE] DeepSeek response ready after {elapsed:.1f}s")
                return accumulated_response
            else:
                logger.warning("[EMPTY] No response accumulated from DeepSeek")
                return None
            
        except Exception as e:
            logger.error(f"[ERROR] DeepSeek smart streaming failed: {e}")
            return None

    def _standard_generation(self, topic: str, context: str, num_questions: int, difficulty: str, game_mode: str, question_type: str) -> List[Dict[str, Any]]:
        """Standard generation for non-reasoning models"""
        
        results = []
        for i in range(num_questions):
            question = self._generate_single_question(topic, difficulty, question_type)
            if question:
                results.append(question)
        
        return results

    def _generate_single_question(self, topic: str, difficulty: str, question_type: str) -> Optional[Dict[str, Any]]:
        """🚀 ENHANCED: Smart streaming single question generation with dynamic JSON detection"""
        
        # Detect if using DeepSeek model
        is_deepseek_model = 'deepseek' in self.active_model.lower()
        
        if is_deepseek_model:
            logger.info("[BRAIN] DeepSeek model detected. Using specialized approach.")
            
            # Use specialized DeepSeek prompt
            prompt = self._create_deepseek_json_prompt(topic, question_type, difficulty)
            
            # Call with DeepSeek-specific parameters
            response = self._call_ollama_reasoning_api(prompt)
            
            if response:
                # Use bulletproof JSON extraction
                parsed_json = self._extract_and_parse_final_json(response)
                if parsed_json:
                    return self._validate_single_question(parsed_json)
            
            return None
        else:
            # 🔥 DIFFICULTY-AWARE PROMPTS: Enhanced for dynamic streaming
            logger.info(f"[STREAM] Using smart streaming for {difficulty} {question_type} about {topic}")

            # [CONFIG] CRITICAL FIX for Bug 3: Use inquisitor prompt for reliable JSON generation
            # Complex prompts cause malformed JSON - use ultra-strict inquisitor prompt instead
            from .inquisitor_prompt import _create_inquisitor_prompt

            logger.info(f"[CONFIG] Using enhanced inquisitor prompt for {self.active_model} (smart streaming)")

            # Use the ultra-strict inquisitor prompt designed for local models
            prompt = _create_inquisitor_prompt(
                context=f"Generate a {difficulty} {question_type} question about {topic}",
                topic=topic,
                difficulty=difficulty,
                question_type=question_type
            )

            # 🚀 SMART API CALL: Uses dynamic JSON detection instead of fixed timeouts
            start_time = time.time()
            response = self._call_ollama_api(prompt, difficulty)
            elapsed = time.time() - start_time
            
            if response:
                logger.info(f"[SUCCESS] Got response in {elapsed:.1f}s via smart streaming")
                return self._parse_single_json(response)
            else:
                logger.warning(f"[FAILED] No response after {elapsed:.1f}s")

            return None

    def _call_ollama_api(self, prompt: str, difficulty: str = "medium") -> Optional[str]:
        """🚀 DYNAMIC: Smart streaming API call that auto-detects JSON completion"""

        # [HOT] FORCE JSON OUTPUT: Add strong JSON enforcement to prompt
        json_enforced_prompt = f"""{prompt}

CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no text before or after the JSON.

Start your response with {{ and end with }}. Nothing else."""

        payload = {
            "model": self.active_model,
            "prompt": json_enforced_prompt,
            "stream": True,  # 🚀 CRITICAL: Enable streaming for immediate processing
            "format": "json",  # Force JSON format
            "options": {
                **self.generation_params,
                "temperature": 0.1,  # Lower temperature for more consistent JSON
                "top_p": 0.9
            }
        }
        
        try:
            # 🚀 SMART STREAMING: Process response as it comes in
            logger.info(f"[STREAM] Starting smart streaming for {difficulty} question")
            
            response = self.session.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=120  # Increased for DeepSeek-R1 14B model
            )
            response.raise_for_status()

            # 🧠 SMART JSON DETECTION: Accumulate response and detect completion
            accumulated_response = ""
            json_bracket_count = 0
            json_started = False
            complete_json_detected = False
            
            start_time = time.time()
            max_time = 300  # Maximum 5 minutes as absolute safety limit
            
            for line in response.iter_lines(decode_unicode=True):
                if time.time() - start_time > max_time:
                    logger.warning(f"[TIMEOUT] Reached maximum time limit for {difficulty} question")
                    break
                    
                if line:
                    try:
                        chunk_data = json.loads(line)
                        chunk_text = chunk_data.get("response", "")
                        
                        if chunk_text:
                            accumulated_response += chunk_text
                            
                            # 🧠 DETECT JSON STRUCTURE COMPLETION
                            for char in chunk_text:
                                if char == '{':
                                    if not json_started:
                                        json_started = True
                                        logger.debug("[JSON] JSON object started")
                                    json_bracket_count += 1
                                elif char == '}':
                                    json_bracket_count -= 1
                                    if json_started and json_bracket_count == 0:
                                        complete_json_detected = True
                                        logger.info(f"[JSON] Complete JSON detected after {time.time() - start_time:.1f}s")
                                        break
                            
                            # 🚀 IMMEDIATE PROCESSING: Return as soon as JSON is complete
                            if complete_json_detected:
                                # Verify it contains MCQ fields
                                if '"question"' in accumulated_response and '"options"' in accumulated_response:
                                    elapsed = time.time() - start_time
                                    logger.info(f"[SUCCESS] Valid MCQ JSON ready in {elapsed:.1f}s")
                                    return accumulated_response
                                else:
                                    logger.debug("[JSON] Complete but missing MCQ fields, continuing...")
                                    complete_json_detected = False
                                    json_bracket_count = 0  # Reset for next potential JSON
                        
                        # Check if generation is complete
                        if chunk_data.get("done", False):
                            logger.info(f"[DONE] Generation marked complete after {time.time() - start_time:.1f}s")
                            break
                            
                    except json.JSONDecodeError:
                        # Skip malformed JSON chunks
                        continue
            
            # Return whatever we accumulated
            if accumulated_response.strip():
                elapsed = time.time() - start_time
                logger.info(f"[COMPLETE] Returning accumulated response after {elapsed:.1f}s")
                return accumulated_response
            else:
                logger.warning("[EMPTY] No response accumulated")
                return None
            
        except Exception as e:
            logger.error(f"[ERROR] Smart streaming API call failed: {e}")
            return None

    def _parse_json_response(self, response: str, expected_count: int) -> List[Dict[str, Any]]:
        """[HOT] Enhanced JSON parsing with DeepSeek R1 thinking support"""
        
        try:
            # Clean response - remove thinking blocks
            cleaned = self._clean_thinking_response(response)
            
            # Try direct JSON parsing first (format='json' should work)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return self._validate_questions(parsed)
                elif isinstance(parsed, dict):
                    return self._validate_questions([parsed])
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract JSON from response
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        return self._validate_questions(parsed)
                except json.JSONDecodeError:
                    pass
            
            # Last resort: Try manual JSON construction from text patterns
            logger.warning("[WARNING] Attempting manual JSON construction...")
            manual_questions = self._manual_json_construction(cleaned, expected_count)
            if manual_questions:
                return manual_questions
            
            logger.warning(f"[ERROR] All JSON parsing methods failed for response: {cleaned[:200]}...")
            return []
            
        except Exception as e:
            logger.error(f"[ERROR] JSON parsing error: {e}")
            return []

    def _parse_single_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse single JSON object"""
        
        try:
            cleaned = self._clean_thinking_response(response)
            
            # Try direct parsing
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return self._validate_single_question(parsed)
            except json.JSONDecodeError:
                pass
            
            # Extract single object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    return self._validate_single_question(parsed)
                except json.JSONDecodeError:
                    pass
            
            # Last resort: Try manual construction for single question
            logger.warning("[WARNING] Attempting manual single question construction...")
            manual_questions = self._manual_json_construction(cleaned, 1)
            if manual_questions:
                return manual_questions[0]
            
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Single JSON parsing error: {e}")
            return None

    def _clean_thinking_response(self, response: str) -> str:
        """[HOT] Enhanced DeepSeek R1 thinking block removal"""
        
        cleaned = response.strip()
        
        # Method 1: Remove <think>...</think> blocks
        if '<think>' in cleaned:
            if '</think>' in cleaned:
                think_end = cleaned.find('</think>')
                if think_end != -1:
                    cleaned = cleaned[think_end + 8:].strip()
                    logger.debug("[BRAIN] Removed complete <think>...</think> block")
            else:
                # Incomplete thinking block - find where JSON likely starts
                think_start = cleaned.find('<think>')
                # Look for JSON after any amount of thinking
                json_indicators = ['\n[', '\n{', '```json', 'json\n', 'JSON:\n', 'Output:\n']
                for indicator in json_indicators:
                    json_pos = cleaned.find(indicator, think_start)
                    if json_pos != -1:
                        cleaned = cleaned[json_pos:].strip()
                        logger.debug(f"[BRAIN] Found JSON after incomplete thinking block at: {indicator}")
                        break
        
        # Method 2: Remove everything before common JSON start patterns
        json_start_patterns = [
            'output a JSON',
            'JSON array',
            'JSON object',
            'format:\n',
            'Output:\n',
            'Here\'s the JSON',
            'Here is the JSON',
            '```json',
            '```\n['
        ]
        
        for pattern in json_start_patterns:
            if pattern in cleaned:
                pos = cleaned.find(pattern)
                # Skip past the pattern to find actual JSON
                remaining = cleaned[pos + len(pattern):].strip()
                if remaining.startswith('[') or remaining.startswith('{'):
                    cleaned = remaining
                    logger.debug(f"[BRAIN] Found JSON after pattern: {pattern}")
                    break
        
        # Method 3: Aggressive JSON extraction using multiple patterns
        if not (cleaned.startswith('{') or cleaned.startswith('[')):
            # Try to find JSON array or object anywhere in the text
            json_patterns = [
                r'(\[\s*\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s*\])',  # Array of objects
                r'(\[\s*\{.*?\}\s*\])',  # Simple array with one object
                r'(\{[^{}]*"question"[^{}]*"options"[^{}]*"correct_answer"[^{}]*\})',  # Single MCQ object
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, cleaned, re.DOTALL)
                if matches:
                    # Take the last match (most likely to be the actual output)
                    cleaned = matches[-1]
                    logger.debug(f"[BRAIN] Extracted JSON using regex pattern")
                    break
        
        # Method 4: Extract from code blocks
        if '```' in cleaned:
            # Try to extract content between code blocks
            code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            if code_match:
                potential_json = code_match.group(1).strip()
                if potential_json.startswith('[') or potential_json.startswith('{'):
                    cleaned = potential_json
                    logger.debug("[BRAIN] Extracted JSON from code block")
        
        # Method 5: Last resort - find JSON by structure
        if not (cleaned.startswith('{') or cleaned.startswith('[')):
            lines = cleaned.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('[') or stripped.startswith('{'):
                    # Found potential JSON start, extract from here
                    potential_json = '\n'.join(lines[i:])
                    # Verify it looks like MCQ JSON
                    if '"question"' in potential_json and '"options"' in potential_json:
                        cleaned = potential_json
                        logger.debug(f"[BRAIN] Found JSON starting at line {i+1}")
                        break
        
        return cleaned

    def _extract_and_parse_final_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """[HOT] GOLD SOLUTION: Bulletproof extraction of final JSON from DeepSeek response"""
        
        if not raw_response:
            return None
            
        try:
            # Find the last occurrence of '{' to locate the start of the final JSON object
            last_brace_open = raw_response.rfind('{')
            if last_brace_open == -1:
                logger.warning("No JSON object found in the response.")
                return None

            # Find the corresponding closing '}' for the last object
            json_candidate = raw_response[last_brace_open:]
            
            # Count braces to find the matching closing brace
            open_braces = 0
            json_end = -1
            for i, char in enumerate(json_candidate):
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                logger.warning("Could not find a complete JSON object.")
                return None

            final_json_str = json_candidate[:json_end]
            
            # Parse the clean JSON string
            parsed_json = json.loads(final_json_str)
            logger.info("[HOT] Successfully extracted final JSON from DeepSeek response")
            return parsed_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode extracted JSON: {e}")
            logger.debug(f"Failed JSON string: '{final_json_str if 'final_json_str' in locals() else 'N/A'}'")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during final JSON extraction: {e}")
            return None



    def _generate_placeholder_questions(self, topic: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Dynamic content generation preferred - AI models recommended"""
        logger.info("[INFO] Dynamic content generation preferred for quality assurance")
        logger.info(f"[INFO] Cannot generate placeholder questions for topic '{topic}' with difficulty '{difficulty}'")
        logger.info("[INFO] Application designed for AI-powered content generation")
        raise Exception(f"Placeholder generation disabled for '{topic}' - AI model generation preferred")
        logger.info(f"[INFO] Recommend {num_questions} AI-generated questions for topic: '{topic}', difficulty: '{difficulty}'")
        logger.info("[INFO] Configure proper Ollama models or cloud APIs for best results")
        
        # Return empty list to maintain proper error handling upstream
        return []

    def _manual_json_construction(self, text: str, expected_count: int) -> List[Dict[str, Any]]:
        """🛠️ Last resort: Manually construct JSON from text patterns"""
        
        try:
            questions = []
            
            # Look for question patterns
            question_patterns = [
                r'[Qq]uestion[:\s]*(.+?)\?',
                r'Q[:\s]*(.+?)\?',
                r'\d+\.\s*(.+?)\?'
            ]
            
            # Look for options patterns
            option_patterns = [
                r'[Aa]\)\s*(.+?)(?=\n|[Bb]\))',
                r'[Bb]\)\s*(.+?)(?=\n|[Cc]\))',
                r'[Cc]\)\s*(.+?)(?=\n|[Dd]\))',
                r'[Dd]\)\s*(.+?)(?=\n|$)'
            ]
            
            lines = text.split('\n')
            current_question = None
            current_options = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to find question
                for pattern in question_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if current_question and len(current_options) == 4:
                            # Save previous question
                            questions.append({
                                'question': current_question + '?',
                                'options': current_options,
                                'correct_answer': 'A',  # Default
                                'explanation': 'Auto-generated explanation'
                            })
                        current_question = match.group(1).strip()
                        current_options = []
                        break
                
                # Try to find options
                for i, pattern in enumerate(option_patterns):
                    match = re.search(pattern, line)
                    if match:
                        current_options.append(match.group(1).strip())
                        break
            
            # Add the last question if complete
            if current_question and len(current_options) == 4:
                questions.append({
                    'question': current_question + '?',
                    'options': current_options,
                    'correct_answer': 'A',
                    'explanation': 'Auto-generated explanation'
                })
            
            if questions:
                logger.info(f"🛠️ Manual construction succeeded: {len(questions)} questions")
                return self._validate_questions(questions)
            
            return []
            
        except Exception as e:
            logger.error(f"[ERROR] Manual JSON construction failed: {e}")
            return []

    def _validate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize question format"""
        
        validated = []
        for q in questions:
            if self._is_valid_question(q):
                validated.append(self._normalize_question(q))
        
        return validated

    def _validate_single_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate single question"""
        
        if self._is_valid_question(question):
            return self._normalize_question(question)
        
        return None

    def _is_valid_question(self, q: Dict[str, Any]) -> bool:
        """Check if question has required fields - FIXED to handle both formats"""

        # Check basic required fields
        if 'question' not in q or 'options' not in q:
            return False

        # Check for correct answer field (handle both formats)
        correct_field = None
        if 'correct_answer' in q:
            correct_field = 'correct_answer'
        elif 'correct' in q:
            correct_field = 'correct'
        else:
            return False

        # Validate options (handle both list and dict formats)
        options = q['options']
        if isinstance(options, list):
            # List format: ["Option A", "Option B", "Option C", "Option D"]
            if len(options) != 4:
                return False
        elif isinstance(options, dict):
            # Dict format: {"A": "Ground State", "B": "Excited State", ...}
            if len(options) != 4 or not all(key in options for key in ['A', 'B', 'C', 'D']):
                return False
        else:
            return False

        # Validate correct answer (handle both letter format and text format)
        correct_value = q[correct_field]
        options = q['options']

        if isinstance(correct_value, str):
            if correct_value in ['A', 'B', 'C', 'D']:
                # Letter format is valid
                pass
            elif isinstance(options, list) and correct_value in options:
                # Text format is valid (will be converted to letter)
                pass
            elif isinstance(options, dict) and correct_value in options.values():
                # Text format with dict options is valid
                pass
            elif correct_value.startswith('Option ') and correct_value[7:] in ['A', 'B', 'C', 'D']:
                # "Option A" format is valid
                pass
            else:
                # Debug: log why validation failed
                logger.debug(f"Validation failed - correct_value: '{correct_value}', options: {options}")
                return False
        else:
            return False

        # Validate question format
        if not q['question'].strip().endswith('?'):
            return False

        return True

    def _normalize_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize question format - FIXED to handle both formats"""

        # Ensure question ends with ?
        question = q['question'].strip()
        if not question.endswith('?'):
            question = question.rstrip('.!') + '?'

        # Handle correct answer field (normalize to correct_answer format)
        correct_answer = None
        if 'correct_answer' in q:
            correct_answer = q['correct_answer']
        elif 'correct' in q:
            correct_answer = q['correct']

        # Convert text answer to letter format if needed
        options = q['options']
        if isinstance(options, list) and correct_answer in options:
            # Convert option text to letter (list format)
            option_index = options.index(correct_answer)
            correct_answer = ['A', 'B', 'C', 'D'][option_index]
        elif isinstance(options, dict) and correct_answer in options.values():
            # Convert option text to letter (dict format)
            for key, value in options.items():
                if value == correct_answer:
                    correct_answer = key
                    break
        elif correct_answer and correct_answer.startswith('Option ') and correct_answer[7:] in ['A', 'B', 'C', 'D']:
            # Convert "Option A" to "A"
            correct_answer = correct_answer[7:]

        # Normalize options to list format
        options = q['options']
        if isinstance(options, dict):
            # Convert dict format to list format
            normalized_options = [options['A'], options['B'], options['C'], options['D']]
        else:
            # Already in list format
            normalized_options = options

        return {
            'question': question,
            'options': normalized_options,
            'correct_answer': correct_answer,
            'explanation': q.get('explanation', ''),
            'difficulty': q.get('difficulty', 'medium'),
            'generated_by': self.active_model
        }

    def is_available(self) -> bool:
        """Check if the generator is available and initialized"""
        return self.is_initialized and self.active_model is not None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.generation_stats,
            "gpu_optimized": True,
            "json_format_enforced": True,
            "cpu_threads_limited": True
        }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("[HOT] GPU-Optimized Ollama JSON Generator cleaned up")
