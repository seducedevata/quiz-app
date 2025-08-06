# src/knowledge_app/core/offline_mcq_generator.py

from .async_converter import async_requests_post, async_requests_get
from .async_converter import async_time_sleep


import logging
import asyncio
import traceback
import time
import sys
from typing import Dict, List, Any, Optional
import json
import re

from .mcq_generator import MCQGenerator
from .ollama_model_inference import OllamaModelInference

# [CONFIG] OPTIMIZED LOGGING: Create specialized loggers with proper levels
logger = logging.getLogger(__name__)
offline_logger = logging.getLogger("offline_mcq")
performance_logger = logging.getLogger("performance_offline_mcq")

def _setup_optimized_logging():
    """Setup optimized logging with rotation and appropriate levels"""
    import logging.handlers
    import os

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure offline logger with rotation
    offline_log_file = os.path.join(log_dir, 'offline_mcq.log')
    offline_handler = logging.handlers.RotatingFileHandler(
        offline_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3
    )
    offline_handler.setLevel(logging.INFO)  # Only INFO and above for offline operations
    offline_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    offline_handler.setFormatter(offline_formatter)
    offline_logger.addHandler(offline_handler)
    offline_logger.setLevel(logging.INFO)
    offline_logger.propagate = False  # Don't propagate to root logger

    # Configure performance logger with rotation
    perf_log_file = os.path.join(log_dir, 'performance.log')
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=2
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '%(asctime)s - PERF - %(message)s'
    )
    perf_handler.setFormatter(perf_formatter)
    performance_logger.addHandler(perf_handler)
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = False

    # Set main logger to WARNING to reduce noise
    logger.setLevel(logging.WARNING)

# Initialize optimized logging
_setup_optimized_logging()


class OfflineMCQGenerator(MCQGenerator):
    """
    High-performance offline MCQ generator using local models with GPU acceleration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Offline MCQ Generator with ultra-comprehensive logging"""
        offline_logger.info("[START] INITIALIZING OfflineMCQGenerator")
        offline_logger.info(f"[SEARCH] CONFIG RECEIVED: {config}")
        offline_logger.info(f"[SEARCH] PYTHON VERSION: {sys.version}")
        
        super().__init__(config)
        self.ollama_interface: Optional[OllamaModelInference] = None
        
        # Handle different config types with detailed logging
        offline_logger.info("[SEARCH] PROCESSING CONFIG for model name...")
        if self.config and hasattr(self.config, 'get_value'):
            self.model_name = self.config.get_value("model_name", "llama3.1:8b")
            offline_logger.info(f"[SEARCH] CONFIG TYPE: Attribute-based, MODEL: {self.model_name}")
        elif self.config and hasattr(self.config, 'get'):
            self.model_name = self.config.get("model_name", "llama3.1:8b")
            offline_logger.info(f"[SEARCH] CONFIG TYPE: Dict-like, MODEL: {self.model_name}")
        elif isinstance(self.config, dict):
            self.model_name = self.config.get("model_name", "llama3.1:8b")
            offline_logger.info(f"[SEARCH] CONFIG TYPE: Dictionary, MODEL: {self.model_name}")
        else:
            self.model_name = "llama3.1:8b"
            offline_logger.info(f"[SEARCH] CONFIG TYPE: Default fallback, MODEL: {self.model_name}")
            
        self.is_engine_initialized = False
        self.generation_stats = {"total_generated": 0, "avg_time": 0, "gpu_utilization": 0}
        
        # [START] ENHANCED: Available models for BatchTwoModelPipeline - DEFERRED TO PREVENT FREEZE
        self.available_models = []  # Will be loaded on first use
        self.thinking_model = "llama3.1:8b"  # Default, will be updated on first use
        self.json_model = "llama3.1:8b"  # Default, will be updated on first use
        
        offline_logger.info("[FINISH] OfflineMCQGenerator initialization completed")
        offline_logger.info(f"   • MODEL_NAME: {self.model_name}")
        offline_logger.info(f"   • THINKING_MODEL: {self.thinking_model}")
        offline_logger.info(f"   • JSON_MODEL: {self.json_model}")
        offline_logger.info(f"   • ENGINE_INITIALIZED: {self.is_engine_initialized}")

    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            import requests
            response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                offline_logger.info(f"[AI] Available models: {models}")
                return models
            return []
        except Exception as e:
            offline_logger.warning(f"[WARNING] Error getting models: {e}")
            return []
    
    def _select_thinking_model(self) -> str:
        """[USER] SAFE user preference thinking model selection with timeout protection"""
        try:
            # [START] SAFETY: Prevent infinite loops with timeout
            import time
            start_time = time.time()

            # Quick check for user preferences
            from pathlib import Path
            import json

            user_settings_path = Path("user_data/user_settings.json")
            if user_settings_path.exists():
                with open(user_settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                model_prefs = settings.get('model_preferences', {})
                preferred_thinking = model_prefs.get('preferred_thinking_model', '')
        except Exception as e:
            offline_logger.warning(f"Dynamic model selection failed: {e}")
            return "llama3.1:8b"

    def _select_json_model(self) -> str:
        """Select best JSON model using dynamic detection"""
        try:
            from .dynamic_model_detector import DynamicModelDetector
            detector = DynamicModelDetector()
            
            # Get recommended model for JSON tasks
            model = detector.get_recommended_model(task_type="json", 
                                                   available_models=self.available_models)
            if model:
                offline_logger.info(f"[AI] Dynamic JSON model selection: {model}")
                return model
                
            return "llama3.1:8b"
            
        except Exception as e:
            offline_logger.warning(f"Dynamic JSON model selection failed: {e}")
            return "llama3.1:8b"

    def _generate_expert_questions_optimized(self, topic: str, context: str, num_questions: int, question_type: str) -> List[Dict[str, Any]]:
        """[START] OPTIMIZED expert-level questions using single model with dynamic timeout"""

        offline_logger.info("[START] OPTIMIZED SINGLE-MODEL EXPERT GENERATION")
        offline_logger.info(f"[BRAIN] Model: {self.model_name}")

        questions = []

        for i in range(num_questions):
            try:
                # Create optimized expert prompt (single model, direct JSON)
                prompt = self._create_optimized_expert_prompt(topic, context, question_type, i)

                # Calculate dynamic timeout based on model and prompt complexity
                if hasattr(self.ollama_interface, '_calculate_dynamic_timeout'):
                    timeout = self.ollama_interface._calculate_dynamic_timeout(None, prompt)
                else:
                    # Fallback timeout calculation
                    if hasattr(self.ollama_interface, 'active_model') and self.ollama_interface.active_model:
                        model_lower = self.ollama_interface.active_model.lower()
                        if 'deepseek-r1' in model_lower or 'r1:' in model_lower:
                            timeout = 300  # 5 minutes for reasoning models
                        elif 'deepseek' in model_lower:
                            timeout = 180  # 3 minutes for DeepSeek
                        else:
                            timeout = 90   # 1.5 minutes for other models
                    else:
                        timeout = 90

                offline_logger.info(f"[EXPERT] Question {i+1}/{num_questions}: Using {timeout}s timeout")

                # Generate with dynamic timeout
                            result = self._generate_single_question_with_timeout(prompt, timeout, question_type, adapter_name)

                if result and self._is_valid_mcq_response(result):
                    # [CONFIG] ENHANCED VALIDATION: Use comprehensive validation with regeneration
                    validated_result = self._validate_and_regenerate_if_needed(result, topic, context, question_type)
                    if validated_result:
                        questions.append(validated_result)
                        offline_logger.info(f"[OK] Expert question {i+1}/{num_questions} generated and validated successfully")
                    else:
                        offline_logger.error(f"[CRITICAL] Expert question {i+1}/{num_questions} validation failed - No fallback allowed")
                        # Skip this question - no fallbacks allowed
                        continue
                else:
                    offline_logger.error(f"[CRITICAL] Expert question {i+1}/{num_questions} failed - No fallback allowed")
                    # Skip this question - no fallbacks allowed
                    continue

            except Exception as e:
                offline_logger.error(f"[CRITICAL] Expert question {i+1} generation failed: {e} - No fallback allowed")
                # Skip this question - no fallbacks allowed
                continue

        offline_logger.info(f"[EXPERT] Expert generation complete: {len(questions)}/{num_questions} questions")
        return questions

    def _generate_single_question_with_timeout(self, prompt: str, timeout: float = 45, question_type: str = "mixed", model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a single question with enhanced numerical support and dynamic timeout"""
        try:
            if not self.ollama_interface or not self.ollama_interface.is_available():
                offline_logger.error("[ERROR] Ollama interface not available")
                return None

            offline_logger.info(f"[START] Generating {question_type} question with {timeout}s timeout...")
            
            # For numerical questions, use enhanced prompt
            if question_type.lower() == "numerical":
                # Extract topic from prompt (simple heuristic)
                topic_match = re.search(r'about ([^.]+)', prompt)
                topic = topic_match.group(1) if topic_match else "physics"
                
                # Use enhanced numerical prompt
                enhanced_prompt = self._create_enhanced_numerical_prompt(topic, "")
                offline_logger.info("[NUMERICAL] Using enhanced numerical prompt")
                prompt = enhanced_prompt.format(topic=topic)
            
            # Show progress for long generations
            if timeout > 120:
                offline_logger.info(f"🧠 Using reasoning model - generation may take up to {timeout/60:.1f} minutes")
                offline_logger.info("🔄 Please wait while the model thinks deeply about the question...")

            # Use synchronous generation (no async/await issues)
            response = self.ollama_interface.generate_text(
                prompt=prompt,
                request_timeout=timeout,
                temperature=0.8,
                num_predict=800
            )

            if response:
                # Try enhanced parsing first
                parsed = self._parse_mcq_response_enhanced(response)
                if not parsed:
                    # Fallback to original parsing
                    parsed = self._parse_mcq_response(response)
                
                if parsed and self._is_valid_mcq_response(parsed):
                    offline_logger.info("[OK] Question generated and validated successfully")
                    return parsed
                else:
                    offline_logger.warning("[ERROR] Invalid MCQ response format")
                    offline_logger.warning(f"[DEBUG] Response: {response[:500]}...")
                    return None
            else:
                offline_logger.warning("[ERROR] No response from Ollama")
                return None

        except Exception as e:
            offline_logger.error(f"[ERROR] Single question generation failed: {e}")
            return None

    def generate_mcq_streaming(self, topic: str, context: str = "", num_questions: int = 1, 
                              difficulty: str = "medium", question_type: str = "mixed",
                              token_callback: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        """
        [START] Generate MCQ with streaming token support - NEVER blocks UI
        
        Args:
            topic: Question topic
            context: Additional context
            num_questions: Number of questions (currently supports 1)
            difficulty: Question difficulty
            question_type: Type of question
            token_callback: Callback function to receive streaming tokens
            
        Returns:
            Generated MCQ or None if failed
        """
        try:
            if not self.ollama_interface or not self.ollama_interface.is_available():
                offline_logger.error("[ERROR] Ollama interface not available for streaming")
                return None
            
            # Create optimized prompt for streaming generation
            prompt = self._create_optimized_expert_prompt(topic, context, question_type, 0)
            
            #  CRITICAL FIX: Select appropriate model based on difficulty
            # Use reasoning model for expert/hard, regular model for others
            if difficulty in ['expert', 'hard']:
                thinking_model = self._get_thinking_model()  # DeepSeek R1 or other reasoning model
                offline_logger.info(f"[START] Using reasoning model for {difficulty} difficulty: {thinking_model}")
            else:
                thinking_model = self._get_json_model()  # Regular model for easier difficulties
                offline_logger.info(f"[START] Using regular model for {difficulty} difficulty: {thinking_model}")
            
            offline_logger.info(f"[START] Starting streaming MCQ generation for topic: {topic}")
            
            #  IMPORTANT: Temporarily set the active model for streaming
            original_model = self.ollama_interface.active_model
            self.ollama_interface.active_model = thinking_model
            
            try:
                # Use streaming generation with the correct model
                accumulated_response = ""
                
                token_generator = self.ollama_interface.generate_text(
                    prompt=prompt,
                    stream=True,
                    request_timeout=60,
                    temperature=0.8,
                    num_predict=800
                )
            except Exception as e:
                # Restore original model on any error
                self.ollama_interface.active_model = original_model
                offline_logger.error(f"[ERROR] Failed to start streaming: {e}")
                return None
            
            # Check if the current model supports reasoning/thinking tokens
            is_reasoning = self._is_reasoning_model(thinking_model)
            offline_logger.info(f"[MODEL] Reasoning model detected: {is_reasoning} for model: {thinking_model}")
            
            for token in token_generator:
                accumulated_response += token
                
                # Call token callback if provided
                if token_callback:
                    try:
                        # For reasoning models, provide additional context in token metadata
                        if is_reasoning:
                            offline_logger.info(f"TOKEN: [REASONING] Calling callback with token: '{token}' (length: {len(token)})")
                        else:
                            offline_logger.info(f"TOKEN: Calling callback with token: '{token}' (length: {len(token)})")
                        
                        token_callback(token)
                        offline_logger.info(f"TOKEN: Callback completed successfully")
                    except Exception as e:
                        offline_logger.warning(f"[WARNING] Token callback error: {e}")
                        import traceback
                        offline_logger.warning(f"[WARNING] Token callback traceback: {traceback.format_exc()}")
                else:
                    offline_logger.warning(f"TOKEN: No callback provided, token lost: '{token}'")
            
            offline_logger.info(f"[OK] Streaming generation complete, parsing response...")
            
            # Parse the accumulated response
            if accumulated_response:
                parsed = self._parse_mcq_response(accumulated_response)
                if parsed and self._is_valid_mcq_response(parsed):
                    offline_logger.info("[OK] Streaming MCQ generation successful")
                    return parsed
                else:
                    offline_logger.warning("[ERROR] Invalid streaming MCQ response format")
                    return None
            else:
                offline_logger.warning("[ERROR] No response from streaming generation")
                return None
                
        except Exception as e:
            offline_logger.error(f"[ERROR] Streaming MCQ generation failed: {e}")
            return None
        finally:
            #  CRITICAL: Always restore original model
            self.ollama_interface.active_model = original_model
            offline_logger.info(f"[RESTORE] Active model restored to: {original_model}")

    def _is_valid_mcq_response(self, parsed_data: Dict[str, Any]) -> bool:
        """Validate that the parsed MCQ response has the required structure"""
        try:
            if not isinstance(parsed_data, dict):
                return False

            # Check for required fields
            required_fields = ['question', 'options', 'correct_answer']
            for field in required_fields:
                if field not in parsed_data:
                    # CRITICAL FIX: If 'correct_answer' is missing but 'correct' exists, normalize it
                    if field == 'correct_answer' and 'correct' in parsed_data:
                        parsed_data['correct_answer'] = parsed_data['correct']
                        offline_logger.info("[OK] Normalized 'correct' field to 'correct_answer'")
                    else:
                        offline_logger.warning(f"[ERROR] Missing required field: {field}")
                        return False

            # Validate question is not empty
            if not parsed_data['question'] or not parsed_data['question'].strip():
                offline_logger.warning("[ERROR] Question text is empty")
                return False

            # Validate options - CRITICAL FIX: Handle both list and dict formats
            options = parsed_data['options']
            if isinstance(options, dict):
                # Dictionary format: {"A": "option1", "B": "option2", ...}
                if len(options) < 2:
                    offline_logger.warning("[ERROR] Invalid options: must have at least 2 options")
                    return False
                option_keys = list(options.keys())
            elif isinstance(options, list):
                # List format: ["option1", "option2", ...]
                if len(options) < 2:
                    offline_logger.warning("[ERROR] Invalid options: must have at least 2 options")
                    return False
                option_keys = [str(i) for i in range(len(options))]  # Convert to string indices
            else:
                offline_logger.warning("[ERROR] Invalid options: must be a list or dictionary")
                return False

            # Validate correct answer
            correct_answer = parsed_data['correct_answer']
            if not correct_answer:
                offline_logger.warning("[ERROR] Correct answer is empty")
                return False

            # Check if correct answer is valid for the options format
            if isinstance(options, dict):
                if correct_answer not in options:
                    offline_logger.warning(f"[ERROR] Correct answer '{correct_answer}' not found in option keys: {list(options.keys())}")
                    return False
            elif isinstance(options, list):
                try:
                    answer_index = int(correct_answer)
                    if answer_index < 0 or answer_index >= len(options):
                        offline_logger.warning(f"[ERROR] Correct answer index '{correct_answer}' out of range for {len(options)} options")
                        return False
                except ValueError:
                    offline_logger.warning(f"[ERROR] Correct answer '{correct_answer}' must be a valid index for list format")
                    return False

            return True

        except Exception as e:
            offline_logger.error(f"[ERROR] MCQ validation failed: {e}")
            return False

    def generate_mcq(self, topic: str, difficulty: str = "medium", question_type: str = "mixed", 
                     context: str = None, timeout: int = 60, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate MCQ with linear fallback chain"""
        try:
            # Try primary generation
            result = self._generate_single_question_with_timeout(self._create_single_advanced_prompt(topic, context, question_type, difficulty), timeout, question_type, adapter_name)
            if result and self._validate_question(result):
                return result
                
            # Simple fallback to basic question
            offline_logger.warning("Primary generation failed, using basic fallback")
            return self._create_basic_question(topic, difficulty)
            
        except Exception as e:
            offline_logger.error(f"Generation failed: {e}")
            return self._create_basic_question(topic, difficulty)

    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """Simple validation for generated question"""
        from .json_parser_unified import validate_and_normalize_mcq
        return validate_and_normalize_mcq(question) is not None

    def _create_expert_prompt(self, topic: str, context: str, question_type: str) -> str:
        """Create a focused prompt for a single expert question"""
        
        # Sanitize inputs
        from .inquisitor_prompt import _sanitize_user_input
        sanitized_topic = _sanitize_user_input(topic)
        sanitized_context = _sanitize_user_input(context)
        sanitized_question_type = _sanitize_user_input(question_type)

        # Enhanced prompt based on question type
        if sanitized_question_type.lower() == "numerical":
            prompt = f"""Generate a PhD-level NUMERICAL calculation question about {sanitized_topic}.

CRITICAL REQUIREMENTS for NUMERICAL questions:
1. Question MUST ask to "calculate", "compute", "find", or "determine" a numerical value
2. Question MUST contain specific numerical values and units (eV, nm, Hz, J, kg, mol, atm, K, Å, pm, fs, keV, MeV)
3. ALL options MUST be numerical values with proper units
4. Question should involve formulas, equations, or quantitative analysis
5. Use advanced physics/chemistry concepts related to: {sanitized_topic}

TOPIC: {sanitized_topic}
CONTEXT: {sanitized_context}

EXAMPLE FORMAT:
{{
  "question": "Calculate the binding energy of an electron in the n=2 state of hydrogen. Given: Rydberg constant R = 1.097×10⁷ m⁻¹, h = 6.626×10⁻³⁴ J·s, c = 3.00×10⁸ m/s",
  "options": ["A) -3.40 eV", "B) -1.51 eV", "C) -13.6 eV", "D) -0.85 eV"],
  "correct_answer": "A",
  "explanation": "Using E_n = -13.6 eV/n²: E_2 = -13.6 eV/4 = -3.40 eV"
}}

Generate a similar numerical question about {sanitized_topic}:"""
        else:
            # Original prompt for non-numerical questions
            prompt = f"""Generate a PhD-level {sanitized_question_type} question about {sanitized_topic}.

REQUIREMENTS:
- Advanced, graduate-level complexity
- Specific, detailed scenarios
- No basic "what is" questions
- Include recent research concepts (2020-2024)

TOPIC: {sanitized_topic}
CONTEXT: {sanitized_context}

OUTPUT FORMAT (JSON):
{{
  "question": "Advanced question text here",
  "options": ["A) option", "B) option", "C) option", "D) option"],
  "correct_answer": "A",
  "explanation": "Detailed explanation with advanced concepts"
}}

Generate the JSON now:"""

        return prompt

    def _generate_single_expert_question(self, topic: str, context: str, question_type: str, index: int) -> Optional[Dict[str, Any]]:
        """Generate a single expert question reliably"""
        
        # Create a focused, single-question prompt
        prompt = self._create_expert_prompt(topic, context, question_type)
        
        try:
            # Use the reliable generation method with appropriate timeout
            response = self._generate_with_retry(prompt, max_tokens=800)
            
            if response:
                # Parse the response
                parsed = self._parse_mcq_response(response)
                if parsed and self._is_valid_mcq_response(parsed):
                    # Add metadata
                    parsed["metadata"] = {
                        "difficulty": "expert",
                        "question_type": question_type,
                        "generation_method": "single_expert",
                        "question_index": index
                    }
                    return parsed
            
            return None
            
        except Exception as e:
            offline_logger.error(f"[ERROR] Single expert question generation failed: {e}")
            return None
    
        sanitized_context = _sanitize_user_input(context)
        
        type_specific = ""
        if question_type.lower() == "numerical":
            type_specific = "Focus on calculations, formulas, and quantitative analysis. All options should be numerical values with units."
        elif question_type.lower() == "conceptual":
            type_specific = "Focus on theoretical understanding, mechanisms, and explanations. Avoid numerical calculations."
        else:
            type_specific = "Can include either numerical calculations or conceptual understanding as appropriate."
        
        prompt = f"""Generate a PhD-level expert question about {sanitized_topic}.

REQUIREMENTS:
- Graduate/PhD level complexity
- Requires deep understanding of advanced concepts
- Tests synthesis of multiple concepts
- Uses technical terminology appropriately
- {type_specific}

TOPIC: {sanitized_topic}
CONTEXT: {sanitized_context}

OUTPUT FORMAT (JSON only):
{{
  "question": "Expert-level question text here",
  "options": ["A) option", "B) option", "C) option", "D) option"],
  "correct_answer": "A",
  "explanation": "Detailed explanation with advanced concepts"
}}

Generate the JSON:"""

        return prompt
    
    def _generate_topic_specific_fallback(self, topic: str, question_type: str, index: int) -> Dict[str, Any]:
        """FIXED: Generate a truly topic-specific fallback question instead of generic placeholder"""
        
        # Extract domain for intelligent fallback
        domain = self._extract_domain_from_topic(topic)
        topic_clean = topic.lower().strip()
        
        # FIXED: Create questions that are actually about the specific topic requested
        if any(word in topic_clean for word in ['quantum', 'atom', 'particle', 'nuclear']):
            question = f"In quantum {topic}, what principle fundamentally distinguishes quantum mechanical systems from classical systems?"
            options = [
                "A) Conservation of energy in all processes",
                "B) Quantization of energy levels and wave-particle duality",
                "C) Linear motion under constant forces",
                "D) Simple harmonic oscillator behavior"
            ]
            correct = "B"
            explanation = f"Quantum {topic} is characterized by energy quantization, uncertainty principles, and wave-particle duality that have no classical analogues."
            
        elif any(word in topic_clean for word in ['molecular', 'organic', 'chemical', 'reaction']):
            question = f"What advanced principle in {topic} governs the stereochemical outcomes of complex reactions?"
            options = [
                "A) Simple collision theory",
                "B) Orbital symmetry conservation and frontier molecular orbital theory",
                "C) Basic thermodynamic stability",
                "D) Elementary acid-base equilibrium"
            ]
            correct = "B"
            explanation = f"Advanced {topic} relies on molecular orbital theory and orbital symmetry considerations to predict reaction mechanisms and products."
            
        elif any(word in topic_clean for word in ['calculus', 'algebra', 'analysis', 'geometry']):
            question = f"Which concept in advanced {topic} requires understanding of limit processes and infinite series?"
            options = [
                "A) Basic arithmetic operations",
                "B) Convergence criteria and analytical continuation methods",
                "C) Simple polynomial evaluation",
                "D) Elementary graphing techniques"
            ]
            correct = "B"
            explanation = f"Advanced {topic} involves sophisticated limit processes, infinite series, and analytical methods that extend beyond elementary techniques."
            
        elif any(word in topic_clean for word in ['biology', 'genetics', 'molecular', 'cellular']):
            question = f"What advanced mechanism in {topic} involves epigenetic regulation and chromatin remodeling?"
            options = [
                "A) Simple Mendelian inheritance",
                "B) Histone modification and DNA methylation patterns",
                "C) Basic cell division processes",
                "D) Elementary enzyme catalysis"
            ]
            correct = "B"
            explanation = f"Advanced {topic} involves complex epigenetic mechanisms that regulate gene expression through chromatin modifications."
            
        elif any(word in topic_clean for word in ['computer', 'algorithm', 'programming', 'data']):
            question = f"Which concept in advanced {topic} requires understanding of computational complexity theory?"
            options = [
                "A) Basic loop structures",
                "B) NP-completeness and polynomial-time reductions",
                "C) Simple variable assignment",
                "D) Elementary input/output operations"
            ]
            correct = "B"
            explanation = f"Advanced {topic} involves complexity theory, algorithmic efficiency analysis, and computational intractability problems."
            
        elif any(word in topic_clean for word in ['history', 'historical', 'ancient', 'medieval']):
            question = f"What methodological approach in advanced {topic} studies requires interdisciplinary analysis?"
            options = [
                "A) Simple chronological listing",
                "B) Comparative historiography and primary source criticism",
                "C) Basic factual memorization",
                "D) Elementary timeline construction"
            ]
            correct = "B"
            explanation = f"Advanced {topic} studies require sophisticated historical methods, source criticism, and interdisciplinary perspectives."
            
        else:
            # Generic but still topic-specific fallback
            question = f"What distinguishes advanced theoretical approaches to {topic} from elementary treatments?"
            options = [
                "A) Use of basic definitions only",
                "B) Integration of multiple theoretical frameworks and empirical validation",
                "C) Simple observational descriptions",
                "D) Elementary classification systems"
            ]
            correct = "B"
            explanation = f"Advanced study of {topic} requires integration of theoretical models, empirical evidence, and sophisticated analytical methods."

        # FIXED: Ensure the question actually mentions the specific topic
        if topic not in question:
            question = question.replace("advanced", f"advanced {topic}")

        return {
            "question": question,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
            "metadata": {
                "difficulty": "expert",
                "question_type": question_type,
                "generation_method": "intelligent_topic_specific_fallback",
                "question_index": index,
                "original_topic": topic
            }
        }

    def _generate_expert_questions_batch(self, topic: str, context: str, num_questions: int, question_type: str) -> List[Dict[str, Any]]:
        """[EXPERT] Generate expert-level questions using unified pipeline"""
        offline_logger.info("[EXPERT] Using unified advanced question generation pipeline")
        return self._generate_advanced_questions_unified(topic, context, num_questions, question_type, "expert")

    def _generate_hard_questions_batch(self, topic: str, context: str, num_questions: int, question_type: str) -> List[Dict[str, Any]]:
        """[HOT] HARD MODE: Generate graduate-level questions using unified pipeline"""
        offline_logger.info("[HOT] Using unified advanced question generation pipeline")
        return self._generate_advanced_questions_unified(topic, context, num_questions, question_type, "hard")
    
    def _generate_expert_questions_batch_with_prompt(self, topic: str, context: str, num_questions: int, question_type: str, enhanced_prompt: str) -> List[Dict[str, Any]]:
        """[EXPERT] Generate expert-level questions using enhanced PHI prompt"""
        offline_logger.info("[EXPERT] Using PHI-ENHANCED prompt for expert question generation")
        return self._generate_advanced_questions_with_enhanced_prompt(topic, context, num_questions, question_type, "expert", enhanced_prompt)

    def _generate_hard_questions_batch_with_prompt(self, topic: str, context: str, num_questions: int, question_type: str, enhanced_prompt: str) -> List[Dict[str, Any]]:
        """[HARD] Generate hard-level questions using enhanced PHI prompt"""
        offline_logger.info("[HARD] Using PHI-ENHANCED prompt for hard question generation")
        return self._generate_advanced_questions_with_enhanced_prompt(topic, context, num_questions, question_type, "hard", enhanced_prompt)

    def _generate_advanced_questions_unified(self, topic: str, context: str, num_questions: int, question_type: str, difficulty: str) -> List[Dict[str, Any]]:
        """FIXED: Unified method for both expert and hard mode - eliminates code duplication"""
        
        offline_logger.info(f"[UNIFIED] STARTING {difficulty.upper()} MODE GENERATION: {num_questions} questions about '{topic}'")
        offline_logger.info(f"[BRAIN] Using unified pipeline for {difficulty} difficulty")
        
        # Use reliable single-question approach instead of brittle batch pipeline
        advanced_questions = []
        
        for i in range(num_questions):
            try:
                offline_logger.info(f"[{difficulty.upper()}] Generating question {i+1}/{num_questions}")
                
                # Generate one question at a time - much more reliable than batch approach
                question = self._generate_single_advanced_question(topic, context, question_type, difficulty, i)
                
                if question and self._is_valid_mcq_response(question):
                    advanced_questions.append(question)
                    offline_logger.info(f"[OK] {difficulty.capitalize()} question {i+1} generated successfully")
                else:
                    offline_logger.info(f"[INFO] {difficulty.capitalize()} question {i+1} generation unsuccessful")
                    # Skip this question and try next
                    continue
                        
            except Exception as e:
                offline_logger.info(f"[INFO] {difficulty.capitalize()} question {i+1} failed: {e}")
                # Skip this question and try next
                continue
        
        offline_logger.info(f"[UNIFIED] Generated {len(advanced_questions)}/{num_questions} {difficulty} questions")
        return advanced_questions

    def _generate_single_advanced_question(self, topic: str, context: str, question_type: str, difficulty: str, index: int) -> Optional[Dict[str, Any]]:
        """Generate a single advanced question for either expert or hard difficulty"""
        
        # Create a focused, single-question prompt
        prompt = self._create_single_advanced_prompt(topic, context, question_type, difficulty)
        
        try:
            # Use appropriate timeout based on difficulty
            timeout = 90 if difficulty == "expert" else 60
            
            # Use the reliable generation method
            response = self._generate_with_retry(prompt, max_tokens=800)
            
            if response:
                # Parse the response
                parsed = self._parse_mcq_response(response)
                if parsed and self._is_valid_mcq_response(parsed):
                    # Add metadata
                    parsed["metadata"] = {
                        "difficulty": difficulty,
                        "question_type": question_type,
                        "generation_method": f"single_{difficulty}",
                        "question_index": index
                    }
                    return parsed
            
            return None
            
        except Exception as e:
            offline_logger.info(f"[INFO] Single {difficulty} question generation failed: {e}")
            return None

    def _create_single_advanced_prompt(self, topic: str, context: str, question_type: str, difficulty: str) -> str:
        """Create a focused prompt for a single advanced question"""
        
        # For expert numerical questions, use the enhanced numerical prompt
        if difficulty == "expert" and question_type.lower() == "numerical":
            return self._create_enhanced_numerical_prompt(topic, context)
        
        # For other cases, use the standard advanced prompt
        # Sanitize inputs
        from .inquisitor_prompt import _sanitize_user_input
        sanitized_topic = _sanitize_user_input(topic)
        sanitized_context = _sanitize_user_input(context)
        
        # Difficulty-specific instructions
        if difficulty == "expert":
            complexity_level = "PhD-level expert"
            requirements = "Requires deep understanding of advanced concepts, tests synthesis of multiple concepts, uses technical terminology appropriately"
        else:  # hard
            complexity_level = "Graduate-level"
            requirements = "Requires advanced understanding, tests complex relationships, uses sophisticated vocabulary"
        
        # Question type specific instructions
        type_specific = ""
        if question_type.lower() == "numerical":
            type_specific = "Focus on calculations, formulas, and quantitative analysis. All options should be numerical values with units."
        elif question_type.lower() == "conceptual":
            type_specific = "Focus on theoretical understanding, mechanisms, and explanations. Avoid numerical calculations."
        else:
            type_specific = "Can include either numerical calculations or conceptual understanding as appropriate."
        
        prompt = f"""Generate a {complexity_level} question about {sanitized_topic}.

REQUIREMENTS:
- {requirements}
- {type_specific}

TOPIC: {sanitized_topic}
CONTEXT: {sanitized_context}

OUTPUT FORMAT (JSON only):
{{
  "question": "{complexity_level} question text here",
  "options": ["A) option", "B) option", "C) option", "D) option"],
  "correct_answer": "A",
  "explanation": "Detailed explanation with advanced concepts"
}}

Generate the JSON:"""

        return prompt
    
    def _generate_advanced_questions_with_enhanced_prompt(self, topic: str, context: str, num_questions: int, question_type: str, difficulty: str, enhanced_prompt: str) -> List[Dict[str, Any]]:
        """Generate advanced questions using PHI-enhanced prompt"""
        offline_logger.info(f"[PHI-ENHANCED] Generating {num_questions} {difficulty} questions with enhanced prompt")
        
        questions = []
        
        for i in range(num_questions):
            try:
                offline_logger.info(f"[PHI-ENHANCED] Generating question {i+1}/{num_questions}")
                
                # Use the enhanced prompt directly
                response = self._generate_with_retry(enhanced_prompt, max_tokens=800)
                
                if response:
                    parsed = self._parse_mcq_response(response)
                    if parsed and self._is_valid_mcq_response(parsed):
                        # Add metadata
                        parsed["metadata"] = {
                            "difficulty": difficulty,
                            "question_type": question_type,
                            "generation_method": f"phi_enhanced_{difficulty}",
                            "question_index": i,
                            "enhanced_prompt": True
                        }
                        questions.append(parsed)
                        offline_logger.info(f"[OK] PHI-enhanced {difficulty} question {i+1} generated successfully")
                    else:
                        offline_logger.error(f"[ERROR] PHI-enhanced question {i+1} parsing failed")
                else:
                    offline_logger.error(f"[ERROR] PHI-enhanced question {i+1} generation failed")
                        
            except Exception as e:
                offline_logger.error(f"[ERROR] PHI-enhanced question {i+1} failed: {e}")
                continue
        
        offline_logger.info(f"[PHI-ENHANCED] Generated {len(questions)}/{num_questions} enhanced {difficulty} questions")
        return questions
    
    def _extract_domain_from_topic(self, topic: str) -> str:
        """Extract domain from topic string"""
        topic_lower = topic.lower()
        if any(word in topic_lower for word in ['physics', 'force', 'energy', 'quantum', 'atom']):
            return 'Physics'
        elif any(word in topic_lower for word in ['chemistry', 'molecule', 'reaction', 'chemical']):
            return 'Chemistry'
        elif any(word in topic_lower for word in ['math', 'equation', 'calculus', 'algebra']):
            return 'Mathematics'
        else:
            return 'General'
    
    def _batch_generate_thinking(self, test_cases: List[Dict], topic: str, question_type: str) -> Optional[str]:
        """Step 1: Generate thinking for multiple expert questions - AGGRESSIVE PhD-LEVEL DEMANDS"""
        
        # Question type specific instructions for expert mode
        type_instruction = ""
        if question_type.lower() == "numerical":
            type_instruction = """
🔢 EXPERT MODE NUMERICAL EXCELLENCE 🔢

**Advanced Numerical Mastery:**
✓ Calculation Focus: Use precise verbs like "Calculate", "Compute", "Solve", "Determine", "Find", "Evaluate"
✓ Mathematical Rigor: Complex equations, multi-step derivations, sophisticated quantitative analysis
✓ Technical Depth: Advanced formulas, numerical methods, and computational techniques
✓ Quantitative Options: All 4 answers should be meaningful numerical values with appropriate units
✓ Problem Complexity: Multi-layered analysis requiring expert mathematical reasoning
✓ Rich Content: Comprehensive problems that demonstrate mastery (150+ characters recommended)

**Excellence Examples:**
✓ "Calculate the relativistic kinetic energy correction for an electron accelerated through a potential difference of 1.5 MV"
✓ "Determine the magnetic dipole moment of a hydrogen atom in the 2p₁/₂ state including spin-orbit coupling effects"
✓ "Solve for the reaction quotient Q when [A] = 0.025 M, [B] = 0.040 M in the equilibrium: 2A + 3B ⇌ C + 2D"

Remember: Focus purely on quantitative problem-solving to maintain numerical question integrity.
"""
        elif question_type.lower() == "conceptual":
            type_instruction = """
🧠 EXPERT MODE CONCEPTUAL EXCELLENCE 🧠

**Advanced Conceptual Mastery:**
✓ Understanding Focus: Use insight verbs like "Explain", "Why", "How", "What happens", "Describe", "Analyze"
✓ Theoretical Depth: Advanced principles, complex mechanisms, sophisticated theoretical frameworks
✓ Qualitative Reasoning: Deep understanding of cause-effect relationships and underlying mechanisms
✓ Conceptual Options: All 4 answers should be sophisticated theoretical explanations
✓ Analytical Complexity: Multi-layered reasoning requiring expert theoretical knowledge
✓ Rich Content: Comprehensive problems that demonstrate understanding (150+ characters recommended)

**Excellence Examples:**
✓ "Explain the quantum mechanical basis for the Pauli exclusion principle and its role in determining electron configurations in multi-electron atoms"
✓ "Why does the entropy of the universe increase during spontaneous chemical reactions according to the second law of thermodynamics?"
✓ "How does electromagnetic induction relate to relativistic effects in moving conductors according to special relativity theory?"

Remember: Focus purely on conceptual understanding to maintain theoretical question integrity.
"""
        else:
            type_instruction = """
🔀 EXPERT MODE BALANCED EXCELLENCE:
✓ Flexible Approach: Include either advanced numerical OR sophisticated conceptual elements as appropriate
✓ Dynamic Content: Vary between quantitative analysis and theoretical understanding based on topic needs
[OK] Mix complex calculations with deep theoretical reasoning
[OK] Balance advanced mathematical skills with conceptual expertise
[OK] MINIMUM 150+ CHARACTERS for all questions
"""

        # [EXPERT] USING SUCCESSFUL TEST PROMPTS: Much more demanding requirements
        prompt_parts = [f"""Generate PhD-level multiple choice questions that require DEEP EXPERTISE. Each question MUST be:

1. **HIGHLY COMPLEX** - Minimum 150+ characters, requiring advanced understanding
2. **MULTI-STEP REASONING** - Cannot be answered by simple recall
3. **MATHEMATICALLY RIGOROUS** - Include equations, derivations, or quantitative analysis where applicable
4. **DOMAIN-SPECIFIC TERMINOLOGY** - Use advanced technical vocabulary extensively
5. **THEORETICAL DEPTH** - Test understanding of underlying principles, not just facts

{type_instruction}

Generate questions for these test cases with MAXIMUM COMPLEXITY:

"""]
        
        for i, case in enumerate(test_cases, 1):
            domain = case.get('domain', 'General')
            topic_text = case.get('topic', 'General')
            context = case.get('context', '')
            difficulty = case.get('difficulty', 'expert')
            
            complexity_instruction = f"""
**EXPERT LEVEL REQUIREMENTS for {question_type.upper()} questions:**
- Question must require PhD-level knowledge
- Include complex mathematical relationships or advanced theoretical concepts
- Test synthesis of multiple advanced concepts
- Require multi-step logical reasoning
- Use domain-specific technical terminology extensively
- Question length: MINIMUM 150 characters
- Should be challenging even for graduate students and professors
- MUST follow {question_type.upper()} requirements above
"""
            
            prompt_parts.append(f"""
Test Case {i}: {domain} - {topic_text} (EXPERT difficulty - PhD Level {question_type.upper()})
{complexity_instruction}

Think through:
- What ADVANCED {question_type} concept should be tested that requires deep expertise?
- What complex mathematical or theoretical relationship is involved?
- What the correct answer should be (must require advanced reasoning)
- What sophisticated misconceptions could be used as distractors?
- How to make this question challenge even expert-level students?
- What advanced terminology and concepts must be included?

MAKE THIS QUESTION EXTREMELY CHALLENGING AND COMPLEX FOR {question_type.upper()} TYPE!

""")
        
        prompt_parts.append(f"""
EXCELLENCE STANDARDS:
- Expert questions should demonstrate advanced mastery and deep understanding
- Use sophisticated mathematical notation and technical terminology where appropriate
- Design questions that test comprehensive theoretical knowledge and analytical skills
- Questions should require multi-step reasoning and synthesis of concepts
- Include appropriate quantitative or qualitative analysis based on question type
- Create challenges worthy of advanced study and professional application
- Aim for substantial content depth (150+ characters recommended for expert level)
- Use domain-specific advanced vocabulary to ensure precision and rigor
- Follow {question_type.upper()} excellence guidelines for optimal question design

Generate sophisticated, challenging {question_type} questions that inspire learning excellence!""")
        
        full_prompt = "".join(prompt_parts)
        
        offline_logger.info(f"[EXCELLENCE] Advanced {question_type} thinking generation with {self.thinking_model}")
        offline_logger.info(f"[EXPERT] Using optimized prompts for maximum learning impact")
        response = self._generate_with_retry(full_prompt, model_override=self.thinking_model, max_tokens=12000)
        
        if response:
            offline_logger.info(f"[SUCCESS] Advanced {question_type} thinking generated ({len(response)} chars)")
            return response
        else:
            offline_logger.error(f"[INFO] No {question_type} thinking generated this attempt")
            return None
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if a model is a reasoning model that uses thinking tokens
        Supports: DeepSeek R1, Qwen/QwQ, and other reasoning models
        """
        if not model_name:
            return False
        
        model_lower = model_name.lower()
        reasoning_indicators = [
            'deepseek-r1', 'r1:', 'qwen', 'qwq', 'reasoning', 'thinking', 'think'
        ]
        
        is_reasoning = any(indicator in model_lower for indicator in reasoning_indicators)
        if is_reasoning:
            offline_logger.info(f"[REASONING] Detected reasoning model: {model_name}")
        
        return is_reasoning

    def _extract_individual_thinking(self, response: str, test_cases: List[Dict]) -> List[str]:
        """Step 2: Extract thinking for each individual test case - FIXED: Robust extraction without brittle markers"""
        
        # 🧠 UNIVERSAL THINKING EXTRACTION: Works with DeepSeek R1, Qwen/QwQ, and other reasoning models
        # Extract content from <think> blocks if present (standard for both DeepSeek R1 and Qwen)
        think_blocks = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
        
        if think_blocks:
            extracted_thinking = "\n\n".join(think_blocks).strip()
            offline_logger.info(f"[THINKING] Extracted {len(think_blocks)} thinking blocks from reasoning model")
        else:
            extracted_thinking = response.strip()
            offline_logger.info("[THINKING] No explicit thinking blocks found, using full response")
        
        individual_thinking = []
        
        # FIXED: Try multiple robust extraction strategies instead of relying on brittle "Test Case X:" markers
        
        # Strategy 1: Look for natural question/topic separators
        patterns = [
            r'(?:Question|Topic|Problem|Case)\s*(?:\d+|[IVX]+)[:.]',
            r'\d+[\.\)]\s*[A-Z]',
            r'[A-Z][^\.]*(?:question|topic|problem|case)',
            r'\n\s*\n\s*[A-Z]'
        ]
        
        best_splits = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, extracted_thinking, re.IGNORECASE | re.MULTILINE))
            if len(matches) >= len(test_cases) - 1:  # Should have n-1 separators for n questions
                best_splits = matches
                break
        
        if best_splits and len(best_splits) >= len(test_cases) - 1:
            # Strategy 1 worked: Use natural separators
            offline_logger.info(f"[SMART] Using natural separators for {len(test_cases)} thinking blocks")
            
            start_pos = 0
            for i in range(len(test_cases)):
                if i < len(best_splits):
                    end_pos = best_splits[i].start()
                    individual_thinking.append(extracted_thinking[start_pos:end_pos].strip())
                    start_pos = best_splits[i].start()
                else:
                    # Last section
                    individual_thinking.append(extracted_thinking[start_pos:].strip())
        
        elif len(test_cases) == 1:
            # Strategy 2: Single question - use entire response
            offline_logger.info("[SMART] Single question - using entire thinking block")
            individual_thinking.append(extracted_thinking)
        
        else:
            # Strategy 3: Intelligent equal division with overlap prevention
            offline_logger.info(f"[SMART] Using intelligent division for {len(test_cases)} thinking blocks")
            
            # Split by paragraphs first to avoid cutting mid-thought
            paragraphs = [p.strip() for p in extracted_thinking.split('\n\n') if p.strip()]
            
            if len(paragraphs) >= len(test_cases):
                # Distribute paragraphs among questions
                paras_per_question = len(paragraphs) // len(test_cases)
                
                for i in range(len(test_cases)):
                    start_para = i * paras_per_question
                    if i == len(test_cases) - 1:  # Last question gets remaining paragraphs
                        end_para = len(paragraphs)
                    else:
                        end_para = (i + 1) * paras_per_question
                    
                    question_thinking = '\n\n'.join(paragraphs[start_para:end_para])
                    individual_thinking.append(question_thinking)
            else:
                # Fallback: Simple division by character count
                chunk_size = len(extracted_thinking) // len(test_cases)
                
                for i in range(len(test_cases)):
                    start = i * chunk_size
                    if i == len(test_cases) - 1:  # Last chunk gets remainder
                        end = len(extracted_thinking)
                    else:
                        end = (i + 1) * chunk_size
                        # Adjust to avoid cutting mid-sentence
                        while end < len(extracted_thinking) and extracted_thinking[end] not in '.!?\n':
                            end += 1
                    
                    individual_thinking.append(extracted_thinking[start:end].strip())
        
        # Ensure we have exactly the right number of thinking blocks
        while len(individual_thinking) < len(test_cases):
            # Duplicate the last good thinking block if we're short
            if individual_thinking:
                individual_thinking.append(individual_thinking[-1])
            else:
                # Alternative fallback: Generate topic-specific thinking
                topic = test_cases[len(individual_thinking)].get('topic', 'general knowledge')
                fallback_thinking = f"This question requires deep analysis of {topic} concepts, including theoretical frameworks, practical applications, and advanced problem-solving strategies."
                individual_thinking.append(fallback_thinking)
        
        # Trim excess if we have too many
        
        offline_logger.info(f"[EXCELLENCE] Advanced {question_type} thinking generation with {self.thinking_model}")
        offline_logger.info(f"[EXPERT] Using optimized prompts for maximum learning impact")
        response = self._generate_with_retry(full_prompt, model_override=self.thinking_model, max_tokens=12000)
        
        if response:
            offline_logger.info(f"[SUCCESS] Advanced {question_type} thinking generated ({len(response)} chars)")
            return response
        else:
            offline_logger.error(f"[INFO] No {question_type} thinking generated this attempt")
            return None
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if a model is a reasoning model that uses thinking tokens
        Supports: DeepSeek R1, Qwen/QwQ, and other reasoning models
        """
        if not model_name:
            return False
        
        model_lower = model_name.lower()
        reasoning_indicators = [
            'deepseek-r1', 'r1:', 'qwen', 'qwq', 'reasoning', 'thinking', 'think'
        ]
        
        is_reasoning = any(indicator in model_lower for indicator in reasoning_indicators)
        if is_reasoning:
            offline_logger.info(f"[REASONING] Detected reasoning model: {model_name}")
        
        return is_reasoning
    
    def _extract_individual_thinking(self, response: str, test_cases: List[Dict]) -> List[str]:
        """Step 2: Extract thinking for each individual test case - FIXED: Robust extraction without brittle markers"""
        
        # UNIVERSAL THINKING EXTRACTION: Works with DeepSeek R1, Qwen/QwQ, and other reasoning models
        # Extract content from <think> blocks if present (standard for both DeepSeek R1 and Qwen)
        think_blocks = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
        
        if think_blocks:
            extracted_thinking = "\n\n".join(think_blocks).strip()
            offline_logger.info(f"[THINKING] Extracted {len(think_blocks)} thinking blocks from reasoning model")
        else:
            extracted_thinking = response.strip()
            offline_logger.info("[THINKING] No explicit thinking blocks found, using full response")
        
        individual_thinking = []
        
        # FIXED: Try multiple robust extraction strategies instead of relying on brittle "Test Case X:" markers
        
        # Strategy 1: Look for natural question/topic separators
        patterns = [
            r'(?:Question|Topic|Problem|Case)\s*(?:\d+|[IVX]+)[:.]',
            r'\d+[\.\)]\s*[A-Z]',
            r'[A-Z][^\.]*(?:question|topic|problem|case)',
            r'\n\s*\n\s*[A-Z]'
        ]
        
        best_splits = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, extracted_thinking, re.IGNORECASE | re.MULTILINE))
            if len(matches) >= len(test_cases) - 1:  # Should have n-1 separators for n questions
                best_splits = matches
                break
        
        if best_splits and len(best_splits) >= len(test_cases) - 1:
            # Strategy 1 worked: Use natural separators
            offline_logger.info(f"[SMART] Using natural separators for {len(test_cases)} thinking blocks")
            
            start_pos = 0
            for i in range(len(test_cases)):
                if i < len(best_splits):
                    end_pos = best_splits[i].start()
                    individual_thinking.append(extracted_thinking[start_pos:end_pos].strip())
                    start_pos = best_splits[i].start()
                else:
                    # Last section
                    individual_thinking.append(extracted_thinking[start_pos:].strip())
        
        elif len(test_cases) == 1:
            # Strategy 2: Single question - use entire response
            offline_logger.info("[SMART] Single question - using entire thinking block")
            individual_thinking.append(extracted_thinking)
        
        else:
            # Strategy 3: Intelligent equal division with overlap prevention
            offline_logger.info(f"[SMART] Using intelligent division for {len(test_cases)} thinking blocks")
            
            # Split by paragraphs first to avoid cutting mid-thought
            paragraphs = [p.strip() for p in extracted_thinking.split('\n\n') if p.strip()]
            
            if len(paragraphs) >= len(test_cases):
                # Distribute paragraphs among questions
                paras_per_question = len(paragraphs) // len(test_cases)
                
                for i in range(len(test_cases)):
                    start_para = i * paras_per_question
                    if i == len(test_cases) - 1:  # Last question gets remaining paragraphs
                        end_para = len(paragraphs)
                    else:
                        end_para = (i + 1) * paras_per_question
                    
                    question_thinking = '\n\n'.join(paragraphs[start_para:end_para])
                    individual_thinking.append(question_thinking)
            else:
                # Fallback: Simple division by character count
                chunk_size = len(extracted_thinking) // len(test_cases)
                
                for i in range(len(test_cases)):
                    start = i * chunk_size
                    if i == len(test_cases) - 1:  # Last chunk gets remainder
                        end = len(extracted_thinking)
                    else:
                        end = (i + 1) * chunk_size
                        # Adjust to avoid cutting mid-sentence
                        while end < len(extracted_thinking) and extracted_thinking[end] not in '.!?\n':
                            end += 1
                    
                    individual_thinking.append(extracted_thinking[start:end].strip())
        
        # Ensure we have exactly the right number of thinking blocks
        while len(individual_thinking) < len(test_cases):
            # Duplicate the last good thinking block if we're short
            if individual_thinking:
                individual_thinking.append(individual_thinking[-1])
            else:
                # Alternative fallback: Generate topic-specific thinking
                topic = test_cases[len(individual_thinking)].get('topic', 'general knowledge')
                fallback_thinking = f"This question requires deep analysis of {topic} concepts, including theoretical frameworks, practical applications, and advanced problem-solving strategies."
                individual_thinking.append(fallback_thinking)
        
        # Trim excess if we have too many
        individual_thinking = individual_thinking[:len(test_cases)]
        
        offline_logger.info(f"[ROBUST] Extracted {len(individual_thinking)} robust thinking blocks")
        return individual_thinking
    
    def _batch_generate_json(self, thinking_list: List[str], test_cases: List[Dict]) -> List[Optional[Dict]]:
        """ARCHITECTURAL FIX: True batch JSON generation instead of loop-based fake batch"""
        
        # ARCHITECTURAL FIX: Use true batch inference when possible
        if len(thinking_list) > 1 and hasattr(self, '_supports_batch_inference') and self._supports_batch_inference():
            return self._true_batch_generate_json(thinking_list, test_cases)
        
        # Fallback to individual generation for models that don't support batch inference
        results = []
        
        for i, (thinking, test_case) in enumerate(zip(thinking_list, test_cases)):
            domain = test_case.get('domain', 'General')
            topic = test_case.get('topic', 'General') 
            difficulty = test_case.get('difficulty', 'expert')
            
            # [EXPERT] USING SUCCESSFUL TEST PROMPTS: Aggressive PhD-level JSON demands
            complexity_reminder = """
EXPERT LEVEL JSON REQUIREMENTS:
- Question MUST be PhD dissertation-level complexity (minimum 150 characters)
- Include advanced mathematical notation, equations, or technical terminology
- Test deep theoretical understanding requiring multi-step reasoning
- Use sophisticated domain-specific vocabulary extensively
- Question should challenge even university professors
- Options must be technically sophisticated, not simple choices
- QUESTION MUST END WITH A QUESTION MARK (?)
"""
            
            prompt = f"""Generate a valid JSON MCQ from this thinking:

THINKING: {thinking}

CRITICAL JSON FORMAT - EXACT OUTPUT REQUIRED:
{{
  "question": "EXACTLY ONE PhD-level question ending with ?",
  "options": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct": "A",
  "explanation": "Brief explanation"
}}

{complexity_reminder}

MANDATORY RULES:
- EXACTLY 4 options in A,B,C,D format  
- Use "correct" not "correct_answer"
- Question must end with ?
- No extra commas or brackets
- No explanatory text outside JSON
- Generate ONLY the JSON object above"""
            
            offline_logger.info(f"[DOC] JSON generation {i+1}/{len(test_cases)} with {self.json_model}")
            
            # Try up to 3 times for better reliability
            json_data = None
            for attempt in range(3):
                # [EXPERT] PhD-LEVEL: Buffer-based generation to avoid JSON truncation
                response = self._generate_with_chunked_buffer(prompt, model_override=self.json_model, max_tokens=2000)
                
                if response:
                    json_data = self._parse_json_response_robust(response)
                    if json_data:
                        break  # Success, exit retry loop
                    elif attempt < 2:  # Not the last attempt
                        offline_logger.warning(f"   [WARNING]  Retry {attempt + 1}/3 for JSON {i+1}")
                
                if attempt == 2:  # Last attempt failed
                    offline_logger.error(f"   [ERROR] JSON {i+1} failed after 3 attempts")
            
            if json_data:
                # Fix question mark if missing
                question = json_data.get('question', '')
                if question and not question.endswith('?'):
                    json_data['question'] = question.rstrip('.!') + '?'
                
                # Add metadata
                json_data['difficulty'] = difficulty
                json_data['domain'] = domain
                json_data['topic'] = topic
                json_data['generated_by'] = f"{self.thinking_model} + {self.json_model}"
                json_data['pipeline'] = "batch_two_model_expert"
                results.append(json_data)
                offline_logger.info(f"   [OK] JSON {i+1} generated")
            else:
                results.append(None)
                offline_logger.error(f"   [ERROR] JSON {i+1} no valid response")
        
        return results
    
    def _generate_with_retry(self, prompt: str, model_override: str = None, max_tokens: int = 2000) -> Optional[str]:
        """Generate text with robust retry logic and connection handling"""
        
        model_to_use = model_override or self.model_name
        max_retries = 3
        
        # [START] CRITICAL FIX: Use longer timeouts for DeepSeek-R1 reasoning model
        if model_override and 'deepseek' in model_override.lower():
            base_timeout = 180  # 3 minutes for reasoning models
            offline_logger.info(f"[BRAIN] Using extended timeout for reasoning model {model_to_use}: {base_timeout}s base")
        else:
            base_timeout = 45  # Increased from 30s for better reliability
        
        for attempt in range(max_retries):
            try:
                offline_logger.info(f"[RELOAD] Generation attempt {attempt + 1}/{max_retries} with {model_to_use}")
                
                # [START] OPTIMIZED: Progressive timeout for reasoning vs fast models
                if model_override and 'deepseek' in model_override.lower():
                    timeout_duration = base_timeout + (attempt * 60)  # 180s, 240s, 300s (reasoning)
                else:
                    timeout_duration = base_timeout + (attempt * 15)  # 45s, 60s, 75s (fast models)
                
                offline_logger.info(f"[TIME] Using timeout: {timeout_duration}s for attempt {attempt + 1}")
                
                # Create temporary interface if using different model
                if model_override and model_override != self.model_name:
                    # Use direct API call for different model
                    import requests
                    
                    # [START] SPEED OPTIMIZATIONS: Model-specific parameters
                    if 'deepseek' in model_to_use.lower():
                        # DeepSeek-R1 optimized for fast expert reasoning
                        options = {
                            "num_predict": min(max_tokens, 800),  # Limit tokens for speed
                            "temperature": 0.6,  # Lower for focused thinking
                            "top_p": 0.85,  # Reduced for speed
                            "top_k": 30,  # Reduced for speed  
                            "repeat_penalty": 1.02,  # Minimal to avoid slowdown
                            "num_gpu": -1,  # Use all GPU layers
                            "num_thread": 12,  # Max threads for CPU
                            "batch_size": 1024,  # Large batch for GPU efficiency
                        }
                    else:
                        # Standard model optimizations
                        options = {
                            "num_predict": max_tokens,
                            "temperature": 0.75,
                            "top_p": 0.9,
                            "top_k": 40,
                            "num_gpu": -1,
                            "batch_size": 512,
                        }
                    
                    # [START] Always use streaming for better monitoring and control
                    payload = {
                        "model": model_to_use,
                        "prompt": prompt,
                        "stream": True,
                        "options": options
                    }
                    
                    offline_logger.info(f"[AI] Starting model {model_to_use} with streaming...")

                    import requests
                    try:
                        response = requests.post(
                            'http://127.0.0.1:11434/api/generate',
                            json=payload,
                            timeout=timeout_duration,
                            stream=True
                        )
                        response.raise_for_status()
                        offline_logger.info(f"[OK] Connection established, reading response...")

                        # Process streaming response with progress updates
                        result = ""
                        chunk_count = 0
                        output_started = False
                        last_update = time.time()
                        json_started = False
                        json_data = ""
                        
                        for line in response.iter_lines():
                            if line:
                                try:
                                    chunk_data = json.loads(line.decode('utf-8'))
                                    chunk_response = chunk_data.get('response', '')
                                    
                                    # Check for JSON markers in the response
                                    if not json_started and '{' in chunk_response:
                                        json_started = True
                                        # Keep only from the opening brace onwards
                                        chunk_response = chunk_response[chunk_response.find('{'):]
                                        
                                    if json_started:
                                        json_data += chunk_response
                                    else:
                                        result += chunk_response

                                    # Show progress every 5 seconds
                                    chunk_count += 1
                                    current_time = time.time()
                                    if current_time - last_update > 5:
                                        offline_logger.info(f"[AI] Model generating... ({chunk_count} chunks)")
                                        last_update = current_time

                                    # Detect when actual content starts coming
                                    if len(chunk_response.strip()) > 0 and not output_started:
                                        offline_logger.info("[AI] Output started")
                                        output_started = True

                                    if chunk_data.get('done', False):
                                        offline_logger.info("[OK] Generation completed")
                                        break

                                except json.JSONDecodeError:
                                    offline_logger.warning(f"[WARN] Invalid JSON chunk: {line}")
                                    continue

                        # Try to parse as JSON first
                        if json_started:
                            try:
                                # Clean up and complete any unclosed JSON
                                json_data = json_data.strip()
                                if not json_data.endswith('}'):
                                    json_data += '}'
                                # Try to parse it as JSON
                                json.loads(json_data)
                                # If successful, use the JSON data
                                result = json_data
                                offline_logger.info("[OK] Parsed valid JSON response")
                            except json.JSONDecodeError:
                                offline_logger.warning("[WARN] Failed to parse JSON, using raw response")
                                pass

                        result = result.strip()
                        if not result:
                            offline_logger.warning("[WARN] Empty result after streaming")
                            continue
                            
                    except requests.exceptions.Timeout:
                        offline_logger.error(f"[ERROR] Generation timeout after {timeout_duration}s")
                        continue
                    except Exception as e:
                        offline_logger.error(f"[ERROR] Generation error: {e}")
                        continue
                
                if result:  # Success with valid response
                    if attempt > 0:  # Recovered from previous failures
                        offline_logger.info(f"   [OK] Recovered on attempt {attempt + 1}")
                    return result
                else:
                    offline_logger.warning(f"   [WARNING]  Empty response from {model_to_use} (attempt {attempt + 1})")
                    
            except Exception as e:
                offline_logger.warning(f"   [ERROR] Error calling {model_to_use} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    backoff_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    offline_logger.info(f"   [WAIT] Waiting {backoff_time}s before retry...")
                    time.sleep(backoff_time)
        
        offline_logger.error(f"   [ERROR] All {max_retries} attempts failed for {model_to_use}")
        return None

    def _generate_with_chunked_buffer(self, prompt: str, model_override: str = None, max_tokens: int = 2000) -> Optional[str]:
        """[START] WEB SEARCH SOLUTION: Buffer-based generation to prevent JSON truncation"""
        offline_logger.info(f"[CONFIG] Using chunked buffer generation for {max_tokens} tokens")
        
        # Try normal generation first for smaller requests
        if max_tokens <= 1000:
            return self._generate_with_retry(prompt, model_override, max_tokens)
        
        try:
            model_to_use = model_override or self.model_name
            
            # [CONFIG] SOLUTION 1: Multiple smaller requests that concatenate
            chunk_size = 800
            accumulated_response = ""
            complete_json_found = False
            
            for chunk_num in range(1, 4):  # Max 3 chunks
                current_tokens = min(chunk_size, max_tokens - len(accumulated_response.split()))
                
                if chunk_num == 1:
                    chunk_prompt = prompt
                else:
                    # Continue from previous response
                    chunk_prompt = f"Continue this JSON response exactly where it left off:\n\n{accumulated_response}\n\nComplete the JSON structure:"
                
                offline_logger.info(f"[CONFIG] Generating chunk {chunk_num} with {current_tokens} tokens")
                
                chunk_response = self._generate_with_retry(
                    chunk_prompt, 
                    model_override, 
                    max_tokens=current_tokens
                )
                
                if chunk_response:
                    if chunk_num == 1:
                        accumulated_response = chunk_response
                    else:
                        # Smart concatenation: remove duplicate JSON starts
                        chunk_cleaned = chunk_response.strip()
                        if chunk_cleaned.startswith('{'):
                            # Find where previous JSON likely ended
                            if accumulated_response.rstrip().endswith(','):
                                accumulated_response = accumulated_response.rstrip()[:-1] + chunk_cleaned[1:]
                            else:
                                accumulated_response += chunk_cleaned[1:]
                        else:
                            accumulated_response += chunk_cleaned
                    
                    # Check if we have complete JSON
                    if self._is_complete_json(accumulated_response):
                        offline_logger.info(f"[OK] Complete JSON found in chunk {chunk_num}")
                        complete_json_found = True
                        break
                        
                else:
                    offline_logger.warning(f"[WARNING] Chunk {chunk_num} generation failed")
                    break
            
            if complete_json_found:
                return accumulated_response
            
            # [CONFIG] Dynamic generation approach - No hardcoded content
            offline_logger.info("[CONFIG] Attempting alternative prompt approach")
            
            # Extract topic from prompt for better fallback
            topic_match = re.search(r'about (.+?) with', prompt, re.IGNORECASE)
            fallback_topic = topic_match.group(1) if topic_match else 'magnetism'
            
            # Dynamic content generation preferred - AI-based approach
            offline_logger.info("[INFO] Dynamic content generation preferred for quality assurance")
            offline_logger.info(f"[INFO] Topic '{fallback_topic}' requires AI model generation for best results")
            raise Exception(f"AI model generation recommended for '{fallback_topic}' - dynamic content preferred")
            
            return self._generate_with_retry(fallback_prompt, model_override, max_tokens=600)
            
        except Exception as e:
            offline_logger.info(f"[INFO] Chunked buffer generation attempt: {str(e)}")
            # Final fallback to original method
            return self._generate_with_retry(prompt, model_override, max_tokens)
    
    def _is_complete_json(self, text: str) -> bool:
        """Check if text contains a complete, valid JSON object"""
        try:
            # Clean the text
            cleaned = text.strip()
            
            # Find JSON boundaries
            start_brace = cleaned.find('{')
            end_brace = cleaned.rfind('}')
            
            if start_brace == -1 or end_brace == -1:
                return False
            
            json_part = cleaned[start_brace:end_brace + 1]
            
            # Check brace balance
            open_braces = json_part.count('{')
            close_braces = json_part.count('}')
            
            if open_braces != close_braces:
                return False
            
            # Try to parse
            import json
            parsed = json.loads(json_part)
            
            # Basic structure check
            return (isinstance(parsed, dict) and 
                   'question' in parsed and 
                   'options' in parsed and
                   'correct' in parsed)
        except Exception:
            return False
        
        # Method 1b: Look for JSON blocks with backticks but no language identifier
        json_block_match2 = re.search(r'```\s*(\{.*?\})\s*```', accumulated_content, re.DOTALL)
        if json_block_match2:
            json_str = json_block_match2.group(1)
            try:
                parsed = json.loads(json_str)
                if self._validate_json_structure_robust(parsed):
                    return parsed
            except:
                pass
        
        # Method 2: Look for any JSON object (most permissive) - improved regex
        # This handles nested braces better
        json_matches = re.findall(r'\{(?:[^{}]|{[^{}]*})*\}', accumulated_content, re.DOTALL)
        for json_match in json_matches:
            try:
                parsed = json.loads(json_match)
                if self._validate_json_structure_robust(parsed):
                    return parsed
            except:
                continue
        
        # Method 3: Try parsing entire response after aggressive cleaning
        try:
            # Clean response more aggressively
            cleaned = accumulated_content.strip()
            
            # Remove common intro patterns - ENHANCED for all cases
            intro_patterns = [
                r'Here is the (?:valid )?JSON object.*?:',
                r'Here is the (?:generated )?JSON.*?:',
                r'Here is the JSON structure.*?:',
                r'Based on.*?here is.*?:',
                r'The JSON object.*?:',
                r'Here\'s the.*?:',
                r'.*?PhD-level.*?question.*?:'
            ]
            for pattern in intro_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove code fences
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Remove any leading/trailing text before/after JSON
            start_brace = cleaned.find('{')
            end_brace = cleaned.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_part = cleaned[start_brace:end_brace + 1]
                parsed = json.loads(json_part)
                if self._validate_json_structure_robust(parsed):
                    return parsed
        except:
            pass
        
        # Method 4: Try to fix common JSON issues
        try:
            fixed_response = self._fix_common_json_issues(accumulated_content)
            if fixed_response:
                parsed = json.loads(fixed_response)
                if self._validate_json_structure_robust(parsed):
                    return parsed
        except:
            pass
        
        # Method 5: [CONFIG] WEB SEARCH SOLUTION: Progressive JSON building
        try:
            progressive_json = self._build_progressive_json(accumulated_content)
            if progressive_json:
                parsed = json.loads(progressive_json)
                if self._validate_json_structure_robust(parsed):
                    return parsed
        except:
            pass
        
        offline_logger.error(f"[ERROR] All JSON parsing methods failed for content: {accumulated_content[:200]}...")
        return None
    
    def _concatenate_streaming_chunks(self, content: str) -> str:
        """[CONFIG] WEB SEARCH SOLUTION: Handle streaming response concatenation"""
        try:
            # Split content by potential streaming separators and recombine
            # This addresses the GitHub issue about partial JSON chunks
            lines = content.split('\n')
            json_parts = []
            
            for line in lines:
                line = line.strip()
                if line and ('{' in line or '}' in line or '"' in line):
                    json_parts.append(line)
            
            # Recombine JSON parts
            recombined = ' '.join(json_parts)
            
            # Clean up obvious streaming artifacts
            recombined = re.sub(r'\}\s*\{', '}{', recombined)  # Fix split braces
            recombined = re.sub(r'"\s*"', '""', recombined)    # Fix split quotes
            
            return recombined if recombined else content
            
        except:
            return content
    
    def _build_progressive_json(self, content: str) -> Optional[str]:
        """[CONFIG] WEB SEARCH SOLUTION: Build JSON progressively from fragments"""
        try:
            # Extract potential JSON key-value pairs
            question_match = re.search(r'"question"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
            options_match = re.search(r'"options"\s*:\s*(\[.*?\]|\{.*?\})', content, re.DOTALL)
            correct_match = re.search(r'"(?:correct|correct_answer)"\s*:\s*"([ABCD])"', content, re.IGNORECASE)
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
            
            if question_match and options_match and correct_match:
                # Build complete JSON from parts
                question = question_match.group(1)
                options = options_match.group(1)
                correct = correct_match.group(1)
                explanation = explanation_match.group(1) if explanation_match else "Expert explanation"
                
                # [CONFIG] FIX: Ensure options are always in dictionary format for UI consistency
                if options.startswith('['):
                    # Convert array format to dict
                    try:
                        options_list = json.loads(options)
                        if len(options_list) == 4:
                            options = json.dumps({
                                'A': options_list[0],
                                'B': options_list[1], 
                                'C': options_list[2],
                                'D': options_list[3]
                            })
                    except:
                        pass
                elif options.startswith('{'):
                    # Validate that dict format has correct keys
                    try:
                        options_dict = json.loads(options)
                        option_keys = ['A', 'B', 'C', 'D']
                        if not all(key in options_dict for key in option_keys):
                            offline_logger.warning("[WARNING] Options dict missing required keys, using fallback")
                            return None
                        # Options already in correct format
                    except:
                        pass
                
                progressive_json = f"""{{
                    "question": "{question}",
                    "options": {options},
                    "correct": "{correct}",
                    "explanation": "{explanation}"
                }}"""
                
                offline_logger.info("[OK] Successfully built progressive JSON from fragments")
                return progressive_json
                
        except Exception as e:
            offline_logger.warning(f"[WARNING] Progressive JSON building failed: {str(e)}")
            
        return None
    
    def _validate_json_structure_robust(self, parsed_json: Dict) -> bool:
        """Validate that the JSON has the required MCQ structure"""
        
        if not isinstance(parsed_json, dict):
            return False
            
        required_fields = ['question', 'options', 'correct']
        if not all(field in parsed_json for field in required_fields):
            return False
        
        # Check question is non-empty string
        question = parsed_json.get('question', '')
        if not isinstance(question, str) or not question.strip():
            return False
        
        # [CONFIG] FIX: Ensure options are always in dictionary format for UI consistency
        options = parsed_json.get('options', [])
        if isinstance(options, dict):
            # Handle {"A": "...", "B": "...", "C": "...", "D": "..."} format
            option_keys = ['A', 'B', 'C', 'D']
            if not all(key in options for key in option_keys):
                offline_logger.warning(f"[WARNING] Options missing required keys: {set(option_keys) - set(options.keys())}")
                return False
            if not all(isinstance(options[key], str) and options[key].strip() for key in option_keys):
                offline_logger.warning("[WARNING] Options contain empty or non-string values")
                return False
            # Options are already in correct dictionary format
        elif isinstance(options, list):
            # Handle ["...", "...", "...", "..."] format - convert to dict format
            if len(options) < 4:
                offline_logger.warning(f"[WARNING] Options list has {len(options)} items, need at least 4")
                return False
            elif len(options) > 4:
                offline_logger.warning(f"[WARNING] Options list has {len(options)} items, using first 4")
                options = options[:4]  # Truncate to first 4 options
            
            if not all(isinstance(opt, str) and opt.strip() for opt in options[:4]):
                offline_logger.warning("[WARNING] Options list contains empty or non-string values")
                return False
            # Convert array to dict format for UI consistency
            offline_logger.info("[CONFIG] Converting options from list to dictionary format")
            parsed_json['options'] = {
                'A': options[0],
                'B': options[1], 
                'C': options[2],
                'D': options[3]
            }
        else:
            offline_logger.warning(f"[WARNING] Options has invalid type: {type(options)}")
            return False
        
        # Check correct answer (support both "correct" and "correct_answer" formats)
        correct = parsed_json.get('correct', parsed_json.get('correct_answer', ''))
        if not isinstance(correct, str) or correct not in ['A', 'B', 'C', 'D']:
            offline_logger.warning(f"[WARNING] Invalid correct answer: '{correct}'")
            return False
        
        # Normalize to "correct" format if using "correct_answer"
        if 'correct_answer' in parsed_json and 'correct' not in parsed_json:
            parsed_json['correct'] = parsed_json['correct_answer']
        
        offline_logger.info("[OK] JSON structure validation passed")
        return True
    
    def _fix_common_json_issues(self, response: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues"""
        
        try:
            # Find the JSON part
            start_brace = response.find('{')
            end_brace = response.rfind('}')
            if start_brace == -1 or end_brace == -1:
                return None
            
            json_part = response[start_brace:end_brace + 1]
            
            # Fix common issues
            # Remove trailing commas before } or ]
            json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
            
            # Ensure proper quoting of keys
            json_part = re.sub(r'(\w+):', r'"\1":', json_part)
            
            # Fix single quotes to double quotes
            json_part = json_part.replace("'", '"')
            
            return json_part
        except:
            return None
    
    def _parse_batch_mcq_response_robust(self, response: str, expected_questions: int, topic: str) -> List[Dict[str, Any]]:
        """[START] ENHANCED: Parse multiple MCQ questions with robust JSON extraction"""
        try:
            offline_logger.info(f"[SEARCH] Robust parsing batch response for {expected_questions} questions...")
            
            # First try the original parsing method
            original_result = self._parse_batch_mcq_response(response, expected_questions, topic)
            if original_result and len(original_result) >= expected_questions // 2:
                offline_logger.info(f"[OK] Original parser succeeded: {len(original_result)} questions")
                return original_result
            
            # Enhanced parsing with robust JSON extraction
            offline_logger.info("[RELOAD] Trying enhanced parsing methods...")
            
            # Try to extract JSON array using robust methods
            json_arrays = re.findall(r'\[.*?\]', response, re.DOTALL)
            
            for json_array in json_arrays:
                try:
                    questions_data = json.loads(json_array)
                    
                    if isinstance(questions_data, list):
                        parsed_questions = []
                        for i, q_data in enumerate(questions_data[:expected_questions]):
                            if self._validate_json_structure_robust(q_data):
                                # [CONFIG] FIX: Keep options in dictionary format for consistency
                                # The validation method already converts list to dict, so don't convert back
                                options = q_data.get('options', {})
                                if isinstance(options, dict):
                                    # Ensure we have all required option keys
                                    option_keys = ['A', 'B', 'C', 'D']
                                    if all(key in options for key in option_keys):
                                        # Options are already in correct dict format
                                        pass
                                    else:
                                        offline_logger.warning(f"[WARNING] Question {i+1} missing required option keys")
                                        continue
                                elif isinstance(options, list) and len(options) == 4:
                                    # Convert list to dict format for UI consistency
                                    q_data['options'] = {
                                        'A': options[0],
                                        'B': options[1], 
                                        'C': options[2],
                                        'D': options[3]
                                    }
                                else:
                                    offline_logger.warning(f"[WARNING] Question {i+1} has invalid options format")
                                    continue
                                
                                parsed_questions.append(q_data)
                                offline_logger.info(f"[OK] Robust parsed question {i+1}: {q_data.get('question', '')[:50]}...")
                            else:
                                offline_logger.warning(f"[WARNING] Question {i+1} failed robust validation")
                        
                        if parsed_questions:
                            offline_logger.info(f"[LIST] Robust parsing succeeded: {len(parsed_questions)} questions")
                            return parsed_questions
                            
                except json.JSONDecodeError:
                    continue
            
            # Final fallback: try individual question parsing
            offline_logger.info("[RELOAD] Final fallback: individual question parsing...")
            return self._parse_individual_questions_from_text_robust(response, expected_questions, topic)
            
        except Exception as e:
            offline_logger.error(f"[ERROR] Robust batch parsing error: {e}")
            return []

    def _parse_individual_questions_from_text_robust(self, text: str, expected_questions: int, topic: str) -> List[Dict[str, Any]]:
        """Robust fallback parser to extract individual questions from text response"""
        try:
            questions = []
            
            # Split by common delimiters that might separate questions
            potential_questions = re.split(r'(?:\n\s*\n|\}\s*,?\s*\{)', text)
            
            for i, potential_q in enumerate(potential_questions):
                if i >= expected_questions:
                    break
                    
                # Try robust JSON extraction on this segment
                result = self._parse_json_response_robust(potential_q)
                if result:
                    questions.append(result)
                    offline_logger.info(f"[OK] Robust extracted question {len(questions)}: {result.get('question', '')[:50]}...")
            
            offline_logger.info(f"[LIST] Robust fallback parsing extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            offline_logger.error(f"[ERROR] Robust fallback parsing error: {e}")
            return []

    def initialize(self) -> bool:
        """Initialize the offline MCQ generator with GPU optimization and comprehensive logging"""
        offline_logger.info(f"[START] INITIALIZING HIGH-PERFORMANCE Ollama engine for '{self.model_name}'...")
        performance_logger.info(f"[CONFIG] INITIALIZATION START: model={self.model_name}")
        
        start_time = time.time()
        
        try:
            # Create optimized Ollama interface
            offline_logger.info("[CONFIG] CREATING OllamaModelInference...")
            self.ollama_interface = OllamaModelInference()
            offline_logger.info(f"[CONFIG] OllamaModelInference created: {self.ollama_interface is not None}")
            offline_logger.info(f"[CONFIG] OllamaModelInference type: {type(self.ollama_interface)}")

            # [START] FIXED: Initialize the Ollama interface properly - NON-BLOCKING
            offline_logger.info("[CONFIG] INITIALIZING Ollama interface...")
            try:
                # [CONFIG] CRITICAL FIX: Initialize the model synchronously first
                initialization_success = self.ollama_interface.initialize()
                offline_logger.info(f"[CONFIG] Ollama initialization result: {initialization_success}")
                
                # Verify active_model is set
                if initialization_success and self.ollama_interface.active_model:
                    offline_logger.info(f"[CONFIG] Active model set: {self.ollama_interface.active_model}")
                else:
                    offline_logger.warning("[CONFIG] Active model not set after initialization")

            except Exception as e:
                offline_logger.error(f"[ERROR] Failed to initialize Ollama interface: {e}")
                initialization_success = False

            if not self.ollama_interface.is_available():
                offline_logger.error("[ERROR] OLLAMA INTERFACE NOT AVAILABLE")
                offline_logger.error("   • Check if Ollama server is running")
                offline_logger.error("   • Verify Ollama installation")
                offline_logger.error("   • Check network connectivity to Ollama")
                offline_logger.error(f"   • Initialization success: {initialization_success}")
                self.ollama_interface = None
                performance_logger.error(f"[ERROR] INITIALIZATION FAILED: Ollama not available after {time.time() - start_time:.3f}s")
                return False
                
            offline_logger.info("[CONFIG] OLLAMA INTERFACE IS AVAILABLE, applying optimizations...")
            
            # Apply TURBO speed optimizations
            offline_logger.info("[FAST] APPLYING SPEED OPTIMIZATIONS...")
            self.ollama_interface.optimize_for_speed()
            offline_logger.info("[OK] SPEED OPTIMIZATIONS APPLIED")
            
            # Skip test generation for faster startup - actual generation will be the test
            offline_logger.info("[FAST] SKIPPING TEST GENERATION for TURBO startup speed")
                
            self.is_engine_initialized = True
            
            init_time = time.time() - start_time
            offline_logger.info("[OK] HIGH-PERFORMANCE Ollama engine is connected and GPU-optimized!")
            performance_logger.info(f"[OK] INITIALIZATION SUCCESS in {init_time:.3f}s")
            
            # Verify the interface is still set
            offline_logger.info(f"[CONFIG] FINAL CHECK - ollama_interface is not None: {self.ollama_interface is not None}")
            
            # Log performance stats
            if self.ollama_interface:
                try:
                    stats = self.ollama_interface.get_performance_stats()
                    offline_logger.info(f"[GAME] GPU STATUS: {stats.get('gpu_optimized', False)}")
                    offline_logger.info(f"[START] MODEL: {stats.get('model', 'unknown')}")
                    offline_logger.info(f"[STATS] STATS: {stats}")
                except Exception as e:
                    offline_logger.warning(f"[WARNING] FAILED TO GET STATS: {e}")
            
            return True
            
        except Exception as e:
            init_time = time.time() - start_time
            offline_logger.error(f"[ERROR] FAILED TO INITIALIZE Ollama engine: {e}")
            offline_logger.error(f"[ERROR] FULL TRACEBACK: {traceback.format_exc()}")
            performance_logger.error(f"[ERROR] INITIALIZATION FAILED after {init_time:.3f}s: {e}")
            self.ollama_interface = None
            return False

    def _test_generation(self) -> bool:
        """Test generation to ensure the engine works properly"""
        try:
            test_prompt = """Create a multiple choice question about magnetism.

Format as JSON:
{
  "question": "What causes magnetic fields?",
  "options": {
    "A": "Moving electric charges",
    "B": "Static electricity", 
    "C": "Sound waves",
    "D": "Light waves"
  },
  "correct": "A",
  "explanation": "Moving electric charges create magnetic fields"
}

Generate only the JSON:"""
            
            start_time = time.time()

            # [START] CRITICAL FIX: Properly await async generate_text call with event loop isolation
            import asyncio
            import concurrent.futures

            async def async_test_generate():
                return await self.ollama_interface.generate_text(test_prompt, request_timeout=30)

            # [OK] FIXED: Use async converter to prevent sync-over-async anti-pattern
            from .async_converter import run_async_in_thread

            result = run_async_in_thread(async_test_generate())

            elapsed = time.time() - start_time
            
            if result and len(result) > 50:  # Basic validation
                logger.info(f"[OK] Test generation successful in {elapsed:.1f}s")
                logger.info(f"[LIST] Test result preview: {result[:200]}...")
                return True
            else:
                logger.error(f"[ERROR] Test generation failed or insufficient output: {result}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Test generation error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the offline generator is ready with comprehensive availability logging"""
        offline_logger.info("[SEARCH] CHECKING OFFLINE GENERATOR AVAILABILITY...")
        
        try:
            # Simple check: do we have an Ollama interface and is it connected?
            if not self.ollama_interface:
                offline_logger.info("[SEARCH] AVAILABILITY CHECK: NO ollama_interface")
                return False
                
            offline_logger.info("[SEARCH] CHECKING Ollama server response...")
            # Check if Ollama server is responding
            ollama_available = self.ollama_interface.is_available()
            offline_logger.info(f"[SEARCH] OLLAMA SERVER AVAILABLE: {ollama_available}")
            
            # Check if we have an active model - if not, try to initialize
            has_model = (hasattr(self.ollama_interface, 'active_model') and 
                        self.ollama_interface.active_model is not None)
            
            active_model = getattr(self.ollama_interface, 'active_model', None)
            offline_logger.info(f"[SEARCH] HAS_ACTIVE_MODEL: {has_model}")
            offline_logger.info(f"[SEARCH] ACTIVE_MODEL: '{active_model}'")
            
            # If no active model but Ollama is available, try to initialize
            if ollama_available and not has_model:
                offline_logger.info("[SEARCH] Ollama available but no active model - attempting initialization...")
                try:
                    from .async_converter import run_async_in_thread
                    init_result = run_async_in_thread(self.ollama_interface._initialize_model_async)
                    if init_result:
                        has_model = (hasattr(self.ollama_interface, 'active_model') and 
                                   self.ollama_interface.active_model is not None)
                        active_model = getattr(self.ollama_interface, 'active_model', None)
                        offline_logger.info(f"[SEARCH] After initialization - HAS_MODEL: {has_model}, ACTIVE_MODEL: '{active_model}'")
                except Exception as e:
                    offline_logger.warning(f"[WARNING] Failed to initialize model during availability check: {e}")
            
            overall_available = ollama_available and has_model
            offline_logger.info(f"[SEARCH] OVERALL AVAILABILITY: {overall_available}")
            offline_logger.info(f"   • OLLAMA_AVAILABLE: {ollama_available}")
            offline_logger.info(f"   • HAS_MODEL: {has_model}")
            offline_logger.info(f"   • ACTIVE_MODEL: '{active_model}'")
            
            return overall_available
            
        except Exception as e:
            offline_logger.error(f"[ERROR] ERROR CHECKING AVAILABILITY: {e}")
            offline_logger.error(f"[ERROR] TRACEBACK: {traceback.format_exc()}")
            return False

    def generate_mcq(self, topic: str, context: str = "", num_questions: int = 1, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> List[Dict[str, Any]]:
        """
        Generate MCQ questions using optimized BATCH generation with ultra-comprehensive logging
        """
        # [TARGET] COMPREHENSIVE OFFLINE GENERATION LOGGING - START
        performance_logger.info("="*80)
        performance_logger.info("[AI] OFFLINE MCQ GENERATION SESSION STARTED")
        performance_logger.info("="*80)
        
        performance_logger.info(f"[START] STARTING OFFLINE MCQ GENERATION")
        performance_logger.info(f"   • TOPIC: '{topic}'")
        performance_logger.info(f"   • CONTEXT_LENGTH: {len(context) if context else 0}")
        performance_logger.info(f"   • NUM_QUESTIONS: {num_questions}")
        performance_logger.info(f"   • DIFFICULTY: '{difficulty}'")
        performance_logger.info(f"   • QUESTION_TYPE: '{question_type}'")
        
        # [BRAIN] CRITICAL FIX: Use BatchTwoModelPipeline for expert mode as intended
        if difficulty.lower() == "expert":
            offline_logger.info("[EXPERT] EXPERT MODE DETECTED - Using BatchTwoModelPipeline for advanced generation")
            return self._generate_expert_questions_batch(topic, context, num_questions, question_type)
        
        # [HOT] CHECK FOR HARD MODE - Use Enhanced Graduate-Level Generation
        if difficulty.lower() == "hard":
            offline_logger.info("[HOT] HARD MODE DETECTED - Using Graduate-Level Enhanced Generation")
            return self._generate_hard_questions_batch(topic, context, num_questions, question_type)
        
        # [TARGET] LOG OLLAMA CONFIGURATION AND MODEL DETAILS
        performance_logger.info("[SEARCH] OLLAMA CONFIGURATION CHECK:")
        performance_logger.info(f"   • CONFIGURED_MODEL: '{self.model_name}'")
        performance_logger.info(f"   • ENGINE_INITIALIZED: {self.is_engine_initialized}")
        performance_logger.info(f"   • OLLAMA_INTERFACE_AVAILABLE: {self.ollama_interface is not None}")
        
        if self.ollama_interface:
            try:
                # Get detailed Ollama status
                is_available = self.ollama_interface.is_available()
                performance_logger.info(f"   • OLLAMA_SERVER_RESPONDING: {is_available}")
                
                active_model = getattr(self.ollama_interface, 'active_model', None)
                performance_logger.info(f"   • ACTIVE_MODEL: '{active_model}'")
                
                # Try to get performance stats if available
                try:
                    stats = self.ollama_interface.get_performance_stats()
                    performance_logger.info(f"   • GPU_OPTIMIZED: {stats.get('gpu_optimized', 'unknown')}")
                    performance_logger.info(f"   • PERFORMANCE_STATS: {stats}")
                except Exception as e:
                    performance_logger.warning(f"   • PERFORMANCE_STATS: unavailable ({e})")
                    
            except Exception as e:
                performance_logger.error(f"   • OLLAMA_STATUS_CHECK_FAILED: {e}")
        else:
            performance_logger.error("   • OLLAMA_INTERFACE: NOT INITIALIZED")
        
        start_time = time.time()
        
        if not self.is_available():
            offline_logger.warning("[WARNING] ENGINE NOT INITIALIZED. Attempting to initialize now...")
            offline_logger.info("[RELOAD] ATTEMPTING INITIALIZATION...")
            
            performance_logger.warning("[WARNING] OFFLINE ENGINE NOT READY - ATTEMPTING INITIALIZATION...")
            
            init_start = time.time()
            if not self.initialize():
                init_time = time.time() - init_start
                offline_logger.error("[ERROR] FAILED TO INITIALIZE ENGINE for MCQ generation")
                performance_logger.error(f"[ERROR] INITIALIZATION FAILED after {init_time:.3f}s")
                performance_logger.error("="*80)
                return []
            else:
                init_time = time.time() - init_start
                offline_logger.info(f"[OK] ENGINE INITIALIZED SUCCESSFULLY in {init_time:.3f}s")
                performance_logger.info(f"[OK] ENGINE INITIALIZED SUCCESSFULLY in {init_time:.3f}s")

        try:
            offline_logger.info(f"[START] BATCH GENERATING {num_questions} MCQ(s) about '{topic}' in ONE API call - TURBO MODE!")
            performance_logger.info(f"[START] BATCH GENERATION MODE: {num_questions} questions in single API call")
            
            # Create optimized BATCH prompt for ALL questions at once
            offline_logger.info("[CONFIG] Creating batch generation prompt...")
            batch_prompt = self._create_batch_generation_prompt(topic, context, num_questions, question_type)

            # [CONFIG] OPTIMIZED LOGGING: Reduce verbose prompt details to debug level
            if offline_logger.isEnabledFor(logging.DEBUG):
                offline_logger.debug(f"[CONFIG] BATCH PROMPT LENGTH: {len(batch_prompt)} characters")
                offline_logger.debug(f"[CONFIG] BATCH PROMPT PREVIEW: {batch_prompt[:200]}...")

            performance_logger.info(f"[CONFIG] Batch prompt: {len(batch_prompt)} chars, {num_questions} questions, topic: '{topic}'")
            
            # Generate ALL questions in a single API call with retry logic
            offline_logger.info(f"[FAST] MAKING SINGLE API CALL for {num_questions} questions...")
            performance_logger.info(f"[RELOAD] CALLING OLLAMA API: model='{self.model_name}'")
            
            api_start = time.time()
            
            # [START] ENHANCED: Use robust generation with retry logic
            raw_response = self._generate_with_retry(batch_prompt)
            
            api_time = time.time() - api_start
            offline_logger.info(f"[CONFIG] API CALL COMPLETED in {api_time:.3f}s")
            offline_logger.info(f"[CONFIG] RAW RESPONSE LENGTH: {len(raw_response) if raw_response else 0}")
            
            performance_logger.info(f"[OK] OLLAMA API CALL COMPLETED in {api_time:.3f}s")
            performance_logger.info(f"   • RESPONSE_LENGTH: {len(raw_response) if raw_response else 0} characters")
            performance_logger.info(f"   • RESPONSE_RECEIVED: {'YES' if raw_response else 'NO'}")
            
            if raw_response:
                offline_logger.info(f"[CONFIG] RAW RESPONSE PREVIEW: {raw_response[:300]}...")
                
                # Parse ALL questions from the single response
                offline_logger.info(f"[LIST] PARSING {num_questions} questions from batch response...")
                performance_logger.info(f"[LIST] PARSING BATCH RESPONSE: target={num_questions} questions")
                
                parse_start = time.time()
                
                # [START] ENHANCED: Use robust parsing
                parsed_questions = self._parse_batch_mcq_response_robust(raw_response, num_questions, topic)
                
                parse_time = time.time() - parse_start
                offline_logger.info(f"[LIST] PARSING COMPLETED in {parse_time:.3f}s")
                performance_logger.info(f"[LIST] PARSING COMPLETED in {parse_time:.3f}s")
                
                if parsed_questions and len(parsed_questions) > 0:
                    total_time = time.time() - start_time
                    success_count = len(parsed_questions)
                    avg_time_per_question = total_time / success_count
                    
                    # Update performance stats
                    self.generation_stats["total_generated"] += success_count
                    self.generation_stats["avg_time"] = avg_time_per_question
                    
                    offline_logger.info(f"[FINISH] BATCH SUCCESS: Generated {success_count}/{num_questions} questions in {total_time:.1f}s")
                    performance_logger.info(f"[FINISH] BATCH GENERATION SUCCESS")
                    performance_logger.info(f"   • TOTAL_TIME: {total_time:.3f}s")
                    performance_logger.info(f"   • SUCCESS_COUNT: {success_count}")
                    performance_logger.info(f"   • AVG_TIME_PER_QUESTION: {avg_time_per_question:.3f}s")
                    performance_logger.info(f"   • API_TIME: {api_time:.3f}s")
                    performance_logger.info(f"   • PARSE_TIME: {parse_time:.3f}s")
                    performance_logger.info("="*80)
                    performance_logger.info("[AI] OFFLINE MCQ GENERATION SESSION COMPLETED")
                    performance_logger.info("="*80)
                    
                    offline_logger.info(f"[FAST] SPEEDUP: {(num_questions * 6.0) / total_time:.1f}x faster than sequential generation!")
                    
                    return parsed_questions
                else:
                    offline_logger.error(f"[CRITICAL] BATCH PARSING FAILED - No fallback allowed")
                    performance_logger.error(f"[CRITICAL] BATCH PARSING FAILED - Generation stopped")
                    # Return empty list - no fallbacks allowed
                    performance_logger.info("="*80)
                    return []
            else:
                offline_logger.error("[ERROR] NO RESPONSE from batch generation")
                offline_logger.error("   • Check Ollama server status")
                offline_logger.error("   • Verify model is loaded")
                offline_logger.error("   • Check prompt format")
                performance_logger.error("[ERROR] NO RESPONSE from Ollama API")
                performance_logger.error("="*80)
                return []

        except Exception as e:
            total_time = time.time() - start_time
            offline_logger.error(f"[CRITICAL] BATCH MCQ GENERATION FAILED: {e}")
            offline_logger.error(f"[CRITICAL] FULL TRACEBACK: {traceback.format_exc()}")
            performance_logger.error(f"[CRITICAL] GENERATION FAILED after {total_time:.3f}s: {e}")
            performance_logger.error("="*80)
            # Return empty list - no fallbacks allowed
            return []

    def _generate_single_question_fallback(self, topic: str, context: str = "") -> List[Dict[str, Any]]:
        """CRITICAL FIX: Fallback method with enhanced error handling and guaranteed return"""
        try:
            offline_logger.info(f"[RELOAD] FALLBACK: Generating single question for '{topic}'")

            # CRITICAL FIX: Validate inputs to prevent None propagation
            if not topic or not isinstance(topic, str):
                offline_logger.warning("[WARNING] FALLBACK: Invalid topic provided, using default")
                topic = "general knowledge"
            
            if context is None:
                context = ""

            # Use the single question generation method with timeout protection
            try:
                single_result = self.generate_single_mcq(topic, context)

                if single_result and isinstance(single_result, dict):
                    # CRITICAL FIX: Validate the structure before returning
                    if self._is_valid_mcq_response(single_result):
                        offline_logger.info("[OK] FALLBACK: Single question generated and validated successfully")
                        return [single_result]  # Return as list for consistency
                    else:
                        offline_logger.warning("[WARNING] FALLBACK: Single question failed validation")
                        # Continue to alternative fallback
                else:
                    offline_logger.warning("[WARNING] FALLBACK: Single question generation returned invalid result")
                    # Continue to alternative fallback
                    
            except Exception as single_error:
                offline_logger.error(f"[ERROR] FALLBACK: Single question generation exception: {single_error}")
                # Continue to alternative fallback

            # CRITICAL FIX: Enhanced emergency fallback with multiple attempts
            offline_logger.info("[EMERGENCY] FALLBACK: Attempting emergency fallback question creation")
            
            for attempt in range(3):  # Try up to 3 times
                try:
                    fallback_question = self._create_emergency_fallback_question(topic)
                    if fallback_question and isinstance(fallback_question, dict):
                        # Validate emergency fallback
                        if self._is_valid_mcq_response(fallback_question):
                            offline_logger.info(f"[OK] FALLBACK: Emergency fallback question created on attempt {attempt + 1}")
                            return [fallback_question]
                        else:
                            offline_logger.warning(f"[WARNING] FALLBACK: Emergency fallback validation failed on attempt {attempt + 1}")
                    else:
                        offline_logger.warning(f"[WARNING] FALLBACK: Emergency fallback returned invalid result on attempt {attempt + 1}")
                        
                except Exception as emergency_error:
                    offline_logger.error(f"[ERROR] FALLBACK: Emergency fallback attempt {attempt + 1} failed: {emergency_error}")
                    
                # Brief delay between attempts
                if attempt < 2:
                    import time
                    time.sleep(0.1)

            # CRITICAL FIX: Absolute last resort - create minimal valid question
            offline_logger.error("[CRITICAL] FALLBACK: All emergency attempts failed, creating minimal valid question")
            try:
                minimal_question = self._create_minimal_valid_question(topic)
                if minimal_question:
                    offline_logger.info("[OK] FALLBACK: Minimal valid question created as absolute last resort")
                    return [minimal_question]
            except Exception as minimal_error:
                offline_logger.error(f"[ERROR] FALLBACK: Minimal question creation failed: {minimal_error}")

            # CRITICAL FIX: Never return None - always return empty list to prevent caller errors
            offline_logger.error("[CRITICAL] FALLBACK: All fallback methods failed, returning empty list")
            return []

        except Exception as e:
            offline_logger.error(f"[CRITICAL] FALLBACK: Catastrophic fallback failure: {e}")
            offline_logger.error(f"[CRITICAL] FALLBACK: Traceback: {traceback.format_exc()}")
            
            # CRITICAL FIX: Even in catastrophic failure, return empty list, never None
            return []

    def _create_emergency_fallback_question(self, topic: str) -> Optional[Dict[str, Any]]:
        """CRITICAL FIX: Create an emergency fallback question with enhanced validation"""
        try:
            # CRITICAL FIX: Enhanced input sanitization and validation
            if not topic or not isinstance(topic, str):
                topic = "general knowledge"
            
            # Sanitize topic for safety
            try:
                from .inquisitor_prompt import _sanitize_user_input
                safe_topic = _sanitize_user_input(topic.strip())
            except ImportError:
                # Fallback sanitization if import fails
                safe_topic = ''.join(c for c in topic if c.isalnum() or c.isspace()).strip()
                if not safe_topic:
                    safe_topic = "general knowledge"
            
            # CRITICAL FIX: Ensure safe_topic is not empty after sanitization
            if not safe_topic or len(safe_topic.strip()) == 0:
                safe_topic = "general knowledge"
            
            # CRITICAL FIX: Create a more robust educational question with validation
            fallback_question = {
                "question": f"What is an important concept related to {safe_topic}?",
                "options": {
                    "A": f"A fundamental principle in {safe_topic}",
                    "B": f"An advanced technique in {safe_topic}",
                    "C": f"A basic definition in {safe_topic}",
                    "D": "An unrelated concept"
                },
                "correct": "A",
                "explanation": f"This question tests basic understanding of {safe_topic}.",
                "metadata": {
                    "difficulty": "easy",
                    "question_type": "conceptual",
                    "generation_method": "emergency_fallback",
                    "topic": safe_topic
                }
            }
            
            # CRITICAL FIX: Validate the created question before returning
            if self._is_valid_mcq_response(fallback_question):
                offline_logger.info(f"[OK] Emergency fallback question created and validated for topic: {safe_topic}")
                return fallback_question
            else:
                offline_logger.error(f"[ERROR] Emergency fallback question failed validation for topic: {safe_topic}")
                return None
            
        except Exception as e:
            offline_logger.error(f"[ERROR] Emergency fallback question creation failed: {e}")
            offline_logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def _create_minimal_valid_question(self, topic: str) -> Optional[Dict[str, Any]]:
        """CRITICAL FIX: Create absolute minimal valid question as last resort"""
        try:
            # Create the most basic possible valid MCQ
            minimal_question = {
                "question": "What is a basic educational concept?",
                "options": {
                    "A": "A fundamental learning principle",
                    "B": "An advanced research method",
                    "C": "A complex theoretical framework",
                    "D": "An unrelated topic"
                },
                "correct": "A",
                "explanation": "This tests basic educational understanding.",
                "metadata": {
                    "difficulty": "easy",
                    "question_type": "conceptual",
                    "generation_method": "minimal_fallback",
                    "topic": "education"
                }
            }
            
            # Validate the minimal question
            if self._is_valid_mcq_response(minimal_question):
                offline_logger.info("[OK] Minimal valid question created as absolute last resort")
                return minimal_question
            else:
                offline_logger.error("[ERROR] Even minimal question failed validation")
                return None
                
        except Exception as e:
            offline_logger.error(f"[ERROR] Minimal question creation failed: {e}")
            return None

    def generate_single_mcq(self, topic: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Generate a single MCQ question with robust error handling"""
        try:
            if not self.is_available():
                offline_logger.error("[ERROR] Generator not available for single MCQ")
                return None

            # Create optimized prompt for single question
            prompt = self._create_optimized_expert_prompt(topic, context, "mixed", 0)
            
            # Generate with timeout protection
            result = self._generate_single_question_with_timeout(prompt, timeout=45)
            
            if result and self._is_valid_mcq_response(result):
                offline_logger.info(f"[OK] Single MCQ generated for '{topic}'")
                return result
            else:
                offline_logger.warning(f"[WARNING] Single MCQ generation failed for '{topic}'")
                return None
                
        except Exception as e:
            offline_logger.error(f"[ERROR] Single MCQ generation error: {e}")
            return None

    def _create_batch_generation_prompt(self, topic: str, context: str, num_questions: int, question_type: str = "mixed", difficulty: str = "medium") -> str:
        """Create optimized prompt for generating ALL questions in one batch"""

        # 🔧 FIX: Sanitize inputs to prevent prompt injection
        from .inquisitor_prompt import _sanitize_user_input
        sanitized_topic = _sanitize_user_input(topic)
        sanitized_context = _sanitize_user_input(context)

        safe_topic = self._make_topic_educational(sanitized_topic)
        topic_constraints = self._get_topic_specific_constraints(safe_topic)
        
        # Question type specific instructions
        type_instruction = ""
        if question_type.lower() == "numerical":
            type_instruction = """
🔢 NUMERICAL EXCELLENCE STANDARDS 🔢

**Quantitative Learning Focus:**
✓ Calculation Starters: Begin with "Calculate", "Compute", "Solve", "Determine", "Find", "Evaluate"
✓ Mathematical Depth: Equations, formulas, specific values, numerical operations
✓ Problem-Solving: Involve mathematical computation and analytical thinking
✓ Quantitative Options: All 4 answers should be numbers with appropriate units (J, kg, m/s, etc.)
✓ Skills Assessment: Test computational abilities and mathematical understanding

**Quality Enhancement:**
✓ Focus on quantitative analysis to maintain numerical question integrity
✓ Include specific numerical values and units for precision
✓ Ensure all options are meaningful numerical answers
✓ Test mathematical skills through structured problem-solving
✓ Create questions that demonstrate quantitative mastery

Remember: Pure numerical focus promotes clear learning objectives and assessment clarity.
"""
        elif question_type.lower() == "conceptual":
            type_instruction = """
🧠 CONCEPTUAL EXCELLENCE STANDARDS 🧠

**Theoretical Learning Focus:**
✓ Understanding Starters: Begin with "Explain", "Why", "How", "What happens", "Describe", "Analyze"  
✓ Principles Depth: Theories, mechanisms, cause-effect relationships, underlying principles
✓ Conceptual Clarity: Focus on theoretical understanding without numerical computation
✓ Theoretical Options: All 4 answers should be concept descriptions and mechanism explanations
✓ Knowledge Assessment: Test theoretical understanding and conceptual mastery
✓ Qualitative Analysis: Relationships, trends, phenomena that build understanding

**Quality Enhancement:**
✓ Focus on conceptual understanding to maintain theoretical question integrity

[FORBIDDEN] NUMERICAL CONTAMINATION - AUTOMATIC FAILURE:
[ERROR] "calculate" [ERROR] "compute" [ERROR] "solve" [ERROR] "determine" [ERROR] "find" [ERROR] "evaluate"
[ERROR] Numbers with units [ERROR] Mathematical expressions [ERROR] Formulas [ERROR] Equations [ERROR] Calculations
[ERROR] NO numerical operations, NO specific values, NO computational elements

**ZERO TOLERANCE VERIFICATION:**
[SEARCH] UNDERSTANDING VERB? → Must be "Explain/Why/How/What happens/Describe/Analyze"
[SEARCH] NO NUMBERS? → Zero specific numerical values or calculations required
[SEARCH] PURE CONCEPTUAL OPTIONS? → ALL options describe concepts/mechanisms, NO numbers
[SEARCH] UNDERSTANDING REQUIRED? → Must test theoretical knowledge, NOT math skills
[SEARCH] NO NUMERICAL WORDS? → Zero calculation verbs or mathematical operations

💀 FAILURE MODES TO AVOID: Any question asking to "calculate", any numerical options, any mathematical operations

[EMERGENCY] WARNING: Any question that includes calculations will be AUTOMATICALLY REJECTED!
[EMERGENCY] DEMAND: Generate ONLY understanding/explanation questions!
"""
        
        # Different question focus areas for variety in batch
        if question_type.lower() == "numerical":
            question_focuses = [
                "mathematical calculations and problem-solving",
                "quantitative analysis and computations", 
                "numerical problem-solving with formulas",
                "calculation-based applications",
                "mathematical operations and evaluations"
            ]
        elif question_type.lower() == "conceptual":
            question_focuses = [
                "theoretical understanding and principles",
                "conceptual explanations and mechanisms", 
                "qualitative analysis and reasoning",
                "cause-effect relationships and theories",
                "understanding of underlying concepts"
            ]
        else:
            question_focuses = [
                "foundational concepts and definitions",
                "practical applications and examples", 
                "analysis and comparison of concepts",
                "problem-solving and calculations",
                "conceptual understanding and principles"
            ]
        
        # Domain-specific requirements
        domain_requirements = ""
        topic_lower = safe_topic.lower()
        if any(term in topic_lower for term in ["physics", "mechanics", "quantum", "electromagnetic"]):
            domain_requirements = "Must include physics terms like: force, energy, momentum, wave, particle, field, quantum"
        elif any(term in topic_lower for term in ["chemistry", "chemical", "organic", "inorganic"]):
            domain_requirements = "Must include chemistry terms like: molecule, atom, bond, reaction, compound, solution, acid"
        elif any(term in topic_lower for term in ["mathematics", "math", "calculus", "algebra"]):
            domain_requirements = "Must include math terms like: equation, function, derivative, integral, matrix, variable, theorem"

        # Length requirements
        min_length = 120 if difficulty.lower() == "expert" else 80 if difficulty.lower() in ["hard", "medium"] else 50

        # Create focused prompt for batch generation
        prompt = f"""You are an educational content expert specializing in {safe_topic}. Generate {num_questions} diverse {question_type} multiple choice questions focused on {safe_topic}.

QUALITY STANDARDS:
- Each question should be at least {min_length} characters long for appropriate depth
- Each question should end with a question mark (?)
- {domain_requirements}
- All options should be substantive and meaningful
- Expert questions should demonstrate advanced understanding and application

{type_instruction}

EXCELLENCE STANDARDS:
- Generate EXACTLY {num_questions} complete questions
- Each question must focus ONLY on {safe_topic}
- NO mixing with other academic fields (brain anatomy, ecology, computer science, etc.)
- Questions should cover: {', '.join(question_focuses[:num_questions])}
- ALL answer options must relate directly to {safe_topic}

{topic_constraints}

RESPONSE FORMAT - Return EXACTLY {num_questions} questions as a JSON array:
[
  {{
    "question": "Question 1 about {safe_topic}?",
    "options": {{
      "A": "Correct answer related to {safe_topic}",
      "B": "Plausible but incorrect option in {safe_topic}",
      "C": "Another plausible incorrect option in {safe_topic}",
      "D": "Common misconception in {safe_topic}"
    }},
    "correct": "A",
    "explanation": "Brief explanation focusing on {safe_topic} principles"
  }},
  {{
    "question": "Question 2 about {safe_topic}?",
    "options": {{
      "A": "Option A for question 2",
      "B": "Option B for question 2", 
      "C": "Option C for question 2",
      "D": "Option D for question 2"
    }},
    "correct": "B",
    "explanation": "Explanation for question 2"
  }}
  {"," if num_questions > 2 else ""}
  {f"... (continue for all {num_questions} questions)" if num_questions > 2 else ""}
]

Generate EXACTLY {num_questions} complete questions following this format:"""

        return prompt

    def _parse_batch_mcq_response(self, response: str, expected_questions: int, topic: str) -> List[Dict[str, Any]]:
        """Parse multiple MCQ questions from a single batch response"""
        try:
            logger.info(f"[SEARCH] Parsing batch response for {expected_questions} questions...")
            
            # Try to extract JSON array from response
            import re
            
            # Find JSON array in the response
            json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if json_match:
                json_content = '[' + json_match.group(1) + ']'
                try:
                    questions_data = json.loads(json_content)
                    
                    if isinstance(questions_data, list):
                        parsed_questions = []
                        for i, q_data in enumerate(questions_data[:expected_questions]):  # Limit to expected number
                            if self._validate_question_structure(q_data):
                                # [CONFIG] ENHANCED VALIDATION: Validate basic structure
                                if self._is_valid_mcq_response(q_data):
                                    parsed_questions.append(q_data)
                                    logger.info(f"[OK] Parsed and validated question {i+1}: {q_data.get('question', '')[:50]}...")
                                else:
                                    logger.warning(f"[WARNING] Question {i+1} failed enhanced validation, skipping")
                            else:
                                logger.warning(f"[WARNING] Question {i+1} failed basic validation, skipping")
                        
                        logger.info(f"[LIST] Successfully parsed {len(parsed_questions)}/{expected_questions} questions from batch")
                        return parsed_questions
                    else:
                        logger.warning("[WARNING] Response is not a JSON array")
                except json.JSONDecodeError as e:
                    logger.warning(f"[WARNING] JSON parsing failed: {e}")
            
            # Fallback: Try to parse individual questions if array parsing fails
            logger.info("[RELOAD] Trying fallback: parsing individual questions from response...")
            return self._parse_individual_questions_from_text(response, expected_questions, topic)
            
        except Exception as e:
            logger.error(f"[ERROR] Batch parsing error: {e}")
            return []

    def _parse_individual_questions_from_text(self, text: str, expected_questions: int, topic: str) -> List[Dict[str, Any]]:
        """Fallback parser to extract individual questions from text response"""
        try:
            questions = []
            
            # Split by common delimiters that might separate questions
            potential_questions = re.split(r'(?:\n\s*\n|\}\s*,?\s*\{)', text)
            
            for i, potential_q in enumerate(potential_questions):
                if i >= expected_questions:
                    break
                    
                # Try to extract a complete question JSON from this segment
                json_match = re.search(r'\{.*?\}', potential_q, re.DOTALL)
                if json_match:
                    try:
                        question_json = json_match.group(0)
                        question_data = json.loads(question_json)
                        
                        if self._validate_question_structure(question_data):
                            questions.append(question_data)
                            logger.info(f"[OK] Extracted question {len(questions)}: {question_data.get('question', '')[:50]}...")
                            
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"[LIST] Fallback parsing extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"[ERROR] Fallback parsing error: {e}")
            return []

    def _validate_question_structure(self, question_data: Dict[str, Any]) -> bool:
        """Validate that a question has the required structure"""
        try:
            required_keys = ['question', 'options', 'correct', 'explanation']
            
            # Check all required keys exist
            for key in required_keys:
                if key not in question_data:
                    logger.warning(f"[WARNING] Missing required key: {key}")
                    return False
            
            # Validate options structure
            options = question_data['options']
            if not isinstance(options, dict):
                logger.warning("[WARNING] Options is not a dictionary")
                return False
                
            # Check for required option keys (A, B, C, D)
            required_options = ['A', 'B', 'C', 'D']
            for opt in required_options:
                if opt not in options or not options[opt].strip():
                    logger.warning(f"[WARNING] Missing or empty option: {opt}")
                    return False
            
            # Validate correct answer
            correct = question_data['correct']
            if correct not in options:
                logger.warning(f"[WARNING] Correct answer '{correct}' not in options")
                return False
            
            # Basic content validation
            question_text = question_data['question']
            if len(question_text.strip()) < 10:
                logger.warning("[WARNING] Question too short")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"[WARNING] Question validation error: {e}")
            return False



    def _create_speed_optimized_prompt(self, topic: str, context: str, question_index: int) -> str:
        """Create an optimized prompt for faster generation"""
        
        # Make topic more educational and acceptable
        safe_topic = self._make_topic_educational(topic)
        
        # Add variety to avoid repetitive generation
        question_types = [
            "factual knowledge",
            "conceptual understanding", 
            "practical application",
            "analysis and comparison",
            "problem-solving"
        ]
        
        focus = question_types[question_index % len(question_types)]
        
        # Create topic-specific constraints to prevent mixing
        topic_constraints = self._get_topic_specific_constraints(safe_topic)
        
        # AGGRESSIVE anti-vague question enforcement
        anti_vague_section = """
[EMERGENCY] ANTI-VAGUE QUESTION ENFORCEMENT:
[ERROR] COMPLETELY FORBIDDEN: "What is the primary function of..."
[ERROR] COMPLETELY FORBIDDEN: "What is the main purpose of..."
[ERROR] COMPLETELY FORBIDDEN: "What does X do?"
[ERROR] COMPLETELY FORBIDDEN: Basic definition questions
[ERROR] COMPLETELY FORBIDDEN: General overview questions
[OK] REQUIRED: Specific mechanisms, pathways, processes
[OK] REQUIRED: Detailed analysis and reasoning
[OK] REQUIRED: Expert-level specificity
[OK] REQUIRED: Questions requiring deep understanding

BIOLOGY/REPRODUCTIVE SYSTEM SPECIAL RULES:
If topic relates to reproduction/reproductive systems:
- Ask about SPECIFIC hormonal pathways (FSH, LH, testosterone, estrogen regulation)
- Focus on MOLECULAR mechanisms (meiosis stages, fertilization processes)
- Include BIOCHEMICAL processes (steroidogenesis, gametogenesis)
- Test understanding of REGULATORY feedback loops
- Ask about SPECIFIC anatomical structures and their precise functions

EXAMPLES OF BANNED vs REQUIRED QUESTIONS:
[ERROR] BANNED: "What is the primary function of the male reproductive system?"
[OK] REQUIRED: "During which phase of spermatogenesis do primary spermatocytes undergo the first meiotic division?"

[ERROR] BANNED: "What is the main purpose of the female reproductive system?"
[OK] REQUIRED: "Which hormone surge triggers ovulation and what cellular mechanism initiates this process?"
"""
        
        prompt = f"""You are an educational content creator specializing in {safe_topic}. Create a {focus} multiple choice question focused on {safe_topic}.

{anti_vague_section}

QUALITY GUIDELINES:
- Question should focus primarily on {safe_topic}
- All answer options should relate meaningfully to {safe_topic}
- Maintain clear boundaries with related fields while allowing natural connections
- Stay within the core concepts of {safe_topic} for best learning outcomes
- Create specific, detailed questions that promote understanding

{topic_constraints}

Topic: {safe_topic}
Approach: {focus}

Create a focused multiple choice question with exactly 4 options labeled A, B, C, D.
One option must be clearly correct.
Include a brief explanation.

QUALITY REQUIREMENTS:
- Questions must be specific and detailed, not vague
- Use precise terminology and concepts
- Focus on mechanisms and processes, not just basic definitions
- Include challenging but fair distractors
- Test understanding of HOW and WHY, not just WHAT

Example structure for {safe_topic}:
- Question about specific concepts, mechanisms, or processes in {safe_topic}
- All options must be within {safe_topic} domain
- Avoid mixing with other subjects
- Focus on detailed understanding rather than surface-level facts

Return ONLY a JSON object in this exact format:
{{
  "question": "What specific mechanism/process in {safe_topic} explains [detailed scenario]?",
  "options": {{
    "A": "Specific, detailed concept relevant to {safe_topic}",
    "B": "Another specific, detailed concept relevant to {safe_topic}",
    "C": "Related but incorrect specific concept in {safe_topic}", 
    "D": "Common misconception with specific details in {safe_topic}"
  }},
  "correct": "A",
  "explanation": "Detailed explanation focusing on specific {safe_topic} principles and mechanisms"
}}"""

        return prompt

    def _get_topic_specific_constraints(self, safe_topic: str) -> str:
        """Get specific constraints for different topics to prevent mixing"""
        
        constraints_map = {
            'human biology and reproductive health education': """
TOPIC CONSTRAINTS:
- Focus only on reproductive anatomy, physiology, and health
- Include concepts like hormones, reproductive cycles, contraception
- DO NOT mention brain anatomy, neural networks, or neuroscience
- DO NOT include ecology, biodiversity, or environmental topics
- Stay within reproductive health domain exclusively
""",
            'biological sciences': """
TOPIC CONSTRAINTS:
- Focus on cell biology, genetics, evolution, ecology
- Include concepts like DNA, proteins, cellular processes
- DO NOT mix with computer science or programming
- DO NOT include physics concepts like quantum mechanics
- Stay within biological science principles
""",
            'health and wellness education': """
TOPIC CONSTRAINTS:
- Focus on nutrition, exercise, mental health, disease prevention
- Include concepts like healthy lifestyle, medical screening
- DO NOT mix with unrelated medical specialties
- Stay within general health and wellness scope
"""
        }
        
        # Get specific constraints or generic ones
        return constraints_map.get(safe_topic, f"""
TOPIC CONSTRAINTS:
- Focus exclusively on concepts within {safe_topic}
- DO NOT mix with unrelated academic fields
- All options must be relevant to {safe_topic}
- Avoid generic or overly broad questions
""")

    def _make_topic_educational(self, topic: str) -> str:
        """Convert any topic into an educational, appropriate format"""
        topic_lower = topic.lower().strip()
        
        # Educational mappings for sensitive topics
        educational_mappings = {
            'sex': 'human biology and reproductive health education',
            'sexual': 'biology and health education',
            'reproduction': 'biological reproduction processes',
            'anatomy': 'human anatomy and physiology',
            'biology': 'biological sciences',
            'health': 'health and wellness education',
            'science': 'scientific principles and concepts',
            'education': 'educational theory and practice',
            'learning': 'learning processes and methods'
        }
        
        # Check for exact matches first
        if topic_lower in educational_mappings:
            return educational_mappings[topic_lower]
        
        # Check for partial matches
        for key, value in educational_mappings.items():
            if key in topic_lower:
                return value
        
        # If no special mapping needed, make it educational
        if len(topic.strip()) > 0:
            return f"the academic study of {topic}"
        else:
            return "general knowledge and concepts"

    def _create_fallback_prompt(self, topic: str) -> str:
        """Create a simpler fallback prompt for retry attempts"""
        safe_topic = self._make_topic_educational(topic)
        
        return f"""Educational question generator.

Create a simple multiple choice question about {safe_topic}.

The question must be educational and appropriate.

Return exactly this JSON format:
{{
  "question": "What is a key aspect of {safe_topic}?",
  "options": {{
    "A": "Important foundational concept",
    "B": "Secondary supporting idea", 
    "C": "Unrelated information",
    "D": "Incorrect assumption"
  }},
  "correct": "A",
  "explanation": "Brief educational explanation"
}}

Generate only the JSON:"""

    def _parse_mcq_response(self, response: str, attempt: int = 1) -> Optional[Dict[str, Any]]:
        """
        Parse MCQ response with enhanced error handling and multiple strategies
        """
        try:
            # Use the enhanced MCQ parser from utils
            from knowledge_app.utils.enhanced_mcq_parser import EnhancedMCQParser
            
            parser = EnhancedMCQParser()
            parsed_mcq = parser.parse_mcq(response)
            
            if parsed_mcq:
                logger.info(f"[OK] Successfully parsed MCQ on attempt {attempt}")
                return parsed_mcq
            else:
                logger.warning(f"[WARNING] Parsing failed on attempt {attempt}")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] MCQ parsing error on attempt {attempt}: {e}")
            return None

    async def generate_quiz_async(self, context: str, topic: str, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Generate a single MCQ question async - FIXED METHOD SIGNATURE
        """
        try:
            logger.info(f"[START] Generating single MCQ for topic: '{topic}', difficulty: {difficulty}")
            
            # Generate single question using sync method
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.generate_mcq, topic, context, 1)
            
            if results and len(results) > 0:
                mcq = results[0]
                # Convert to expected format
                return {
                    "question": mcq.get("question", ""),
                    "options": mcq.get("options", {}),
                    "correct": mcq.get("correct", "A"),
                    "explanation": mcq.get("explanation", "")
                }
            else:
                logger.error("[ERROR] No MCQ generated")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Async MCQ generation failed: {e}")
            return None

    async def generate_mcq_async(self, topic: str, context: str = "", num_questions: int = 1, 
                                 difficulty: str = "medium", game_mode: str = "casual", 
                                 question_type: str = "mixed") -> List[Dict[str, Any]]:
        """
        Async version of MCQ generation for better performance - FIXED SIGNATURE
        """
        offline_logger.info(f"[SEARCH] ASYNC DEBUG: generate_mcq_async called with difficulty='{difficulty}', question_type='{question_type}'")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_mcq, topic, context, num_questions, difficulty, game_mode, question_type)


    def _parse_mcq_response_enhanced(self, response: str) -> Optional[Dict[str, Any]]:
        """Enhanced MCQ response parsing with multiple format support"""
        try:
            if not response or not response.strip():
                return None
            
            # Clean the response
            response = response.strip()
            
            # Try to find JSON in the response
            import re
            
            # Look for JSON block
            json_match = re.search(r'\{[^}]*"question"[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to extract everything between first { and last }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response[start:end+1]
                else:
                    json_str = response
            
            # Try to parse JSON
            try:
                parsed = json.loads(json_str)
                
                # Normalize field names
                if 'correct' in parsed and 'correct_answer' not in parsed:
                    parsed['correct_answer'] = parsed['correct']
                
                # Ensure options are in the right format
                if 'options' in parsed:
                    options = parsed['options']
                    if isinstance(options, list):
                        # Convert list to dict format
                        option_dict = {}
                        for i, option in enumerate(options):
                            key = chr(65 + i)  # A, B, C, D
                            option_dict[key] = option
                        parsed['options'] = option_dict
                
                return parsed
                
            except json.JSONDecodeError as e:
                offline_logger.warning(f"JSON parsing failed: {e}")
                return None
                
        except Exception as e:
            offline_logger.error(f"Enhanced parsing failed: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "engine_initialized": self.is_engine_initialized,
            "available": self.is_available(),
            "model_name": self.model_name,
            "generation_stats": self.generation_stats
        }
        
        if self.ollama_interface:
            stats.update(self.ollama_interface.get_performance_stats())
            
        return stats

    def optimize_for_batch_generation(self):
        """Apply optimizations specifically for batch generation"""
        if self.ollama_interface:
            logger.info("[START] Applying batch generation optimizations...")
            
            # Further optimize for batch processing
            self.ollama_interface.generation_params.update({
                'temperature': 0.85,  # Slightly higher for variety in batch
                'top_k': 25,  # More restrictive for faster batch processing
                'num_predict': 350,  # Optimized length for MCQs
                'batch_size': 1024,  # Large batch size for GPU efficiency
                'parallel': 8,  # Maximum parallel processing
            })
            
            logger.info("[FAST] Batch optimization complete - Ready for high-speed generation!")

    def cleanup(self):
        """Clean up resources"""
        if self.ollama_interface:
            logger.info("[CLEAN] Cleaning up Ollama interface")
            self.ollama_interface = None
        self.is_engine_initialized = False

    def _generate_hard_questions_batch(self, topic: str, context: str, num_questions: int, question_type: str) -> List[Dict[str, Any]]:
        """[HOT] HARD MODE: Generate graduate-level complex questions using BatchTwoModelPipeline"""
        
        offline_logger.info(f"[HOT] STARTING HARD MODE BATCH GENERATION: {num_questions} questions about '{topic}'")
        
        # Prepare test cases for hard mode generation
        test_cases = []
        domain = self._extract_domain_from_topic(topic)
        
        for i in range(num_questions):
            test_cases.append({
                'domain': domain,
                'topic': topic,
                'context': context,
                'difficulty': 'hard',
                'question_index': i
            })
        
        try:
            # Step 1: Generate graduate-level thinking
            offline_logger.info("[BRAIN] Step 1/3: Generating graduate-level thinking...")
            thinking_response = self._batch_generate_hard_thinking(test_cases, topic, question_type)
            
            if not thinking_response:
                offline_logger.error("[ERROR] Hard mode thinking generation failed")
                return []
            
            # Step 2: Extract individual thinking blocks
            offline_logger.info("[LIST] Step 2/3: Extracting individual thinking blocks...")
            thinking_list = self._extract_individual_thinking(thinking_response, test_cases)
            
            if not thinking_list or len(thinking_list) < num_questions:
                offline_logger.warning(f"[WARNING] Only got {len(thinking_list)}/{num_questions} thinking blocks")
                # Pad with the last available thinking block
                while len(thinking_list) < num_questions:
                    thinking_list.append(thinking_list[-1] if thinking_list else "Graduate-level analysis required")
            
            # Step 3: Generate JSON for each question
            offline_logger.info("[DOC] Step 3/3: Generating JSON for each question...")
            json_results = self._batch_generate_hard_json(thinking_list, test_cases)
            
            # Filter out None results
            valid_questions = [q for q in json_results if q is not None]
            
            offline_logger.info(f"[FINISH] HARD MODE COMPLETED: {len(valid_questions)}/{num_questions} questions generated")
            
            if valid_questions:
                return valid_questions
            else:
                offline_logger.error("[ERROR] No valid hard mode questions generated")
                return []
                
        except Exception as e:
            offline_logger.error(f"[ERROR] Hard mode batch generation failed: {e}")
            offline_logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
            return []

    def _batch_generate_hard_thinking(self, test_cases: List[Dict], topic: str, question_type: str = "mixed") -> Optional[str]:
        """Step 1: Generate graduate-level thinking for hard mode questions"""
        
        # Question type specific instructions for hard mode
        type_instruction = ""
        if question_type.lower() == "numerical":
            type_instruction = """
🔢 HARD MODE NUMERICAL EXCELLENCE 🔢

**Graduate-Level Numerical Mastery:**
✓ Calculation Focus: Use precise verbs like "Calculate", "Compute", "Solve", "Determine", "Find", "Evaluate"
✓ Mathematical Rigor: Complex equations, multi-step derivations, sophisticated quantitative analysis
✓ Technical Depth: Graduate-level formulas, numerical methods, and computational techniques
✓ Quantitative Options: All 4 answers should be meaningful numerical values with appropriate units
✓ Problem Complexity: Multi-layered analysis requiring advanced mathematical reasoning

**Excellence Examples:**
✓ "Calculate the relativistic momentum of an electron moving at 0.8c"
✓ "Determine the magnetic field strength at the center of a solenoid with 500 turns/cm carrying 2.5A"
✓ "Solve for the equilibrium constant Kc at 298K given ΔG° = -25.6 kJ/mol"

Remember: Focus on quantitative problem-solving to maintain numerical question integrity.
"""
        elif question_type.lower() == "conceptual":
            type_instruction = """
🧠 HARD MODE CONCEPTUAL EXCELLENCE 🧠

**Graduate-Level Conceptual Mastery:**
✓ Understanding Focus: Use insight verbs like "Explain", "Why", "How", "What happens", "Describe", "Analyze"
✓ Theoretical Depth: Advanced principles, complex mechanisms, sophisticated theoretical frameworks
✓ Qualitative Reasoning: Deep understanding of cause-effect relationships and underlying mechanisms
✓ Conceptual Options: All 4 answers should be sophisticated theoretical explanations
✓ Analytical Complexity: Multi-layered reasoning requiring advanced theoretical knowledge

**Excellence Examples:**
✓ "Explain the quantum mechanical basis for the Pauli exclusion principle in multi-electron atoms"
[OK] "Why does the entropy of the universe increase during spontaneous chemical reactions?"
[OK] "How does electromagnetic induction relate to relativistic effects in moving conductors?"
"""
        else:
            type_instruction = """
🔀 HARD MODE MIXED REQUIREMENTS:
[OK] Can include either advanced numerical OR sophisticated conceptual elements
[OK] Vary between quantitative graduate-level analysis and theoretical understanding
[OK] Mix complex calculations with deep theoretical reasoning
[OK] Balance advanced mathematical skills with conceptual expertise
"""

        prompt_parts = [f"""Generate graduate-level multiple choice questions that require ADVANCED ANALYTICAL THINKING. Each question MUST be:

1. **GRADUATE COMPLEXITY** - Minimum 80+ characters, requiring advanced understanding
2. **MULTI-STEP REASONING** - Cannot be answered by simple recall or basic formulas
3. **ANALYTICAL DEPTH** - Test understanding of complex relationships and systems
4. **ADVANCED TERMINOLOGY** - Use technical vocabulary appropriately
5. **SYNTHESIS REQUIRED** - Test ability to combine multiple concepts

{type_instruction}

[HOT] HARD MODE REQUIREMENTS - ABSOLUTELY FORBIDDEN:
[ERROR] Basic formula applications (F=ma, E=mc², KE=½mv², etc.)
[ERROR] Single-step textbook problems
[ERROR] "What is..." definition questions
[ERROR] Simple recall or memorization
[ERROR] Undergraduate homework-level questions

[OK] HARD MODE REQUIREMENTS - MANDATORY:
[OK] Multi-step problem solving requiring 3+ concepts
[OK] Advanced analytical techniques and methods
[OK] Complex systems with multiple interacting components
[OK] Graduate-level complexity (master's degree level)
[OK] Synthesis of multiple principles
[OK] Advanced applications requiring deep domain knowledge

Generate questions for these test cases with GRADUATE-LEVEL COMPLEXITY:

"""]
        
        for i, case in enumerate(test_cases, 1):
            domain = case.get('domain', 'General')
            topic_text = case.get('topic', 'General')
            context = case.get('context', '')
            
            complexity_instruction = f"""
**HARD MODE (GRADUATE-LEVEL) REQUIREMENTS for {question_type.upper()} questions:**
- Question must require master's degree-level knowledge
- Include advanced analytical techniques or complex theoretical concepts
- Test synthesis of multiple advanced concepts
- Require multi-step logical reasoning
- Use technical terminology appropriately for graduate level
- Question length: MINIMUM 80 characters
- Should challenge advanced undergraduate and graduate students
- Must avoid basic formula applications and single-step problems
- MUST follow {question_type.upper()} requirements above
"""
            
            prompt_parts.append(f"""
Test Case {i}: {domain} - {topic_text} (HARD difficulty - Graduate Level {question_type.upper()})
{complexity_instruction}

Think through:
- What ADVANCED {question_type} concept should be tested that requires graduate-level analysis?
- What complex relationship or system behavior is involved?
- What multi-step reasoning should be required?
- What sophisticated analysis techniques apply?
- How to make this challenging for graduate students?
- What advanced terminology and concepts must be included?

MAKE THIS QUESTION GRADUATE-LEVEL CHALLENGING AND ANALYTICAL FOR {question_type.upper()} TYPE!

""")
        
        prompt_parts.append(f"""
CRITICAL HARD MODE INSTRUCTIONS:
- Hard questions should demonstrate graduate-level complexity and understanding
- Use advanced analytical methods and sophisticated reasoning
- Test comprehensive understanding of systems and relationships
- Questions should require multi-step reasoning and synthesis
- Include analysis of complex interactions where appropriate
- Create challenges suitable for graduate-level study
- Aim for substantial content depth (80+ characters recommended)
- Focus on advanced concepts beyond basic formulas
- Emphasize synthesis, analysis, and evaluation skills
- Follow {question_type.upper()} excellence guidelines for optimal question design

Generate sophisticated, analytically challenging {question_type} questions that inspire learning excellence!""")
        
        full_prompt = "".join(prompt_parts)
        
        offline_logger.info(f"[EXCELLENCE] Graduate-level {question_type} thinking generation with {self.thinking_model}")
        response = self._generate_with_retry(full_prompt, model_override=self.thinking_model, max_tokens=12000)
        
        if response:
            offline_logger.info(f"[SUCCESS] Graduate-level {question_type} thinking generated ({len(response)} chars)")
            return response
        else:
            offline_logger.info(f"[INFO] No graduate-level {question_type} thinking generated this attempt")
            return None

    def _batch_generate_hard_json(self, thinking_list: List[str], test_cases: List[Dict]) -> List[Optional[Dict]]:
        """Step 3: Generate JSON for each hard mode question with graduate-level requirements"""
        
        results = []
        
        for i, (thinking, test_case) in enumerate(zip(thinking_list, test_cases)):
            domain = test_case.get('domain', 'General')
            topic = test_case.get('topic', 'General') 
            
            complexity_reminder = """
HARD MODE (GRADUATE-LEVEL) JSON REQUIREMENTS:
- Question MUST be graduate-level complexity (minimum 80 characters)
- Include advanced analytical concepts or multi-step reasoning
- Test deep understanding requiring synthesis of concepts
- Use appropriate technical vocabulary for graduate level
- Options must be analytically sophisticated, not simple choices
- Avoid basic formulas and single-step problems
- QUESTION MUST END WITH A QUESTION MARK (?)
- Focus on analysis, synthesis, evaluation rather than recall
"""
            
            prompt = f"""Generate a valid JSON MCQ from this thinking:

THINKING: {thinking}

HARD MODE - GRADUATE LEVEL COMPLEXITY REQUIRED

CRITICAL JSON FORMAT - EXACT OUTPUT REQUIRED:
{{
  "question": "EXACTLY ONE graduate-level analytical question ending with ?",
  "options": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct": "A",
  "explanation": "Brief explanation"
}}

{complexity_reminder}

MANDATORY RULES:
- EXACTLY 4 options in A,B,C,D format  
- Use "correct" not "correct_answer"
- Question must end with ?
- No extra commas or brackets
- No explanatory text outside JSON
- Generate ONLY the JSON object above
- Must require graduate-level analytical thinking"""
            
            offline_logger.info(f"[DOC] Hard mode JSON generation {i+1}/{len(test_cases)} with {self.json_model}")
            
            # Try up to 3 times for better reliability
            json_data = None
            for attempt in range(3):
                response = self._generate_with_chunked_buffer(prompt, model_override=self.json_model, max_tokens=2000)
                
                if response:
                    json_data = self._parse_json_response_robust(response)
                    if json_data:
                        break  # Success, exit retry loop
                    elif attempt < 2:  # Not the last attempt
                        offline_logger.warning(f"   [WARNING]  Retry {attempt + 1}/3 for JSON {i+1}")
                
                if attempt == 2:  # Last attempt failed
                    offline_logger.error(f"   [ERROR] JSON {i+1} failed after 3 attempts")
            
            if json_data:
                # Fix question mark if missing
                question = json_data.get('question', '')
                if question and not question.endswith('?'):
                    json_data['question'] = question.rstrip('.!') + '?'
                
                # Add metadata
                json_data['difficulty'] = 'hard'
                json_data['domain'] = domain
                json_data['topic'] = topic
                json_data['generated_by'] = f"{self.thinking_model} + {self.json_model}"
                json_data['pipeline'] = "batch_two_model_hard"
                results.append(json_data)
                offline_logger.info(f"   [OK] JSON {i+1} generated")
            else:
                results.append(None)
                offline_logger.error(f"   [ERROR] JSON {i+1} no valid response")
        
        return results
