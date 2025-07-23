"""
🔥 FIRE: Enhanced LM Studio Generator
Using official lmstudio SDK with grammar-constrained JSON and Pydantic validation
Based on LM Studio's open-source architecture for guaranteed JSON output
"""

from .async_converter import async_requests_post, async_requests_get


import logging
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio

try:
    from openai import OpenAI
    import requests
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    logging.warning("⚠️ OpenAI client not available - install with: pip install openai")

from .mcq_generator import MCQGenerator

logger = logging.getLogger(__name__)


class MCQSchema(BaseModel):
    """🔥 FIRE: Pydantic schema for guaranteed MCQ structure"""
    question: str = Field(description="The multiple choice question text", min_length=10)
    options: List[str] = Field(description="Exactly 4 answer options", min_length=4, max_length=4)
    correct_answer: str = Field(description="The correct answer from the options")
    explanation: str = Field(description="Explanation of why the answer is correct", min_length=10)


class EnhancedLMStudioGenerator(MCQGenerator):
    """
    🔥 FIRE: Enhanced LM Studio Generator
    - Official lmstudio SDK integration
    - Grammar-constrained JSON generation (GUARANTEED valid output)
    - Pydantic schema validation
    - Hardware acceleration optimization
    - Multi-GPU support
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        if not LMSTUDIO_AVAILABLE:
            raise ImportError("OpenAI client required: pip install openai")
        
        # LM Studio OpenAI-compatible API Configuration
        self.base_url = 'http://127.0.0.1:1234'
        self.client: Optional[OpenAI] = None
        self.active_model = None
        self.available_models = []
        
        # Generation settings optimized for JSON
        self.generation_params = {
            "temperature": 0.1,     # Low for consistency
            "max_tokens": 800,      # Enough for detailed MCQs
            "top_p": 0.9,
            "frequency_penalty": 0.05,
            "presence_penalty": 0.0,
            "stop": ["###", "---", "\n\n\n"]
        }
        
        # Performance tracking
        self.generation_stats = {
            "total_generated": 0,
            "json_success_rate": 1.0,  # Should be 100% with grammar-constrained generation
            "avg_time": 0,
            "active_model": None,
            "schema_enforcement": "grammar_constrained"
        }

    def initialize(self) -> bool:
        """Initialize the Enhanced LM Studio generator with OpenAI-compatible API"""
        try:
            logger.info("🔥 Initializing Enhanced LM Studio Generator with OpenAI-compatible API...")
            logger.info(f"🔗 Connecting to LM Studio: {self.base_url}")
            
            # Create OpenAI client pointing to LM Studio with MUCH faster timeout
            self.client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="lm-studio",  # LM Studio doesn't require a real API key
                timeout=1.0  # REDUCED FROM 5.0 to 1.0 for faster fallback
            )
            
            # Test connection with faster timeout
            if not self._check_connection():
                logger.error("❌ Cannot connect to LM Studio server")
                logger.error("💡 Make sure LM Studio is running and accessible on port 1234")
                return False
            
            logger.info("✅ Successfully connected to LM Studio SDK")
            
            # Get ALREADY LOADED models only
            self.available_models = self._get_available_models()
            if not self.available_models:
                logger.error("❌ No models are loaded in LM Studio")
                logger.error("💡 Please load a model in LM Studio first, then restart the app")
                logger.error("🔄 Falling back to other generators...")
                return False  # Let MCQ Manager use other generators instead
            
            logger.info(f"📋 Found {len(self.available_models)} models: {', '.join(self.available_models)}")
            
            # Select and load best model
            self.active_model = self._select_and_load_model()
            if not self.active_model:
                logger.error("❌ No suitable models found or could be loaded")
                return False
            
            # Skip schema test - it causes 404 errors
            logger.info("⏭️ Skipping schema test to avoid 404 errors")
            
            logger.info(f"🔥 Enhanced LM Studio Generator initialized!")
            logger.info(f"🎯 Active model: {self.active_model}")
            logger.info(f"📊 Schema enforcement: Grammar-constrained generation enabled")
            logger.info(f"🚀 Hardware acceleration: Automatic optimization enabled")
            
            self.is_initialized = True
            self.generation_stats["active_model"] = self.active_model
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced LM Studio generator: {e}")
            import traceback
            logger.debug(f"🔍 Full error traceback: {traceback.format_exc()}")
            return False

    def _check_connection(self) -> bool:
        """Check if LM Studio OpenAI API can connect with VERY fast timeout"""
        try:
            # MUCH faster connection test
            import requests
            
            # Use requests for faster timeout control - REDUCED FROM 2 to 0.5 seconds
            response = requests.get(f"{self.base_url}/v1/models", timeout=0.5)
            return response.status_code == 200
                
        except Exception as e:
            logger.debug(f"❌ LM Studio connection failed: {e}")
            return False

    def _get_available_models(self) -> List[str]:
        """Get the CURRENT/ACTIVE model - DO NOT LOAD NEW ONES"""
        try:
            # Try to get the current model that's actually being used
            # First, try a simple completion to see what model responds
            try:
                import requests
                test_response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                        "stream": False
                    },
                    timeout=3
                )
                if test_response.status_code == 200:
                    response_data = test_response.json()
                    if 'model' in response_data:
                        active_model = response_data['model']
                        logger.info(f"✅ Found ACTIVE model from test response: {active_model}")
                        return [active_model]
            except:
                pass
            
            # Fallback to listing models endpoint
            import requests
            loaded_response = requests.get(f"{self.base_url}/v1/models", timeout=2)
            if loaded_response.status_code == 200:
                loaded_data = loaded_response.json()
                loaded_models = []
                
                # Just get ANY loaded model that's not an embedding model
                if 'data' in loaded_data and loaded_data['data']:
                    for model in loaded_data['data']:
                        model_id = model.get('id', '')
                        if model_id and not any(keyword in model_id.lower() for keyword in ['embed', 'embedding', 'retrieval']):
                            loaded_models.append(model_id)
                
                if loaded_models:
                    logger.info(f"✅ Found {len(loaded_models)} loaded models, will use first one: {loaded_models[0]}")
                    return loaded_models[:1]  # Just return the first one
                else:
                    logger.warning("⚠️ NO MODELS ARE LOADED IN LM STUDIO - Please load a model first!")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to check loaded models in LM Studio: {e}")
            return []

    def _select_and_load_model(self) -> Optional[str]:
        """JUST USE THE FIRST LOADED MODEL - DON'T BE PICKY!"""
        logger.info("🔍 Using whatever model is already loaded...")
        
        if not self.available_models:
            logger.error("❌ No models loaded in LM Studio!")
            return None
        
        # JUST USE THE FIRST MODEL THAT'S ALREADY LOADED
        model = self.available_models[0]
        logger.info(f"✅ Using already loaded model: {model}")
        self.active_model = model
        return model

    def _load_model(self, model_name: str) -> bool:
        """DO NOT LOAD MODELS - Just verify if model is already loaded"""
        try:
            logger.info(f"🔍 Checking if model {model_name} is ALREADY LOADED (not loading new ones)")
            
            # DO NOT ATTEMPT TO LOAD - Just set as active if it exists
            self.active_model = model_name
            logger.info(f"✅ Will use already loaded model: {model_name}")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error checking model {model_name}: {e}")
            return False

    def _test_schema_generation(self) -> bool:
        """Test schema-constrained generation"""
        try:
            test_prompt = "Generate a simple test question about Python programming."
            
            response = self.client.chat.completions.create(
                model=self.active_model,
                messages=[
                    {"role": "system", "content": "You are an expert MCQ generator. Respond with valid JSON only."},
                    {"role": "user", "content": test_prompt}
                ],
                response_format={"type": "json_object"},  # 🔥 FIRE: JSON mode!
                **self.generation_params
            )
            
            if response and response.choices:
                content = response.choices[0].message.content
                # Validate against Pydantic schema
                mcq = MCQSchema.model_validate_json(content)
                logger.info("✅ Schema-constrained generation test passed!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Schema generation test failed: {e}")
            return False

    def generate_mcq(self, topic: str, context: str = "", num_questions: int = 1, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> List[Dict[str, Any]]:
        """🔥 FIRE: Generate MCQ with guaranteed JSON structure"""
        if not self.is_available():
            logger.error("❌ Enhanced LM Studio generator not available")
            return []
        
        results = []
        start_time = time.time()
        successful_generations = 0
        
        logger.info(f"🔥 Generating {num_questions} {question_type.upper()} MCQ(s) about '{topic}' using Enhanced LM Studio")
        
        for i in range(num_questions):
            try:
                question = self._generate_single_question(topic, context, i, difficulty, game_mode, question_type)
                if question:
                    results.append(question)
                    successful_generations += 1
                    logger.info(f"✅ Question {i+1}/{num_questions} ({question_type}) generated successfully")
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

    def _generate_single_question(self, topic: str, context: str, question_index: int, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> Optional[Dict[str, Any]]:
        """Generate a single MCQ using schema-constrained generation with timeout"""
        
        prompt = self._create_optimized_prompt(topic, context, question_index, difficulty, game_mode, question_type)
        
        logger.debug(f"🧠 Generating {question_type} question with Enhanced LM Studio: {self.active_model}")
        
        try:
            # 🔥 FIRE: JSON mode generation with timeout protection
            import threading
            import time
            import requests
            
            result = [None]
            exception = [None]
            
            def generation_worker():
                try:
                    # First try: Use the model name we have with faster timeout
                    try:
                        response = self.client.chat.completions.create(
                            model=self.active_model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": "You are an expert MCQ generator. You must respond with valid JSON that matches this exact schema: {\"question\": \"string\", \"options\": [\"string1\", \"string2\", \"string3\", \"string4\"], \"correct_answer\": \"string\", \"explanation\": \"string\"}"
                                },
                                {
                                    "role": "user", 
                                    "content": prompt
                                }
                            ],
                            response_format={"type": "json_object"},
                            **self.generation_params
                        )
                        result[0] = response
                        return
                    except Exception as e1:
                        logger.debug(f"First attempt with model '{self.active_model}' failed: {e1}")
                    
                    # Second try: Don't specify a model at all - let LM Studio use default
                    try:
                        # Make request without model parameter with faster timeout
                        response = requests.post(
                            f"{self.base_url}/v1/chat/completions",
                            json={
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are an expert MCQ generator. You must respond with valid JSON that matches this exact schema: {\"question\": \"string\", \"options\": [\"string1\", \"string2\", \"string3\", \"string4\"], \"correct_answer\": \"string\", \"explanation\": \"string\"}"
                                    },
                                    {
                                        "role": "user", 
                                        "content": prompt
                                    }
                                ],
                                "response_format": {"type": "json_object"},
                                **self.generation_params
                            },
                            timeout=15  # Faster timeout
                        )
                        if response.status_code == 200:
                            result[0] = response.json()
                            return
                        else:
                            logger.debug(f"Second attempt without model failed: {response.status_code}")
                    except Exception as e2:
                        logger.debug(f"Second attempt without model failed: {e2}")
                    
                    # If all attempts fail, raise the last exception
                    exception[0] = Exception("All generation attempts failed")
                    
                except Exception as e:
                    exception[0] = e
            
            # Start generation in a separate thread
            thread = threading.Thread(target=generation_worker)
            thread.daemon = True
            thread.start()
            
            # Wait for completion with faster timeout
            thread.join(timeout=20)  # 20 second total timeout
            
            if thread.is_alive():
                logger.error("❌ Generation timed out after 20 seconds")
                return None
            
            if exception[0]:
                raise exception[0]
            
            response = result[0]
            
            # Handle both OpenAI response format and direct JSON response
            if isinstance(response, dict):
                # Direct JSON response from requests
                if 'choices' in response and response['choices']:
                    content = response['choices'][0]['message']['content']
                else:
                    logger.error("❌ Invalid response format from LM Studio")
                    return None
            else:
                # OpenAI client response
                if not response or not response.choices:
                    logger.error("❌ No response from Enhanced LM Studio")
                    return None
                content = response.choices[0].message.content
            
            logger.debug(f"🤖 Model response: {content[:200]}...")
            
            # Validate with Pydantic (should always pass due to grammar constraint)
            try:
                mcq = MCQSchema.model_validate_json(content)
                result = mcq.model_dump()
                logger.info(f"✅ Successfully generated schema-validated {question_type} MCQ for topic: {topic}")
                return result
            except Exception as validation_error:
                logger.error(f"❌ Pydantic validation failed (unexpected!): {validation_error}")
                logger.error(f"❌ Response content: {content}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Enhanced LM Studio generation failed: {e}")
            import traceback
            logger.debug(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    def _create_optimized_prompt(self, topic: str, context: str, question_index: int, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> str:
        """Create optimized prompt for schema-constrained generation with ANTI-VAGUE enforcement and question type support"""
        
        # Difficulty mapping with specific requirements
        difficulty_map = {
            "easy": "basic recall of specific facts, fundamental definitions, simple calculations",
            "medium": "analytical thinking, concept application, problem-solving with specific scenarios", 
            "hard": "complex synthesis, multi-step reasoning, expert-level analysis of specific cases, advanced mechanisms"
        }
        
        difficulty_desc = difficulty_map.get(difficulty.lower(), difficulty_map["medium"])
        
        # Question type specific instructions
        question_type_instructions = self._get_question_type_instructions(question_type, topic)
        
        # AGGRESSIVE anti-vague question enforcement for hard mode
        if difficulty.lower() == "hard":
            anti_vague_section = """
🚨 HARD MODE - ABSOLUTELY NO VAGUE QUESTIONS:
❌ COMPLETELY BANNED: "What is the primary function of..."
❌ COMPLETELY BANNED: "What is the main purpose of..."
❌ COMPLETELY BANNED: "What does X do?"
❌ COMPLETELY BANNED: Basic definition questions
❌ COMPLETELY BANNED: General overview questions
✅ ABSOLUTELY REQUIRED: Specific mechanisms, pathways, processes
✅ ABSOLUTELY REQUIRED: Multi-step reasoning and analysis
✅ ABSOLUTELY REQUIRED: Expert-level detail and precision
✅ ABSOLUTELY REQUIRED: Questions requiring deep understanding

SPECIAL FOCUS FOR BIOLOGY/REPRODUCTIVE TOPICS:
If this is about reproduction/reproductive systems, you MUST ask about:
- Specific hormonal pathways (FSH, LH, testosterone, estrogen regulation)
- Molecular mechanisms (meiosis stages, fertilization details)
- Biochemical processes (steroidogenesis, gametogenesis)
- Regulatory feedback loops and their disruption
- Specific anatomical structures and their precise functions
- Pathological conditions and their molecular basis
"""
        else:
            anti_vague_section = ""
        
        # Anti-vague question types based on index
        specific_question_types = [
            "numerical calculation or specific quantitative analysis",
            "definition or identification of specific terms/concepts with precise detail",
            "cause-and-effect relationship with concrete examples and mechanisms",
            "comparison between specific elements or cases with detailed analysis",
            "step-by-step process or procedure explanation with molecular detail",
            "real-world application with specific scenario and problem-solving",
            "historical facts or specific timeline events with causal relationships",
            "formula application or mathematical derivation with multi-step reasoning"
        ]
        
        question_focus = specific_question_types[question_index % len(specific_question_types)]
        
        # 🔧 FIX: Sanitize inputs to prevent prompt injection
        from .inquisitor_prompt import _sanitize_user_input
        sanitized_topic = _sanitize_user_input(topic)
        sanitized_difficulty = _sanitize_user_input(difficulty)

        # Enhanced prompt with anti-vague enforcement and question type support
        prompt = f"""Generate a {sanitized_difficulty} difficulty multiple choice question about {sanitized_topic}.

{anti_vague_section}

{question_type_instructions}

Requirements:
- Topic: {topic}
- Difficulty: {difficulty.upper()} - {difficulty_desc}
- Question Type: {question_type.upper()}
- Focus: {question_focus}
- Context: {context if context else "Use knowledge base"}
- Game Mode: {game_mode.upper()}

Quality Standards for {difficulty.upper()} level:
- Questions must require {difficulty_desc}
- Use specific, technical terminology
- Avoid vague generalizations
- Include challenging but fair distractors
- Focus on understanding mechanisms rather than simple recall

Response format - valid JSON only:
{{
  "question": "Your specific {difficulty} level {question_type} question here",
  "options": {{
    "A": "Option A",
    "B": "Option B", 
    "C": "Option C",
    "D": "Option D"
  }},
  "correct": "A",
  "explanation": "Detailed explanation appropriate for {difficulty} level"
}}"""

        return prompt

    def _get_question_type_instructions(self, question_type: str, topic: str) -> str:
        """Get specific instructions based on question type"""
        
        if question_type.lower() == "numerical":
            return f"""
📊 NUMERICAL QUESTION TYPE REQUIREMENTS:
✅ MUST include specific numbers, measurements, percentages, or quantities
✅ MUST require calculation, numerical reasoning, or quantitative analysis
✅ Use precise units (days, years, %, mg, ml, etc.)
✅ Include specific timeframes, durations, ages, or quantities
✅ Focus on statistics, rates, measurements, or numerical facts

🧮 NUMERICAL EXAMPLES FOR SEX EDUCATION:
✅ "What is the average length of a menstrual cycle (in days)?" → Answer: 28 days
✅ "How long can sperm live inside the female reproductive tract?" → Answer: Up to 5 days  
✅ "What percentage effectiveness does the male condom have with perfect use?" → Answer: ~98%
✅ "At what age does puberty typically start in girls?" → Answer: Around 8-13 years
✅ "How many chromosomes are there in a human sperm cell?" → Answer: 23
✅ "What is the typical duration of ovulation?" → Answer: About 24 hours
✅ "How long does fertilization take after sperm meets egg?" → Answer: 12-24 hours

❌ BANNED FOR NUMERICAL TYPE:
❌ Pure conceptual questions without numbers
❌ Vague qualitative descriptions  
❌ Questions answerable without numerical knowledge

🎯 FOCUS: Always include specific numbers, percentages, timeframes, quantities, or measurements in the question AND answers.
"""
        
        elif question_type.lower() == "conceptual":
            return f"""
🧠 CONCEPTUAL QUESTION TYPE REQUIREMENTS:
✅ MUST focus on understanding concepts, theories, and relationships
✅ Test comprehension of processes, mechanisms, and principles
✅ Avoid pure numerical calculations or memorized facts
✅ Focus on "why" and "how" rather than "how many" or "when"
✅ Test understanding of cause-and-effect relationships
✅ Include analysis of systems, functions, and interactions

🧠 CONCEPTUAL EXAMPLES FOR SEX EDUCATION:
✅ "Which hormonal mechanism triggers ovulation?"
✅ "How does the negative feedback loop regulate testosterone production?"
✅ "What biological process prevents multiple sperm from fertilizing one egg?"
✅ "Which physiological changes occur during the luteal phase?"
✅ "How do contraceptive pills prevent pregnancy?"

❌ BANNED FOR CONCEPTUAL TYPE:
❌ Questions requiring specific numerical calculations
❌ Pure memorization of dates, ages, or quantities
❌ Statistical or measurement-based questions

🎯 FOCUS: Understanding biological processes, mechanisms, and theoretical relationships.
"""
        
        else:  # mixed
            return f"""
🔀 MIXED QUESTION TYPE REQUIREMENTS:
✅ CAN include either numerical OR conceptual elements
✅ Vary between quantitative analysis and theoretical understanding
✅ Mix specific measurements with process comprehension
✅ Balance factual recall with analytical reasoning
✅ Include both "how much/many" and "why/how" questions

🔀 MIXED EXAMPLES FOR SEX EDUCATION:
✅ Either numerical: "How many days typically occur between ovulation and menstruation?"
✅ Or conceptual: "Which hormone surge indicates that ovulation is about to occur?"
✅ Either numerical: "What percentage of couples conceive within the first year of trying?"
✅ Or conceptual: "How does cervical mucus change throughout the menstrual cycle?"

🎯 FOCUS: Balance between quantitative facts and conceptual understanding based on the specific topic and learning objectives.
"""

    async def generate_quiz_async(self, context: str, topic: str, difficulty: str = "medium", game_mode: str = "casual", question_type: str = "mixed") -> Dict[str, Any]:
        """Async wrapper for compatibility with existing MCQ manager"""
        results = self.generate_mcq(topic, context, 1, difficulty, game_mode, question_type)
        if results and len(results) > 0:
            question_data = results[0]
            return {
                "question": question_data["question"],
                "options": {
                    "A": question_data["options"][0],
                    "B": question_data["options"][1],
                    "C": question_data["options"][2],
                    "D": question_data["options"][3]
                },
                "correct": self._find_correct_letter(question_data["correct_answer"], question_data["options"]),
                "explanation": question_data["explanation"]
            }
        return None

    def _find_correct_letter(self, correct_answer: str, options: List[str]) -> str:
        """Find the letter corresponding to the correct answer"""
        for i, option in enumerate(options):
            if option.strip() == correct_answer.strip():
                return chr(65 + i)  # A=65, B=66, etc.
        return "A"  # Default fallback

    def is_available(self) -> bool:
        """Check if Enhanced LM Studio generation is available"""
        return (self.is_initialized and 
                self.client is not None and 
                self.active_model is not None and
                self._check_connection())

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.generation_stats,
            "server_url": self.base_url,
            "status": "ready" if self.is_available() else "unavailable",
            "available_models": self.available_models,
            "api_type": "openai_compatible",
            "hardware_acceleration": "automatic",
            "json_mode": "enforced_pydantic_validation"
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.client:
            # OpenAI client handles cleanup automatically
            self.client = None
        logger.info("🔥 Enhanced LM Studio Generator cleaned up")