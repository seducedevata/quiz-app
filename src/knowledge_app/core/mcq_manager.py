"""
MCQ Manager - Unified interface for online and offline MCQ generation
[HOT] FIRE: Now uses UnifiedInferenceManager for all AI operations
Eliminates race conditions and provides consistent error handling
[BRAIN] ENHANCED: Now includes semantic topic mapping with Phi model
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional
import asyncio

from .unified_inference_manager import (
    get_unified_inference_manager,
    generate_mcq_unified,
    initialize_unified_inference,
    get_inference_status
)
from .mcq_coherence_monitor import get_coherence_monitor

logger = logging.getLogger(__name__)

class MCQManager:
    """
    [HOT] FIRE Unified MCQ Manager - Single Source of Truth for MCQ Generation
    
    Now uses the UnifiedInferenceManager for all AI operations:
    - Eliminates race conditions between multiple model managers
    - Provides consistent error handling
    - Thread-safe operation with proper timeout handling
    - Centralized resource management
    """

    # noinspection PyPackageRequirements
    def __init__(self, config=None):
        self.config = config

        # [START] ULTRA LAZY: Defer ALL heavy operations until actually needed
        self._unified_manager = None  # Lazy load
        self._unified_manager_initialized = False
        self._initialized = False
        self._initialization_lock = threading.RLock()

        # [BRAIN] SEMANTIC MAPPING: Lazy initialization - defer heavy operations
        self._topic_analyzer = None
        self._semantic_mapper = None
        self._semantic_processing_initialized = False

        # [START] GOLDEN PATH: Lazy initialization - defer heavy operations
        self._rag_engine = None
        self._rag_engine_initialized = False

        # MCQ quality enhancement - lightweight
        self._coherence_monitor = None  # Lazy load

        # Content filtering - lightweight
        self._inappropriate_topics = {
            'violence', 'nsfw', 'illegal', 'harmful', 'explicit',
            'gore', 'terrorism', 'drugs', 'weapons', 'hate'
        }

        # DeepSeek integration removed - using BatchTwoModelPipeline via unified_inference_manager

        logger.info("[OK] MCQManager created with ULTRA LAZY initialization (instant startup)")

    def get_best_available_generator(self):
        """🔧 FIX: Delegate to UnifiedInferenceManager for consistent model selection

        This eliminates conflicting AI generation logic by making UnifiedInferenceManager
        the single source of truth for all model selection decisions.
        """
        try:
            # 🔧 FIX: Always delegate to UnifiedInferenceManager
            self._ensure_unified_manager()

            if self._unified_manager:
                # Let UnifiedInferenceManager handle all model selection logic
                status = self._unified_manager.get_status()
                if status.get("state") == "ready":
                    logger.info("[OK] UnifiedInferenceManager is ready - delegating model selection")
                    return self._unified_manager  # Return the manager itself as the "generator"
                else:
                    logger.warning(f"[WARNING] UnifiedInferenceManager not ready: {status}")
                    return None
            else:
                logger.error("[ERROR] UnifiedInferenceManager not available")
                return None

        except Exception as e:
            logger.error(f"[ERROR] Error delegating to UnifiedInferenceManager: {e}")
            return None

    def _ensure_unified_manager(self):
        """Lazy initialization of unified inference manager"""
        if self._unified_manager_initialized:
            return

        try:
            logger.info("[CONFIG] Lazy loading unified inference manager...")
            from .unified_inference_manager import get_unified_inference_manager
            self._unified_manager = get_unified_inference_manager()
            self._unified_manager_initialized = True
            logger.info("[OK] Unified inference manager initialized")
        except Exception as e:
            logger.error(f"[ERROR] Unified inference manager initialization failed: {e}")
            self._unified_manager = None
            self._unified_manager_initialized = True

    def _ensure_semantic_processing(self):
        """Lazy initialization of semantic processing components"""
        if self._semantic_processing_initialized:
            return

        try:
            logger.info("[BRAIN] Lazy loading semantic processing components...")

            # Initialize semantic mapper (uses Phi model)
            from .intelligent_semantic_mapper import get_semantic_mapper
            self._semantic_mapper = get_semantic_mapper()
            logger.info("[BRAIN] Semantic mapper initialized with Phi model")

            # Initialize topic analyzer with semantic mapping enabled
            from .topic_analyzer import TopicAnalyzer
            self._topic_analyzer = TopicAnalyzer(use_semantic_mapping=True)
            logger.info("[TARGET] Topic analyzer initialized with semantic mapping enabled")

            self._semantic_processing_initialized = True

        except Exception as e:
            logger.error(f"[ERROR] Semantic processing initialization failed: {e}")
            # Use fallback simple topic analyzer
            from .topic_analyzer import TopicAnalyzer
            self._topic_analyzer = TopicAnalyzer(use_semantic_mapping=False)
            self._semantic_processing_initialized = True

    def _ensure_rag_engine(self):
        """Lazy initialization of RAG engine"""
        if self._rag_engine_initialized:
            return

        try:
            logger.info("[START] Lazy loading RAG engine...")
            from .rag_engine import get_rag_engine
            self._rag_engine = get_rag_engine()
            self._rag_engine_initialized = True
            logger.info("[OK] RAG engine initialized")
        except Exception as e:
            logger.error(f"[ERROR] RAG engine initialization failed: {e}")
            self._rag_engine = None
            self._rag_engine_initialized = True

    def _ensure_deepseek_pipeline(self):
        """Legacy method - DeepSeek integration removed, using BatchTwoModelPipeline instead"""
        # DeepSeek integration has been removed - BatchTwoModelPipeline is used via unified_inference_manager
        self.deepseek_pipeline = None
        self._deepseek_initialized = True
        logger.info("[EXPERT] Using BatchTwoModelPipeline via unified_inference_manager for expert mode")

    def _ensure_coherence_monitor(self):
        """Lazy initialization of coherence monitor"""
        if self._coherence_monitor is None:
            try:
                self._coherence_monitor = get_coherence_monitor()
            except Exception as e:
                logger.error(f"[ERROR] Coherence monitor initialization failed: {e}")
                self._coherence_monitor = None

    def generate_quiz(self, quiz_params: Dict[str, Any]):
        """
        [START] DEPRECATED: Use generate_quiz_async instead to prevent UI blocking

        This method is kept for backward compatibility but should not be used
        from UI threads as it can cause freezing.
        """
        logger.warning("[WARNING] DEPRECATED: generate_quiz() called - use generate_quiz_async() instead")
        try:
            # Run async generation in thread pool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.generate_quiz_async(quiz_params))
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"[ERROR] Sync quiz generation failed: {e}")
            return None

    def get_available_lora_adapters(self) -> List[str]:
        """Get list of available LoRA adapters for UI selection - DISABLED"""
        # [FORBIDDEN] LORA DISABLED: Use normal Ollama models instead
        logger.info("[LIST] LoRA adapters disabled - using normal Ollama models")
        return []
    
    def _get_rag_context(self, topic: str) -> str:
        """Get RAG context for a topic - lazy initialize RAG engine if needed"""
        try:
            # Ensure RAG engine is loaded
            self._ensure_rag_engine()

            if self._rag_engine:
                # Use RAG engine to get context
                return self._rag_engine.get_context(topic)
            else:
                # Fallback to simple context
                return f"Context for {topic}: Use your knowledge base to create relevant questions."

        except Exception as e:
            logger.warning(f"[WARNING] RAG context generation failed: {e}")
            return f"Use your knowledge about {topic} to create questions."

    def _preprocess_topic_with_semantic_mapping(self, raw_topic: str) -> Dict[str, Any]:
        """
        [BRAIN] SEMANTIC PREPROCESSING: Process topic through Phi model for intelligent analysis

        This is the missing piece that ensures topics are properly analyzed before generation.

        Args:
            raw_topic: Raw user input topic

        Returns:
            Dict with processed topic info and semantic analysis
        """
        try:
            logger.info(f"[BRAIN] SEMANTIC PREPROCESSING: Processing topic '{raw_topic}'")

            # Ensure semantic processing components are loaded
            self._ensure_semantic_processing()

            # Step 1: Use semantic mapper for intelligent analysis
            if self._semantic_mapper:
                semantic_result = self._semantic_mapper.map_topic_semantically(raw_topic)
                logger.info(f"[TARGET] Semantic mapping: '{raw_topic}' → '{semantic_result.expanded_topic}' ({semantic_result.question_type}, confidence: {semantic_result.confidence:.2f})")

                # Step 2: Use topic analyzer for additional analysis
                if self._topic_analyzer:
                    topic_profile = self._topic_analyzer.get_topic_profile(semantic_result.expanded_topic)
                    logger.info(f"[STATS] Topic analysis: {topic_profile.get('detected_type', 'unknown')} (confidence: {topic_profile.get('confidence', 'unknown')})")

                    # Combine semantic mapping and topic analysis results
                    return {
                        "original_topic": raw_topic,
                        "processed_topic": semantic_result.expanded_topic,
                        "semantic_analysis": {
                            "question_type": semantic_result.question_type,
                            "confidence": semantic_result.confidence,
                            "reasoning": semantic_result.reasoning,
                            "is_abbreviation": semantic_result.is_abbreviation,
                            "full_form": semantic_result.full_form
                        },
                        "topic_profile": topic_profile,
                        "recommended_question_type": semantic_result.question_type,
                        "processing_method": "semantic_mapping"
                    }
                else:
                    # Only semantic mapping available
                    return {
                        "original_topic": raw_topic,
                        "processed_topic": semantic_result.expanded_topic,
                        "semantic_analysis": {
                            "question_type": semantic_result.question_type,
                            "confidence": semantic_result.confidence,
                            "reasoning": semantic_result.reasoning,
                            "is_abbreviation": semantic_result.is_abbreviation,
                            "full_form": semantic_result.full_form
                        },
                        "recommended_question_type": semantic_result.question_type,
                        "processing_method": "semantic_only"
                    }
            else:
                logger.warning("[WARNING] Semantic mapper not available, using basic processing")
                return {
                    "original_topic": raw_topic,
                    "processed_topic": raw_topic,
                    "recommended_question_type": "mixed",
                    "processing_method": "basic"
                }

        except Exception as e:
            logger.error(f"[ERROR] Semantic preprocessing failed for '{raw_topic}': {e}")
            return {
                "original_topic": raw_topic,
                "processed_topic": raw_topic,
                "recommended_question_type": "mixed",
                "processing_method": "error_recovery",
                "error": str(e)
            }

    def _ensure_initialized(self):
        """Ensure the unified inference manager is initialized - PREVENTS RE-INITIALIZATION"""
        if self._initialized:
            return True

        with self._initialization_lock:
            if self._initialized:
                return True

            # Ensure unified manager is loaded first
            self._ensure_unified_manager()

            if not self._unified_manager:
                logger.error("[ERROR] Failed to initialize unified manager")
                return False

            # [HOT] CRITICAL FIX: Check if UnifiedInferenceManager is already ready
            from .unified_inference_manager import get_inference_status, initialize_unified_inference
            current_status = get_inference_status()
            if current_status.get("state") == "ready":
                logger.info("[START] UnifiedInferenceManager already ready - skipping re-initialization")
                self._initialized = True
                return True

            # [START] GOLDEN PATH FIX: Always ensure UnifiedInferenceManager is properly initialized
            logger.info("[RELOAD] Initializing UnifiedInferenceManager for Golden Path MCQ generation...")
            success = initialize_unified_inference(self.config)

            if success:
                self._initialized = True
                logger.info("[START] UnifiedInferenceManager ready for Golden Path generation")

                # Verify Ollama is actually detected
                final_status = get_inference_status()
                logger.info(f"[SEARCH] Final UIM status: {final_status}")
                return True
            else:
                logger.error("[ERROR] Failed to initialize UnifiedInferenceManager")
                return False
    
    def _ensure_generators_initialized(self):
        """Ensure generators are initialized (legacy compatibility method)"""
        return self._ensure_initialized()
    
    async def generate_quiz_async(self, quiz_params: Dict[str, Any]):
        """
        [START] ASYNC GOLDEN PATH: Generate grounded MCQ using RAG + Semantic Mapping

        Follows the "Grounded Scholar" pipeline ASYNCHRONOUSLY:
        1. Semantic preprocessing with Phi model (ASYNC)
        2. Retrieve relevant context from user's documents using RAG
        3. Generate MCQ grounded in retrieved context (ASYNC)
        4. Validate and post-process for quality
        """
        try:
            # [TARGET] COMPREHENSIVE GENERATION LOGGING - START
            logger.info("="*80)
            logger.info("[TARGET] QUESTION GENERATION SESSION STARTED")
            logger.info("="*80)
            
            # Extract and validate parameters
            topic = quiz_params.get('topic', 'General Knowledge')
            difficulty = quiz_params.get('difficulty', 'medium')
            question_type = quiz_params.get('submode', 'mixed')
            # [FORBIDDEN] LORA DISABLED: No adapter needed - use normal Ollama models
            adapter_name = None
            mode = quiz_params.get('mode', 'auto')  # [HOT] FIX: Extract mode from parameters
            num_questions = quiz_params.get('num_questions', 1)
            game_mode = quiz_params.get('game_mode', 'casual')
            timer_setting = quiz_params.get('timer', '30s')

            # [BRAIN] SEMANTIC PREPROCESSING: Process topic through Phi model FIRST (ASYNC)
            logger.info("[BRAIN] STEP 1: SEMANTIC TOPIC PREPROCESSING (ASYNC)")

            # Run semantic preprocessing in thread pool to avoid blocking UI
            loop = asyncio.get_event_loop()
            topic_analysis = await loop.run_in_executor(
                None,
                self._preprocess_topic_with_semantic_mapping,
                topic
            )

            # Update topic and question_type based on semantic analysis
            original_topic = topic
            processed_topic = topic_analysis.get('processed_topic', topic)
            recommended_question_type = topic_analysis.get('recommended_question_type', question_type)

            # Use processed topic for generation
            topic = processed_topic

            # Override question_type if semantic analysis provides better recommendation
            if question_type == 'mixed' and recommended_question_type != 'mixed':
                logger.info(f"[TARGET] Semantic analysis recommends '{recommended_question_type}' over '{question_type}'")
                question_type = recommended_question_type

            # Log semantic analysis results
            logger.info("[BRAIN] SEMANTIC ANALYSIS RESULTS:")
            logger.info(f"   • ORIGINAL_TOPIC: '{original_topic}'")
            logger.info(f"   • PROCESSED_TOPIC: '{processed_topic}'")
            logger.info(f"   • PROCESSING_METHOD: {topic_analysis.get('processing_method', 'unknown')}")
            if 'semantic_analysis' in topic_analysis:
                semantic = topic_analysis['semantic_analysis']
                logger.info(f"   • SEMANTIC_CONFIDENCE: {semantic.get('confidence', 'unknown')}")
                logger.info(f"   • SEMANTIC_REASONING: {semantic.get('reasoning', 'unknown')}")
                logger.info(f"   • IS_ABBREVIATION: {semantic.get('is_abbreviation', False)}")
                if semantic.get('full_form'):
                    logger.info(f"   • FULL_FORM: {semantic.get('full_form')}")

            # [BRAIN] EXPERT MODE: Auto-detect DeepSeek usage for expert difficulty
            use_deepseek = (difficulty == 'expert')
            if use_deepseek:
                logger.info("[BRAIN] EXPERT DIFFICULTY DETECTED - Using DeepSeek pipeline automatically")

            # [TARGET] LOG ALL GENERATION PARAMETERS (UPDATED)
            logger.info("[LIST] FINAL GENERATION PARAMETERS:")
            logger.info(f"   • ORIGINAL_TOPIC: '{original_topic}'")
            logger.info(f"   • PROCESSED_TOPIC: '{topic}'")
            logger.info(f"   • DIFFICULTY: '{difficulty}'")
            logger.info(f"   • QUESTION_TYPE: '{question_type}' {'(SEMANTIC OVERRIDE)' if question_type != quiz_params.get('submode', 'mixed') else ''}")
            logger.info(f"   • MODE: '{mode}'")
            logger.info(f"   • NUM_QUESTIONS: {num_questions}")
            logger.info(f"   • GAME_MODE: '{game_mode}'")
            logger.info(f"   • TIMER_SETTING: '{timer_setting}'")
            logger.info(f"   • LORA_ADAPTER: '{adapter_name}' {'(ENABLED)' if adapter_name else '(NONE)'}")
            
            # [TARGET] LOG SYSTEM AVAILABILITY STATUS
            logger.info("[SEARCH] SYSTEM AVAILABILITY CHECK:")
            system_status = self.get_system_status()
            logger.info(f"   • UNIFIED_MANAGER_STATUS: {system_status.get('status', 'unknown')}")
            logger.info(f"   • OFFLINE_AVAILABLE: {self.is_offline_available()}")
            logger.info(f"   • ONLINE_AVAILABLE: {self.is_online_available()}")
            
            # Ensure unified manager is initialized
            if not self._ensure_initialized():
                logger.error("[ERROR] UnifiedInferenceManager not available")
                logger.error("="*80)
                return None
            
            # [TARGET] DETERMINE AND LOG GENERATOR SELECTION
            logger.info("[START] GENERATOR SELECTION LOGIC:")
            
            inference_status = get_inference_status()
            local_available = inference_status.get("local_available", False)
            cloud_available = inference_status.get("cloud_available", False)
            
            logger.info(f"   • LOCAL_MODELS_AVAILABLE: {local_available}")
            logger.info(f"   • CLOUD_APIS_AVAILABLE: {cloud_available}")
            
            # Determine which generator will be used based on mode and availability
            selected_generator = "unknown"
            generator_reason = "unknown"
            
            if mode == "offline":
                if local_available:
                    selected_generator = "OFFLINE (Local Models)"
                    generator_reason = "User selected offline mode and local models available"
                else:
                    selected_generator = "NONE - OFFLINE REQUESTED BUT UNAVAILABLE"
                    generator_reason = "User requested offline but no local models available"
            elif mode == "online":
                if cloud_available:
                    selected_generator = "ONLINE (Cloud APIs)"
                    generator_reason = "User selected online mode and cloud APIs available"
                else:
                    selected_generator = "NONE - ONLINE REQUESTED BUT UNAVAILABLE" 
                    generator_reason = "User requested online but no cloud APIs available"
            elif mode == "auto":
                if local_available and cloud_available:
                    selected_generator = "AUTO (Both Available - Will Auto-Select)"
                    generator_reason = "Auto mode with both local and cloud available"
                elif local_available:
                    selected_generator = "AUTO -> OFFLINE (Only Local Available)"
                    generator_reason = "Auto mode defaulting to local models"
                elif cloud_available:
                    selected_generator = "AUTO -> ONLINE (Only Cloud Available)"
                    generator_reason = "Auto mode defaulting to cloud APIs"
                else:
                    selected_generator = "NONE - NO GENERATORS AVAILABLE"
                    generator_reason = "No local models or cloud APIs available"
            
            logger.info(f"[TARGET] SELECTED_GENERATOR: {selected_generator}")
            logger.info(f"[TARGET] SELECTION_REASON: {generator_reason}")
            
            # [HOT] CRITICAL FIX: Set the inference mode based on user selection
            self._ensure_unified_manager()
            if self._unified_manager:
                self._unified_manager.set_inference_mode(mode)
                logger.info(f"[TARGET] Mode set to: {mode} for MCQ generation")

            # [START] BUG FIX 32: Apply content filtering BEFORE any topic resolution
            # This prevents inappropriate topics from being laundered through the resolver fallback
            raw_topic_for_filtering = topic  # Keep original for filtering
            filtered_result = self._comprehensive_content_filter(raw_topic_for_filtering)

            if not filtered_result["is_safe"]:
                logger.warning(f"🛡️ CONTENT FILTER BLOCKED: '{topic}' - {filtered_result['reason']}")
                # Immediately abort generation for inappropriate content
                raise ValueError(f"Content policy violation: {filtered_result['reason']}")

            if filtered_result["topic"] != topic:
                logger.info(f"🛡️ Content filter applied: '{topic}' → '{filtered_result['topic']}'")
                topic = filtered_result["topic"]

            logger.info(f"[TARGET] GOLDEN PATH: Generating MCQ: topic='{topic}', difficulty='{difficulty}', adapter='{adapter_name}'")

            # GOLDEN PATH DECISION TREE 🌟
            context_for_generation = None
            generation_method = "raw_model"

            # STEP 1: PRIORITIZE LORA ADAPTER (Domain Specialization) [LINK]
            if adapter_name:
                logger.info(f"[LINK] GOLDEN PATH: Using LoRA adapter '{adapter_name}' for domain specialization")
                generation_method = "lora_specialized"
                # LoRA adapters don't need RAG context - they ARE the domain knowledge
                context_for_generation = None

            # STEP 2: RAG CONTEXT (Document Grounding) 📚
            elif self._rag_engine and self._rag_engine.is_initialized:
                logger.info(f"📚 GOLDEN PATH: No LoRA adapter specified, using RAG context for grounding")
                retrieved_context = self._retrieve_grounded_context(topic)
                if retrieved_context:
                    context_for_generation = "\n\n---\n\n".join(retrieved_context)
                    generation_method = "rag_grounded"
                    logger.info(f"📚 Retrieved {len(retrieved_context)} context chunks ({len(context_for_generation)} chars)")
                else:
                    logger.warning(f"[WARNING] No RAG context found for '{topic}'")

            # STEP 3: RAW MODEL GENERATION [AI]
            else:
                logger.info(f"[AI] GOLDEN PATH: Using raw model generation (no LoRA or RAG available)")

            # [TARGET] LOG FINAL GENERATION METHOD
            logger.info("[START] FINAL GENERATION SETUP:")
            logger.info(f"   • GENERATION_METHOD: {generation_method}")
            logger.info(f"   • CONTEXT_PROVIDED: {'YES' if context_for_generation else 'NO'}")
            logger.info(f"   • CONTEXT_LENGTH: {len(context_for_generation) if context_for_generation else 0} chars")
            logger.info(f"   • LORA_ADAPTER: '{adapter_name}' {'(ACTIVE)' if adapter_name else '(NONE)'}")

            # [BRAIN] EXPERT MODE: Now handled by unified_inference_manager with BatchTwoModelPipeline
            if use_deepseek:
                logger.info("[BRAIN] Expert mode will use BatchTwoModelPipeline via unified_inference_manager...")
                # DeepSeek integration removed - expert mode now uses BatchTwoModelPipeline automatically
                result = None
            else:
                result = None

            # Use unified inference if DeepSeek failed or not expert mode
            if not result:
                logger.info(f"[START] Generating MCQ using {generation_method} method...")
                logger.info("[RELOAD] Calling UnifiedInferenceManager...")

                # [TARGET] PASS SEMANTIC GENERATION INSTRUCTIONS TO UNIFIED INFERENCE
                generation_instructions = None
                if 'semantic_analysis' in topic_analysis:
                    generation_instructions = topic_analysis['semantic_analysis'].get('reasoning', None)
                    logger.info(f"[BRAIN] Using Phi-generated instructions: {generation_instructions}")

                # [START] ASYNC GENERATION: Use async unified inference to prevent UI blocking
                from .unified_inference_manager import InferenceRequest

                # Create async request
                request = InferenceRequest(
                    request_id=f"mcq_async_{int(time.time() * 1000)}",
                    operation="generate_mcq",
                    params={
                        "topic": topic,
                        "difficulty": difficulty,
                        "question_type": question_type,
                        "context": context_for_generation,
                        "adapter_name": None,  # [FORBIDDEN] LORA DISABLED
                        "generation_instructions": generation_instructions
                    },
                    timeout=90.0
                )

                # Call async handler directly
                self._ensure_unified_manager()
                if self._unified_manager:
                    result = await self._unified_manager._handle_mcq_generation(request)
                else:
                    logger.error("[ERROR] Unified manager not available")
                    return None

            if result:
                # STEP 4: VALIDATION AND POST-PROCESSING [OK]
                # Handle DeepSeek MCQResult objects vs regular dict results
                if isinstance(result, MCQResult):
                    # DeepSeek already returned a formatted MCQResult
                    mcq_result = result
                    logger.info("[OK] Using pre-formatted DeepSeek MCQResult")
                else:
                    # Regular dict result needs validation and formatting
                    mcq_result = self._validate_and_format_result(result, topic, quiz_params)

                if mcq_result:
                    # Add Golden Path metadata (update if DeepSeek, set if regular)
                    if not hasattr(mcq_result, 'generation_method') or mcq_result.generation_method == "raw_model":
                        mcq_result.generation_method = generation_method
                    mcq_result.grounded = (generation_method in ["lora_specialized", "rag_grounded", "batch_two_model_pipeline"])
                    mcq_result.context_chunks = len(context_for_generation.split('\n\n---\n\n')) if context_for_generation else 0
                    mcq_result.adapter_used = adapter_name

                    logger.info(f"[OK] GOLDEN PATH: {mcq_result.generation_method} MCQ generated and validated successfully")
                    logger.info("="*80)
                    logger.info("[TARGET] QUESTION GENERATION SESSION COMPLETED SUCCESSFULLY")
                    logger.info("="*80)

                    # [HOT] CRITICAL FIX: Convert MCQResult to dictionary for UI compatibility
                    return {
                        "question": mcq_result.question,
                        "options": mcq_result.options,
                        "correct_answer": mcq_result.correct_answer,
                        "explanation": mcq_result.explanation,
                        "difficulty": getattr(mcq_result, 'difficulty', 'medium'),
                        "generation_method": getattr(mcq_result, 'generation_method', 'unknown'),
                        "grounded": getattr(mcq_result, 'grounded', False),
                        "context_chunks": getattr(mcq_result, 'context_chunks', 0),
                        "adapter_used": getattr(mcq_result, 'adapter_used', None)
                    }
                else:
                    logger.error("[ERROR] MCQ validation failed - no question generated")
                    logger.error("="*80)
                    return None
            else:
                logger.error("[ERROR] UnifiedInferenceManager returned no result")
                logger.error("="*80)
                return None

        except Exception as e:
            logger.error(f"[ERROR] GOLDEN PATH MCQ generation failed: {e}")
            logger.error("="*80)
            return None
    
    def _retrieve_grounded_context(self, topic: str, top_k: int = 3) -> List[str]:
        """
        STEP 1 of Golden Path: Retrieve relevant context using RAG
        
        Returns:
            List of relevant text chunks from user's documents
        """
        try:
            if not self._rag_engine:
                logger.warning("📚 RAG engine not available for context retrieval")
                return []
            
            logger.info(f"[SEARCH] Retrieving context for topic: '{topic}'")
            contexts = self._rag_engine.retrieve_context(topic, top_k=top_k)
            
            if contexts:
                logger.info(f"[OK] Retrieved {len(contexts)} relevant context chunks")
                # Log first chunk preview for debugging
                if len(contexts) > 0:
                    preview = contexts[0][:200] + "..." if len(contexts[0]) > 200 else contexts[0]
                    logger.debug(f"📖 Context preview: {preview}")
                return contexts
            else:
                logger.warning(f"[WARNING] No relevant context found for topic: '{topic}'")
                return []
                
        except Exception as e:
            logger.error(f"[ERROR] Context retrieval failed: {e}")
            return []
    
    def _validate_and_format_result(self, result: Dict[str, Any], topic: str, quiz_params: Dict[str, Any]) -> Optional['MCQResult']:
        """Validate and format MCQ result with coherence monitoring"""
        try:
            # Handle both single dict and list formats from different generators
            if isinstance(result, list) and len(result) > 0:
                # Ollama generator returns a list - use first item
                result_dict = result[0]
                logger.debug("[OK] Using first item from list result")
            elif isinstance(result, dict):
                # Direct dict format from some generators
                result_dict = result
                logger.debug("[OK] Using direct dict result")
            else:
                logger.error(f"[ERROR] Invalid result format: {type(result)}")
                return None
            
            # Extract data from result
            question = result_dict.get('question', '')
            options = result_dict.get('options', [])
            # CRITICAL FIX: Handle both field names for correct answer
            correct_answer = result_dict.get('correct_answer', result_dict.get('correct', ''))
            explanation = result_dict.get('explanation', 'No explanation available.')
            
            # Basic validation
            if not question or len(options) < 4 or not correct_answer:
                logger.error("[ERROR] MCQ result missing required fields")
                return None
            
            # [EMERGENCY] COHERENCE MONITORING - Detect AI spaghetti in real-time!
            self._ensure_coherence_monitor()
            if self._coherence_monitor:
                is_coherent, issues = self._coherence_monitor.monitor_mcq(
                    question=question,
                    options=options,
                    topic=topic,
                    context=f"Difficulty: {quiz_params.get('difficulty', 'medium')}"
                )
            else:
                # Fallback if coherence monitor failed to load
                is_coherent, issues = True, []
            
            if not is_coherent:
                logger.error(f"[EMERGENCY] SPAGHETTI DETECTED! Issues: {[issue.description for issue in issues]}")
                
                # Check for critical issues that make the question unusable
                critical_issues = [issue for issue in issues if hasattr(issue, 'severity') and issue.severity == 'critical']
                if critical_issues:
                    logger.error("[EMERGENCY] CRITICAL SPAGHETTI - Question is unusable!")
                    return None
                else:
                    logger.warning("[WARNING] Minor coherence issues detected but question is usable")
            else:
                logger.info("[OK] MCQ passed coherence monitoring")
            
            # Create MCQResult object
            mcq_result = MCQResult(
                question=question,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation
            )
            
            return mcq_result
            
        except Exception as e:
            logger.error(f"[ERROR] MCQ validation failed: {e}")
            return None
    

    
    def _comprehensive_content_filter(self, topic: str) -> Dict[str, Any]:
        """
        [START] BUG FIX 32: Comprehensive content filtering that prevents bypass attacks

        This method performs thorough content filtering on the raw user input
        BEFORE any topic resolution occurs, preventing inappropriate topics
        from being laundered through the resolver's fallback mechanisms.

        Args:
            topic: Raw user input topic

        Returns:
            Dict with filtering results: {"is_safe": bool, "topic": str, "reason": str}
        """
        if not topic or not isinstance(topic, str):
            return {"is_safe": True, "topic": "general knowledge", "reason": "empty input"}

        original_topic = topic
        topic_lower = topic.lower().strip()

        # [START] BUG FIX 32: Enhanced inappropriate content detection
        # Check for direct inappropriate keywords
        for inappropriate in self._inappropriate_topics:
            if inappropriate in topic_lower:
                return {
                    "is_safe": False,
                    "topic": original_topic,
                    "reason": f"Contains inappropriate keyword: '{inappropriate}'"
                }

        # [START] BUG FIX 32: Check for bypass attempts with garbage text
        # Look for inappropriate content mixed with random characters
        suspicious_patterns = [
            r'bomb\s*\w*',  # bomb followed by any characters
            r'weapon\s*\w*',  # weapon followed by any characters
            r'kill\s*\w*',   # kill followed by any characters
            r'harm\s*\w*',   # harm followed by any characters
            r'illegal\s*\w*', # illegal followed by any characters
            r'drug\s*\w*',   # drug followed by any characters
            r'violence\s*\w*', # violence followed by any characters
            r'terror\s*\w*',  # terror followed by any characters
            r'hate\s*\w*',    # hate followed by any characters
            r'nsfw\s*\w*',    # nsfw followed by any characters
        ]

        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, topic_lower):
                return {
                    "is_safe": False,
                    "topic": original_topic,
                    "reason": f"Contains suspicious pattern that may be attempting to bypass content filters"
                }

        # [START] BUG FIX 32: Check for mixed inappropriate content with random text
        # This catches attempts like "how to build a bomb zxqwer"
        words = topic_lower.split()
        inappropriate_word_count = 0
        random_char_count = 0

        for word in words:
            # Check if word contains inappropriate content
            for inappropriate in self._inappropriate_topics:
                if inappropriate in word:
                    inappropriate_word_count += 1
                    break

            # Check if word looks like random characters (bypass attempt)
            if len(word) > 3 and not any(word.startswith(prefix) for prefix in ['the', 'and', 'how', 'what', 'why', 'when', 'where']):
                # Count consonant clusters as potential random text
                consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', word)
                if len(consonant_clusters) > 0:
                    random_char_count += 1

        # If we have inappropriate words mixed with random text, it's likely a bypass attempt
        if inappropriate_word_count > 0 and random_char_count > 0:
            return {
                "is_safe": False,
                "topic": original_topic,
                "reason": "Detected potential content filter bypass attempt (inappropriate content mixed with random text)"
            }

        # [START] BUG FIX 32: Check topic length for potential abuse
        if len(topic_lower) > 200:
            return {
                "is_safe": False,
                "topic": original_topic,
                "reason": "Topic too long (potential abuse)"
            }

        # Topic passed all filters
        return {
            "is_safe": True,
            "topic": original_topic,
            "reason": "passed content filtering"
        }

    def _filter_inappropriate_topic(self, topic: str) -> str:
        """Legacy method - kept for backward compatibility"""
        result = self._comprehensive_content_filter(topic)
        if not result["is_safe"]:
            logger.warning(f"🛡️ Inappropriate topic detected: {topic}")
            return "General Knowledge"
        return result["topic"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive Golden Path system status"""
        try:
            if not self._ensure_initialized():
                return {
                    "status": "error",
                    "message": "UnifiedInferenceManager not initialized",
                    "available_methods": []
                }
            
            # Get status from unified inference manager
            inference_status = get_inference_status()
            
            # Check Golden Path capabilities
            available_lora_adapters = self.get_available_lora_adapters()
            rag_available = self._rag_engine and self._rag_engine.is_initialized
            
            return {
                "status": "ready" if inference_status.get("state") == "ready" else "initializing",
                "unified_manager": inference_status,
                "golden_path": {
                    "lora_adapters_available": len(available_lora_adapters),
                    "lora_adapter_names": available_lora_adapters,
                    "rag_engine_ready": rag_available,
                    "generation_hierarchy": ["LoRA Adapter", "RAG Context", "Raw Model"]
                },
                "coherence_monitoring": "enabled",
                "content_filtering": "enabled",
                "error_handling": "active"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get system status: {e}")
            return {
                "status": "error",
                "message": str(e),
                "available_methods": []
            }

    # DeepSeek integration removed - using BatchTwoModelPipeline via unified_inference_manager

    # Legacy compatibility methods - these now use the unified system
    def set_offline_mode(self, offline: bool):
        """Set offline mode (legacy compatibility)"""
        # The unified system handles mode switching automatically
        logger.info(f"Mode preference noted: {'offline' if offline else 'online'}")
    
    def is_offline_available(self) -> bool:
        """Check if offline mode is available"""
        try:
            if not self._ensure_initialized():
                return False
            status = get_inference_status()
            return status.get("local_available", False)
        except:
            return False
    
    def is_online_available(self) -> bool:
        """Check if online mode is available"""
        try:
            if not self._ensure_initialized():
                return False
            status = get_inference_status()
            return status.get("cloud_available", False)
        except:
            return False


class MCQResult:
    """MCQ result container with Golden Path metadata"""

    def __init__(self, question: str, options: List[str], correct_answer: str, explanation: str = "No explanation available.", difficulty: str = "medium"):
        self.question = question
        self.options = options
        self.correct_answer = correct_answer
        self.explanation = explanation
        self.difficulty = difficulty  # Preserve difficulty level

        # Golden Path metadata
        self.generation_method = "raw_model"  # Which generation method was used
        self.grounded = False  # Whether specialized knowledge was used (LoRA or RAG)
        self.context_chunks = 0  # Number of context chunks retrieved (RAG only)
        self.adapter_used = None  # LoRA adapter name used (LoRA only)


def get_mcq_manager(config=None) -> MCQManager:
    """
    [HOT] FIRE: Get MCQ manager with unified inference system
    
    This is the main function for getting an MCQ manager instance.
    Now uses the UnifiedInferenceManager for all operations.
    """
    return MCQManager(config)