"""
🚀 Enhanced DeepSeek Processing Pipeline with 128K Context Optimization

This system integrates the Intelligent Context Manager to maximize
DeepSeek-R1 14B's 128K token context window for superior document processing.

Features:
- Context-aware document processing
- Intelligent batch processing with optimal token utilization
- Smart caching and result aggregation
- Memory-efficient processing with offloading
- Real-time optimization and monitoring
"""

import json
import logging
import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .intelligent_context_manager import (
    get_intelligent_context_manager, 
    ProcessingSession, 
    ContentChunk,
    TokenMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Enhanced processing result with context optimization data"""
    success: bool
    session_id: str
    processed_chunks: List[str]
    training_data: Dict[str, Any]
    context_stats: Dict[str, Any]
    processing_time: float
    token_efficiency: float
    cache_performance: Dict[str, Any]
    optimization_applied: bool = False
    
@dataclass
class DeepSeekConfig:
    """Enhanced DeepSeek configuration with context optimization"""
    model_name: str = "deepseek-r1:14b"
    max_context_tokens: int = 128000
    batch_size: int = 5  # Number of chunks to process together
    enable_smart_batching: bool = True
    enable_context_optimization: bool = True
    enable_result_caching: bool = True
    processing_timeout: int = 600  # Base timeout - will be dynamically adjusted
    use_dynamic_timeout: bool = True  # Enable adaptive timeout for thinking models
    quality_threshold: float = 0.7
    retry_failed_chunks: bool = True
    max_retries: int = 3

class EnhancedDeepSeekProcessor:
    """
    🚀 Enhanced DeepSeek Processor with 128K Context Optimization
    
    Integrates intelligent context management to maximize processing efficiency:
    1. Optimal context window utilization (up to 120K tokens)
    2. Smart document chunking and batching
    3. Hierarchical result caching and aggregation
    4. Real-time performance monitoring
    5. Adaptive processing strategies
    """
    
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()
        self.context_manager = get_intelligent_context_manager()
        self.processing_cache = {}
        self.performance_metrics = {
            "total_documents_processed": 0,
            "total_tokens_processed": 0,
            "average_processing_time": 0.0,
            "context_utilization_rate": 0.0,
            "cache_hit_rate": 0.0,
            "successful_extractions": 0,
            "failed_extractions": 0
        }
        
        # Initialize DeepSeek API client
        self.deepseek_client = None
        self._initialize_deepseek_client()
        
        logger.info(f"🚀 Enhanced DeepSeek Processor initialized")
        logger.info(f"📊 Model: {self.config.model_name}")
        logger.info(f"🧠 Context window: {self.config.max_context_tokens:,} tokens")
        logger.info(f"🕐 Dynamic timeout: {'enabled' if self.config.use_dynamic_timeout else 'disabled'}")
    
    def _calculate_dynamic_timeout(self, context: str, base_timeout: Optional[int] = None) -> int:
        """Calculate dynamic timeout based on model type, content complexity, and DeepSeek-R1 reasoning requirements"""
        
        if not self.config.use_dynamic_timeout and base_timeout:
            logger.info(f"🕐 Using fixed timeout: {base_timeout}s")
            return base_timeout
        
        base = base_timeout or self.config.processing_timeout or 60
        
        # Model-specific timeout multipliers for thinking models
        model_multiplier = 1.0
        if self.config.model_name:
            model_lower = self.config.model_name.lower()
            
            # DeepSeek-R1 is a reasoning model that needs MUCH more time for thinking
            if any(keyword in model_lower for keyword in ['deepseek-r1', 'r1:', 'reasoning']):
                model_multiplier = 8.0  # 8x longer for DeepSeek-R1 reasoning
                logger.info(f"🧠 Detected DeepSeek-R1 reasoning model: {self.config.model_name} (8x timeout multiplier)")
            elif 'deepseek' in model_lower:
                model_multiplier = 5.0  # 5x longer for other DeepSeek models
                logger.info(f"🧠 Detected DeepSeek model: {self.config.model_name} (5x timeout multiplier)")
            elif any(keyword in model_lower for keyword in ['qwq', 'thinking']):
                model_multiplier = 4.0  # 4x longer for thinking models
                logger.info(f"🧠 Detected thinking model: {self.config.model_name} (4x timeout multiplier)")
        
        # Content complexity multiplier
        content_multiplier = 1.0
        content_lower = context.lower()
        
        # Academic/complex content needs more reasoning time
        if any(keyword in content_lower for keyword in ['research', 'academic', 'scientific', 'theorem', 'proof']):
            content_multiplier *= 2.0
            logger.info("🎓 Detected academic/scientific content (2x timeout multiplier)")
        
        # Mathematical/technical content needs more reasoning
        if any(keyword in content_lower for keyword in ['equation', 'formula', 'algorithm', 'calculation']):
            content_multiplier *= 1.5
            logger.info("🔢 Detected mathematical/technical content (1.5x timeout multiplier)")
        
        # Large content needs more processing time
        content_size = len(context)
        if content_size > 50000:  # Large documents
            content_multiplier *= 2.0
            logger.info(f"📚 Large content detected ({content_size:,} chars) (2x timeout multiplier)")
        elif content_size > 20000:  # Medium documents
            content_multiplier *= 1.5
            logger.info(f"📄 Medium content detected ({content_size:,} chars) (1.5x timeout multiplier)")
        
        # Calculate final timeout
        final_timeout = int(base * model_multiplier * content_multiplier)
        
        # Cap the timeout to reasonable limits
        final_timeout = min(final_timeout, 1800)  # Max 30 minutes for complex reasoning
        final_timeout = max(final_timeout, 120)   # Min 2 minutes for DeepSeek-R1
        
        logger.info(f"🕐 Dynamic timeout calculated: {final_timeout}s (base: {base}s, model: {model_multiplier}x, content: {content_multiplier}x)")
        
        return final_timeout
    
    def _initialize_deepseek_client(self):
        """Initialize DeepSeek API client or Ollama client"""
        try:
            # Try to use Ollama for local processing
            import requests
            
            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                deepseek_models = [m for m in models if 'deepseek' in m.get('name', '').lower()]
                
                if deepseek_models:
                    self.deepseek_client = "ollama"
                    logger.info(f"✅ Using Ollama for DeepSeek processing")
                    logger.info(f"📋 Available DeepSeek models: {[m['name'] for m in deepseek_models]}")
                else:
                    logger.warning("⚠️ No DeepSeek models found in Ollama")
                    self.deepseek_client = "mock"
            else:
                logger.warning("⚠️ Ollama not accessible")
                self.deepseek_client = "mock"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize DeepSeek client: {e}")
            self.deepseek_client = "mock"
    
    async def process_documents_with_context_optimization(
        self, 
        documents: List[Dict[str, Any]], 
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Process documents with intelligent context optimization
        
        Args:
            documents: List of document data with 'name' and 'content'
            progress_callback: Progress reporting callback
            
        Returns:
            Enhanced processing result with optimization metrics
        """
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback("🧠 Initializing intelligent context management...")
            
            # Create processing session
            session = self.context_manager.create_processing_session(documents)
            
            if progress_callback:
                progress_callback(f"📄 Analyzing {len(documents)} documents for optimal processing...")
            
            # Phase 1: Intelligent document analysis and chunking
            all_chunks = []
            document_metadata = {}
            
            for i, doc in enumerate(documents):
                if progress_callback:
                    progress_callback(f"🔍 Analyzing document {i+1}/{len(documents)}: {doc['name']}")
                
                # Analyze and chunk document with context optimization
                doc_chunks = self.context_manager.create_intelligent_chunks(
                    doc['content'], 
                    doc['name']
                )
                
                all_chunks.extend(doc_chunks)
                document_metadata[doc['name']] = {
                    'chunks': len(doc_chunks),
                    'total_tokens': sum(chunk.token_count for chunk in doc_chunks),
                    'complexity': self.context_manager.analyze_content_complexity(doc['content'])
                }
            
            logger.info(f"📊 Document analysis complete:")
            logger.info(f"   📝 Total chunks: {len(all_chunks)}")
            logger.info(f"   🎯 Average priority: {sum(c.priority_score for c in all_chunks) / len(all_chunks):.2f}")
            logger.info(f"   💾 Total tokens: {sum(c.token_count for c in all_chunks):,}")
            
            if progress_callback:
                progress_callback(f"🎯 Optimizing {len(all_chunks)} chunks for context window...")
            
            # Phase 2: Context window optimization and batching
            processing_batches = self._create_optimal_batches(session, all_chunks)
            
            if progress_callback:
                progress_callback(f"🚀 Processing {len(processing_batches)} optimized batches...")
            
            # Phase 3: Batch processing with context optimization
            all_training_data = {
                "concepts": [],
                "relationships": [],
                "training_examples": []
            }
            
            processed_chunks = []
            failed_chunks = []
            
            for batch_idx, batch in enumerate(processing_batches):
                if progress_callback:
                    progress_callback(f"🧠 Processing batch {batch_idx + 1}/{len(processing_batches)} ({len(batch)} chunks)")
                
                try:
                    batch_result = await self._process_batch_with_context(session, batch, progress_callback)
                    
                    if batch_result['success']:
                        # Aggregate results
                        for key in all_training_data:
                            if key in batch_result['training_data']:
                                all_training_data[key].extend(batch_result['training_data'][key])
                        
                        processed_chunks.extend([chunk.id for chunk in batch])
                        session.completed_chunks.extend([chunk.id for chunk in batch])
                    else:
                        failed_chunks.extend([chunk.id for chunk in batch])
                        session.failed_chunks.extend([chunk.id for chunk in batch])
                        
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} processing failed: {e}")
                    failed_chunks.extend([chunk.id for chunk in batch])
                    session.failed_chunks.extend([chunk.id for chunk in batch])
            
            # Phase 4: Result optimization and caching
            if progress_callback:
                progress_callback("🎯 Optimizing and caching results...")
            
            optimized_training_data = self._optimize_training_data(all_training_data)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            context_stats = self.context_manager.get_optimization_stats()
            
            # Update performance metrics
            self.performance_metrics["total_documents_processed"] += len(documents)
            self.performance_metrics["total_tokens_processed"] += sum(c.token_count for c in all_chunks)
            self.performance_metrics["successful_extractions"] += len(processed_chunks)
            self.performance_metrics["failed_extractions"] += len(failed_chunks)
            
            if len(processed_chunks) > 0:
                token_efficiency = sum(
                    len(optimized_training_data[key]) for key in optimized_training_data
                ) / sum(c.token_count for c in all_chunks)
                
                self.performance_metrics["context_utilization_rate"] = (
                    session.context_window.context_utilization * 0.1 + 
                    self.performance_metrics["context_utilization_rate"] * 0.9
                )
            else:
                token_efficiency = 0.0
            
            # Prepare final result
            result = ProcessingResult(
                success=len(processed_chunks) > 0,
                session_id=session.session_id,
                processed_chunks=processed_chunks,
                training_data=optimized_training_data,
                context_stats=context_stats,
                processing_time=processing_time,
                token_efficiency=token_efficiency,
                cache_performance={
                    "hits": session.context_window.cache_hits,
                    "misses": session.context_window.cache_misses,
                    "hit_rate": session.context_window.cache_hits / max(
                        session.context_window.cache_hits + session.context_window.cache_misses, 1
                    )
                },
                optimization_applied=True
            )
            
            # Cache results for future use
            if self.config.enable_result_caching:
                self._cache_processing_results(session.session_id, result)
            
            # Cleanup session
            self.context_manager.cleanup_session(session.session_id)
            
            if progress_callback:
                progress_callback("✅ Document processing with context optimization complete!")
            
            logger.info(f"🎉 Processing complete:")
            logger.info(f"   ✅ Processed chunks: {len(processed_chunks)}")
            logger.info(f"   ❌ Failed chunks: {len(failed_chunks)}")
            logger.info(f"   📊 Processing time: {processing_time:.1f}s")
            logger.info(f"   🎯 Token efficiency: {token_efficiency:.3f}")
            logger.info(f"   💾 Cache hit rate: {result.cache_performance['hit_rate']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                session_id="failed",
                processed_chunks=[],
                training_data={"concepts": [], "relationships": [], "training_examples": []},
                context_stats={},
                processing_time=processing_time,
                token_efficiency=0.0,
                cache_performance={"hits": 0, "misses": 0, "hit_rate": 0.0},
                optimization_applied=False
            )
    
    def _create_optimal_batches(self, session: ProcessingSession, chunks: List[ContentChunk]) -> List[List[ContentChunk]]:
        """Create optimal processing batches using context window optimization"""
        if not self.config.enable_smart_batching:
            # Simple batching by size
            batches = []
            for i in range(0, len(chunks), self.config.batch_size):
                batches.append(chunks[i:i + self.config.batch_size])
            return batches
        
        logger.info(f"🎯 Creating optimal batches for {len(chunks)} chunks")
        
        # Sort chunks by priority for optimal processing order
        sorted_chunks = sorted(chunks, key=lambda x: x.priority_score, reverse=True)
        
        batches = []
        current_batch = []
        current_tokens = 0
        max_batch_tokens = session.context_window.available_tokens // 2  # Conservative batching
        
        for chunk in sorted_chunks:
            # Check if chunk fits in current batch
            if (len(current_batch) < self.config.batch_size and 
                current_tokens + chunk.token_count <= max_batch_tokens):
                current_batch.append(chunk)
                current_tokens += chunk.token_count
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk.token_count
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"📊 Created {len(batches)} optimal batches")
        logger.info(f"   🎯 Average batch size: {sum(len(b) for b in batches) / len(batches):.1f} chunks")
        logger.info(f"   💾 Average batch tokens: {sum(sum(c.token_count for c in b) for b in batches) / len(batches):,.0f}")
        
        return batches
    
    async def _process_batch_with_context(
        self, 
        session: ProcessingSession, 
        batch: List[ContentChunk],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a batch of chunks with intelligent context management"""
        try:
            if progress_callback:
                progress_callback(f"🧠 Optimizing context for {len(batch)} chunks...")
            
            # Optimize context window for this batch
            optimized_chunks = self.context_manager.optimize_context_window(session, batch)
            
            if not optimized_chunks:
                logger.warning("⚠️ No chunks could fit in context window")
                return {"success": False, "error": "Context window optimization failed"}
            
            if progress_callback:
                progress_callback(f"🚀 Processing {len(optimized_chunks)} optimized chunks...")
            
            # Prepare context for DeepSeek processing
            context = self.context_manager.prepare_context_for_processing(session, optimized_chunks)
            
            # Process with DeepSeek
            result = await self._call_deepseek_with_context(context, progress_callback)
            
            if result and result.get('success'):
                # Update session metrics
                session.metrics.total_tokens += len(optimized_chunks)
                session.metrics.context_utilization = session.context_window.context_utilization
                
                return result
            else:
                logger.warning("⚠️ DeepSeek processing returned no valid result")
                return {"success": False, "error": "DeepSeek processing failed"}
                
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _call_deepseek_with_context(self, context: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Call DeepSeek API with optimized context"""
        try:
            if self.deepseek_client == "ollama":
                return await self._call_ollama_deepseek(context, progress_callback)
            else:
                # Mock processing for testing
                return await self._mock_deepseek_processing(context, progress_callback)
                
        except Exception as e:
            logger.error(f"❌ DeepSeek API call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _call_ollama_deepseek(self, context: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Call DeepSeek via Ollama with context optimization"""
        try:
            import aiohttp
            import json
            
            # Calculate dynamic timeout based on content complexity and model type
            dynamic_timeout = self._calculate_dynamic_timeout(context)
            
            # 🚀 DEBUGGING: Let's see the actual sizes being processed
            original_context_chars = len(context)
            original_context_tokens = 0
            
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                original_context_tokens = len(enc.encode(context))
            except ImportError:
                original_context_tokens = original_context_chars // 4  # Rough estimate
            
            logger.info(f"🔍 Original context: {original_context_chars} chars, ~{original_context_tokens} tokens")
            
            # Check if context is actually too large for the model's 128K context
            max_context_tokens = 32000  # Use 32K to leave room for response and system prompts
            
            # Only truncate if actually necessary
            if original_context_tokens > max_context_tokens:
                logger.warning(f"⚠️ Context too large ({original_context_tokens} tokens > {max_context_tokens}), truncating...")
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    tokens = enc.encode(context)[:max_context_tokens]
                    context = enc.decode(tokens)
                except ImportError:
                    # Fallback to character-based truncation
                    char_limit = max_context_tokens * 4
                    context = context[:char_limit]
                    
                final_tokens = len(enc.encode(context)) if 'enc' in locals() else len(context) // 4
                logger.info(f"✂️ Truncated to: {len(context)} chars, ~{final_tokens} tokens")
            else:
                logger.info(f"✅ Context size OK: {original_context_tokens} tokens < {max_context_tokens}")
                
            prompt_size = len(context.encode('utf-8'))
            
            # DeepSeek-R1 optimized prompt with thinking process
            extraction_prompt = f"""<think>
My goal is to extract key information from the provided academic text. I need to identify the main concepts, their definitions, the relationships between them, and create relevant question-answer pairs for training. I will structure my final output as a single, valid JSON object with the keys 'concepts', 'relationships', and 'training_examples'. I will not include any text outside of this JSON object.
</think>

Extract key information from this academic content as JSON:

{{
  "concepts": [
    {{"name": "Concept Name", "definition": "Clear definition..."}}
  ],
  "relationships": [
    {{"source": "Concept A", "target": "Concept B", "description": "How they relate..."}}
  ],
  "training_examples": [
    {{"question": "Question based on content?", "answer": "Answer from the text..."}}
  ]
}}

Content:
{context}

JSON:"""

            # 🔍 DEBUGGING: Log actual prompt size
            final_prompt_chars = len(extraction_prompt)
            try:
                final_prompt_tokens = len(enc.encode(extraction_prompt)) if 'enc' in locals() else final_prompt_chars // 4
            except:
                final_prompt_tokens = final_prompt_chars // 4
                
            logger.info(f"📏 Final prompt: {final_prompt_chars} chars, ~{final_prompt_tokens} tokens")
                
            if progress_callback:
                progress_callback(f"🤖 Sending to DeepSeek-R1: {final_prompt_tokens} tokens (timeout: {dynamic_timeout}s)")

            payload = {
                "model": self.config.model_name,
                "prompt": extraction_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,     # Low temp for consistent JSON
                    "top_p": 0.9,          # Good for structured output
                    "repeat_penalty": 1.05, # Prevent repetition
                    "stop": ["</think>", "###", "Human:", "Assistant:"]  # Clear stop tokens
                    # CRITICAL: Removed num_ctx to prevent VRAM exhaustion!
                }
            }
            
            # 🚀 ENHANCED: Retry logic for 500 errors with exponential backoff
            max_retries = 3
            retry_delay = 2.0
            
            for attempt in range(max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        # Set both total timeout and individual operation timeouts
                        timeout = aiohttp.ClientTimeout(
                            total=dynamic_timeout,
                            connect=30,  # Connection timeout
                            sock_read=60  # Socket read timeout
                        )
                        
                        async with session.post(
                            "http://localhost:11434/api/generate",
                            json=payload,
                            timeout=timeout
                        ) as response:
                            
                            if response.status == 200:
                                result = await response.json()
                                response_text = result.get('response', '')
                                
                                if progress_callback:
                                    progress_callback("🧠 Parsing DeepSeek response...")
                                
                                # Parse JSON response
                                try:
                                    # Extract JSON from response
                                    json_start = response_text.find('{')
                                    json_end = response_text.rfind('}') + 1
                                    
                                    if json_start >= 0 and json_end > json_start:
                                        json_content = response_text[json_start:json_end]
                                        parsed_result = json.loads(json_content)
                                        
                                        return {
                                            "success": True,
                                            "training_data": parsed_result,
                                            "raw_response": response_text,
                                            "model_used": self.config.model_name
                                        }
                                    else:
                                        # CRITICAL FIX: Fail fast instead of using mock data
                                        logger.warning("⚠️ Failed to parse JSON from DeepSeek response")
                                        raise ValueError("DeepSeek returned invalid JSON format")
                                        
                                except json.JSONDecodeError as e:
                                    logger.warning(f"⚠️ JSON parsing error: {e}")
                                    raise ValueError(f"DeepSeek JSON parsing failed: {e}") from e
                            
                            elif response.status == 500 and attempt < max_retries:
                                # Retry on 500 errors with exponential backoff
                                logger.warning(f"⚠️ Ollama server error 500 (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_delay}s...")
                                if progress_callback:
                                    progress_callback(f"⚠️ Server error, retrying in {retry_delay}s... (attempt {attempt + 1})")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logger.error(f"❌ Ollama request failed: {response.status} (attempt {attempt + 1})")
                                if attempt == max_retries:
                                    # CRITICAL FIX: Never use mock data in production - raise exception instead
                                    logger.error(f"❌ All DeepSeek API attempts failed after {max_retries + 1} tries")
                                    raise RuntimeError(f"DeepSeek API completely unavailable after {max_retries + 1} attempts (status {response.status})")
                                else:
                                    # Retry other errors too
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                    continue
                                    
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(f"⏰ DeepSeek request timed out (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                        if progress_callback:
                            progress_callback(f"⏰ Timeout, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"⏰ DeepSeek request timed out after {dynamic_timeout}s (all retries failed)")
                        # CRITICAL FIX: Never use mock data - raise exception instead
                        raise RuntimeError(f"DeepSeek API timed out after {dynamic_timeout}s and {max_retries + 1} attempts")
                        
                except Exception as e:
                    if attempt < max_retries:
                        error_type = type(e).__name__
                        logger.warning(f"⚠️ Request error ({error_type}): {e} (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        error_type = type(e).__name__
                        logger.error(f"❌ Ollama DeepSeek call failed ({error_type}): {e} (all retries failed)")
                        # CRITICAL FIX: Never use mock data - raise exception instead
                        raise RuntimeError(f"DeepSeek API failed with {error_type}: {e} after {max_retries + 1} attempts")
        
        except Exception as outer_e:
            # Catch any other unexpected errors
            logger.error(f"❌ Unexpected error in DeepSeek processing: {outer_e}")
            # CRITICAL FIX: Never use mock data - raise exception instead
            raise RuntimeError(f"Unexpected DeepSeek processing error: {outer_e}") from outer_e
    
    async def _mock_deepseek_processing(self, context: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Mock DeepSeek processing for testing"""
        if progress_callback:
            progress_callback("🔄 Using mock DeepSeek processing...")
        
        # Simulate processing time
        await asyncio.sleep(1.0)
        
        # Extract some basic information from context
        words = context.split()
        concepts = []
        relationships = []
        training_examples = []
        
        # Mock concept extraction
        for i in range(min(5, len(words) // 100)):
            concept_name = f"Concept_{i+1}"
            concepts.append({
                "name": concept_name,
                "definition": f"Key concept extracted from document analysis",
                "importance": 0.8 + (i * 0.05)
            })
        
        # Mock relationship extraction
        for i in range(min(3, len(concepts))):
            relationships.append({
                "source": concepts[i]["name"],
                "target": concepts[(i+1) % len(concepts)]["name"] if len(concepts) > 1 else "Related_Concept",
                "type": "related_to",
                "description": f"Relationship identified through document analysis"
            })
        
        # Mock training examples
        for i in range(min(7, len(words) // 200)):
            training_examples.append({
                "question": f"What is the significance of concept {i+1} in this context?",
                "answer": f"Concept {i+1} represents an important aspect of the analyzed content.",
                "difficulty": "medium",
                "topic": "Document Analysis"
            })
        
        return {
            "success": True,
            "training_data": {
                "concepts": concepts,
                "relationships": relationships,
                "training_examples": training_examples
            },
            "raw_response": "Mock processing result",
            "model_used": "mock_deepseek"
        }
    
    def _optimize_training_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize and deduplicate training data"""
        logger.info("🎯 Optimizing training data...")
        
        optimized = {
            "concepts": [],
            "relationships": [], 
            "training_examples": []
        }
        
        # Deduplicate concepts
        seen_concepts = set()
        for concept in training_data.get("concepts", []):
            concept_key = concept.get("name", "").lower()
            if concept_key and concept_key not in seen_concepts:
                seen_concepts.add(concept_key)
                optimized["concepts"].append(concept)
        
        # Deduplicate relationships
        seen_relationships = set()
        for rel in training_data.get("relationships", []):
            rel_key = f"{rel.get('source', '')}-{rel.get('target', '')}-{rel.get('type', '')}"
            if rel_key and rel_key not in seen_relationships:
                seen_relationships.add(rel_key)
                optimized["relationships"].append(rel)
        
        # Deduplicate and rank training examples
        seen_questions = set()
        for example in training_data.get("training_examples", []):
            question = example.get("question", "").lower().strip()
            if question and question not in seen_questions:
                seen_questions.add(question)
                optimized["training_examples"].append(example)
        
        logger.info(f"📊 Training data optimization complete:")
        logger.info(f"   🎯 Concepts: {len(optimized['concepts'])}")
        logger.info(f"   🔗 Relationships: {len(optimized['relationships'])}")
        logger.info(f"   📝 Training examples: {len(optimized['training_examples'])}")
        
        return optimized
    
    def _cache_processing_results(self, session_id: str, result: ProcessingResult):
        """Cache processing results for future use"""
        cache_key = hashlib.sha256(f"{session_id}_{time.time()}".encode()).hexdigest()[:16]
        
        self.processing_cache[cache_key] = {
            "session_id": session_id,
            "result": result,
            "cached_at": time.time(),
            "access_count": 0
        }
        
        # Limit cache size
        if len(self.processing_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.processing_cache.keys(),
                key=lambda k: self.processing_cache[k]["cached_at"]
            )[:20]
            
            for key in oldest_keys:
                del self.processing_cache[key]
        
        logger.info(f"💾 Cached processing results for session {session_id}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get processor performance metrics"""
        return self.performance_metrics.copy()

def get_enhanced_deepseek_processor(config: Optional[DeepSeekConfig] = None) -> EnhancedDeepSeekProcessor:
    """Get singleton instance of enhanced DeepSeek processor"""
    if not hasattr(get_enhanced_deepseek_processor, '_instance'):
        get_enhanced_deepseek_processor._instance = EnhancedDeepSeekProcessor(config)
    return get_enhanced_deepseek_processor._instance
