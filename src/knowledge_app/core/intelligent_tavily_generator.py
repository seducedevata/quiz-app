"""
Intelligent Quiz Generator using Tavily Web Search + Local Llama 3.1 8B

This module provides intelligent quiz generation when only Tavily is enabled:
1. Uses Tavily web search to gather comprehensive information about any topic
2. Uses local Llama 3.1 8B to curate high-quality quiz questions from search results
3. Implements fallback mechanisms for robust operation
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

from .tavily_integration import TavilyIntegration, get_tavily_integration
from .unified_inference_manager import UnifiedInferenceManager, get_unified_inference_manager
from .offline_mcq_generator import OfflineMCQGenerator

logger = logging.getLogger(__name__)

@dataclass
class IntelligentQuizRequest:
    """Request for intelligent quiz generation"""
    topic: str
    difficulty: str = "medium"
    question_type: str = "mixed"
    num_questions: int = 1
    use_tavily_only: bool = True

class IntelligentTavilyGenerator:
    """
    Intelligent quiz generator using Tavily + local Llama 3.1 8B
    
    Features:
    - Automatic Tavily web search for topic information
    - Local Llama 3.1 8B for intelligent question curation
    - Smart topic analysis and question generation
    - Fallback mechanisms for reliability
    """
    
    def __init__(self):
        self.tavily = None
        self.inference_manager = None
        self.offline_generator = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the intelligent generator"""
        try:
            # Initialize Tavily integration
            self.tavily = get_tavily_integration()
            if not self.tavily:
                logger.warning("Tavily integration not available")
                return False
                
            # Initialize unified inference manager
            self.inference_manager = get_unified_inference_manager()
            if not self.inference_manager:
                logger.warning("Unified inference manager not available")
                return False
                
            # Initialize offline generator as fallback
            self.offline_generator = OfflineMCQGenerator()
            
            self._initialized = True
            logger.info("✅ Intelligent Tavily Generator initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Intelligent Tavily Generator: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if intelligent generator is available"""
        return self._initialized and self.tavily and self.inference_manager
    
    async def generate_intelligent_quiz(self, request: IntelligentQuizRequest) -> Dict[str, Any]:
        """
        Generate intelligent quiz using Tavily + local Llama 3.1 8B - NO FALLBACK ALLOWED
        
        Args:
            request: IntelligentQuizRequest with generation parameters
            
        Returns:
            Dict containing generated quiz questions or error
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"🌐 Starting intelligent quiz generation: {request.topic} ({request.difficulty})")
            
            # Step 1: Search Tavily for topic information
            search_results = await self._search_tavily_topic(request.topic)
            if not search_results:
                error_msg = "Tavily search failed - no results found"
                logger.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Step 2: Process search results into context
            context = self._process_search_results(search_results)
            if not context:
                error_msg = "Failed to process Tavily search results"
                logger.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Step 3: Generate intelligent questions using local model
            questions = await self._generate_intelligent_questions(request, context)
            if not questions:
                error_msg = "Failed to generate questions from Tavily context"
                logger.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
            
            logger.info(f"✅ Generated {len(questions)} intelligent questions")
            return {
                "success": True,
                "questions": questions,
                "source": "tavily_intelligent",
                "context_used": len(context),
                "search_results": len(search_results.get('results', []))
            }
            
        except Exception as e:
            logger.error(f"❌ Intelligent quiz generation failed: {e}")
            return {"success": False, "error": f"Tavily generation failed: {str(e)}"}
    
    async def _search_tavily_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Search Tavily for comprehensive topic information"""
        try:
            # Use Tavily to search for educational content
            search_query = f"{topic} educational content quiz questions facts"
            
            # Perform search with educational focus
            search_result = self.tavily.search_for_topic(
                topic=search_query,
                max_results=10
            )
            
            if search_result and search_result.results:
                logger.info(f"🔍 Found {len(search_result.results)} Tavily results")
                return {
                    'results': search_result.results,
                    'query': search_result.query,
                    'success': search_result.success
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Tavily search failed: {e}")
            return None
    
    def _process_search_results(self, search_results: Dict[str, Any]) -> str:
        """Process Tavily search results into usable context"""
        try:
            results = search_results.get('results', [])
            context_parts = []
            
            for result in results[:5]:  # Use top 5 results
                title = result.get('title', '')
                content = result.get('content', '')
                raw_content = result.get('raw_content', '')
                
                # Prioritize raw content if available
                usable_content = raw_content if raw_content else content
                if usable_content and len(usable_content) > 50:
                    context_parts.append(f"{title}: {usable_content}")
            
            # Combine context
            full_context = "\n\n".join(context_parts)
            
            # Truncate if too long (keep under 4000 chars for model)
            if len(full_context) > 4000:
                full_context = full_context[:4000] + "..."
            
            logger.info(f"📄 Processed {len(context_parts)} search snippets into context")
            return full_context
            
        except Exception as e:
            logger.error(f"❌ Failed to process search results: {e}")
            return f"Basic information about {search_results.get('query', 'the topic')}"
    
    async def _generate_intelligent_questions(self, request: IntelligentQuizRequest, context: str) -> List[Dict[str, Any]]:
        """Generate intelligent questions using local Llama 3.1 8B"""
        try:
            # Create intelligent prompt based on search context
            prompt = self._create_intelligent_prompt(request, context)
            
            # Use unified inference manager for generation
            inference_manager = get_unified_inference_manager()
            
            # Generate questions using local model
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: inference_manager.generate_mcq_sync(
                    topic=request.topic,
                    difficulty=request.difficulty,
                    question_type=request.question_type,
                    context=context,
                    generation_instructions=prompt,
                    timeout=120  # Allow more time for intelligent generation
                )
            )
            
            if result and isinstance(result, dict):
                questions = [result] if 'question' in result else []
                if 'questions' in result:
                    questions = result['questions']
                
                return questions
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Intelligent question generation failed: {e}")
            return []
    
    def _create_intelligent_prompt(self, request: IntelligentQuizRequest, context: str) -> str:
        """Create intelligent prompt for Llama 3.1 8B based on Tavily context"""
        
        base_prompt = f"""
Based on the following research about '{request.topic}', create high-quality quiz questions:

RESEARCH CONTEXT:
{context}

REQUIREMENTS:
- Difficulty: {request.difficulty}
- Type: {request.question_type}
- Questions: {request.num_questions}
- Use real facts from the research context
- Ensure questions are educational and accurate
- Include detailed explanations based on the research
- Make questions challenging but fair

FORMAT:
Create well-structured multiple choice questions with:
- Clear, accurate questions
- 4 plausible options (1 correct, 3 incorrect)
- Detailed explanations citing the research
- Appropriate difficulty level
"""
        
        # Add difficulty-specific instructions
        difficulty_instructions = {
            "easy": "Focus on basic concepts and straightforward facts from the research.",
            "medium": "Balance factual recall with some analytical thinking based on the research.",
            "hard": "Require deeper understanding and analysis of the research findings.",
            "expert": "Challenge with complex synthesis and critical thinking about the research."
        }
        
        if request.difficulty in difficulty_instructions:
            base_prompt += f"\n\nDIFFICULTY GUIDANCE: {difficulty_instructions[request.difficulty]}"
        
        return base_prompt
    
    async def _fallback_generation(self, request: IntelligentQuizRequest) -> Dict[str, Any]:
        """REMOVED: No fallback allowed - user explicitly selected online mode"""
        # This method is deprecated - no fallback mechanisms allowed
        return {
            "success": False,
            "error": "Tavily generation failed - no fallback available in online mode"
        }

# Global instance
_intelligent_generator = None

def get_intelligent_tavily_generator() -> IntelligentTavilyGenerator:
    """Get the global intelligent Tavily generator instance"""
    global _intelligent_generator
    if _intelligent_generator is None:
        _intelligent_generator = IntelligentTavilyGenerator()
    return _intelligent_generator

def initialize_intelligent_tavily_generator() -> bool:
    """Initialize the intelligent Tavily generator"""
    generator = get_intelligent_tavily_generator()
    return generator.initialize()

def is_intelligent_tavily_available() -> bool:
    """Check if intelligent Tavily generation is available"""
    generator = get_intelligent_tavily_generator()
    return generator.is_available()

async def generate_intelligent_tavily_quiz(topic: str, difficulty: str = "medium", 
                                         question_type: str = "mixed", 
                                         num_questions: int = 1) -> Dict[str, Any]:
    """
    High-level function to generate intelligent quiz using Tavily + local Llama 3.1 8B
    
    Args:
        topic: Quiz topic
        difficulty: Question difficulty (easy/medium/hard/expert)
        question_type: Type of questions (mixed/conceptual/numerical)
        num_questions: Number of questions to generate
        
    Returns:
        Dict with generated questions or error information
    """
    generator = get_intelligent_tavily_generator()
    
    if not generator.is_available():
        logger.warning("Intelligent generator not available, using basic generation")
        # Fallback to basic generation
        inference_manager = get_unified_inference_manager()
        if inference_manager:
            result = inference_manager.generate_mcq_sync(
                topic=topic,
                difficulty=difficulty,
                question_type=question_type
            )
            return {
                "success": bool(result),
                "questions": [result] if result else [],
                "source": "basic_generation"
            }
    
    request = IntelligentQuizRequest(
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        num_questions=num_questions
    )
    
    return await generator.generate_intelligent_quiz(request)
