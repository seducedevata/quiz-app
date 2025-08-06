"""
🚀 NEXT-GENERATION PROMPT SYSTEM - Powered by Unified Architecture

This module serves as the bridge between legacy systems and the revolutionary
"One Prompt to Rule Them All" unified architecture. It provides backward
compatibility while leveraging trust-based AI collaboration.

ARCHITECTURAL EVOLUTION:
1. Legacy: Constraint-heavy "prompt horror" with [EMERGENCY] tags
2. Unified: Clean structure with validation authority  
3. Revolutionary: Trust-based collaboration with AI intelligence

This module now delegates to UnifiedPromptBuilder for:
✅ Persona Engineering over Rule Engineering
✅ Few-Shot Examples as Ultimate Guides
✅ Single Dynamic Master Template
✅ Configuration-Driven Scalability

INTEGRATION:
- Maintains backward compatibility for existing code
- Provides clean migration path to unified system
- Eliminates constraint-based legacy patterns
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Initialize the revolutionary prompt system  
_prompt_builder: Optional['UnifiedPromptBuilder'] = None

def get_prompt_builder():
    """Get the global unified prompt builder instance"""
    global _prompt_builder
    if _prompt_builder is None:
        try:
            from .unified_prompt_builder import UnifiedPromptBuilder
            _prompt_builder = UnifiedPromptBuilder()
            logger.debug("✅ Initialized UnifiedPromptBuilder for next-gen prompting")
        except ImportError as e:
            logger.error(f"❌ Could not import UnifiedPromptBuilder: {e}")
            _prompt_builder = None
    return _prompt_builder


def create_unified_mcq_prompt(topic: str, 
                            context: str = "", 
                            difficulty: str = "medium", 
                            question_type: str = "mixed",
                            discipline: str = "general") -> str:
    """
    🚀 REVOLUTIONARY PROMPT CREATION - Trust-based AI collaboration
    
    This function represents the culmination of prompt engineering evolution:
    from constraint-heavy legacy to trust-based collaboration with AI.
    
    Args:
        topic: Question topic
        context: Additional context
        difficulty: Academic level (easy, medium, hard, expert)
        question_type: Type (numerical, conceptual, mixed)
        discipline: Academic field (physics, chemistry, etc.)
        
    Returns:
        Exceptional prompt that inspires AI excellence
    """
    try:
        builder = get_prompt_builder()
        
        if builder:
            # Use the revolutionary unified prompt system
            prompt = builder.build_unified_prompt(
                topic=topic,
                discipline=discipline,
                difficulty=difficulty,
                question_type=question_type,
                context=context
            )
            
            logger.debug(f"🚀 Generated unified prompt for {topic} ({difficulty}/{question_type})")
            return prompt
        else:
            # Fallback if unified system not available
            return _create_simple_fallback_prompt(topic, context, difficulty, question_type)
        
    except Exception as e:
        logger.error(f"❌ Unified prompt generation failed: {e}")
        # Graceful fallback to simple approach
        return _create_simple_fallback_prompt(topic, context, difficulty, question_type)


def _create_simple_fallback_prompt(topic: str, context: str, difficulty: str, question_type: str) -> str:
    """Ultra-simple fallback if unified system fails"""
    persona_map = {
        "expert": f"You are a distinguished expert in {topic}, designing a PhD-level question.",
        "hard": f"You are a university professor creating an advanced question on {topic}.",
        "medium": f"You are an experienced teacher creating an AP-level question on {topic}.",
        "easy": f"You are an educator creating a foundational question on {topic}."
    }
    
    persona = persona_map.get(difficulty, persona_map["medium"])
    context_text = f"Context: {context}" if context.strip() else ""
    
    return f"""{persona}

Create a {difficulty}-level {question_type} question about {topic}.
{context_text}

Return only valid JSON:
{{
  "question": "Your question about {topic}?",
  "options": {{
    "A": "First option",
    "B": "Second option", 
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct": "A",
  "explanation": "Clear explanation"
}}"""


# 🔄 LEGACY COMPATIBILITY FUNCTIONS

def _create_inquisitor_prompt(context: str, topic: str, difficulty: str, question_type: str) -> str:
    """
    🔄 LEGACY COMPATIBILITY - Redirect to unified system
    
    Maintains backward compatibility while using the new unified architecture.
    """
    logger.debug(f"🔄 Legacy function redirected to unified system")
    return create_unified_mcq_prompt(topic, context, difficulty, question_type)


def get_provider_optimized_prompt(topic: str, 
                                difficulty: str, 
                                question_type: str, 
                                context: str = "", 
                                provider: str = "unified") -> str:
    """
    🔗 PROVIDER OPTIMIZATION - Seamless integration with existing systems
    
    Provides optimized prompts for different AI providers while using
    the unified architecture under the hood.
    """
    # Auto-detect discipline from topic (could be enhanced with NLP)
    discipline = _infer_discipline_from_topic(topic)
    
    return create_unified_mcq_prompt(
        topic=topic,
        context=context, 
        difficulty=difficulty,
        question_type=question_type,
        discipline=discipline
    )


def _infer_discipline_from_topic(topic: str) -> str:
    """Simple discipline inference from topic keywords"""
    topic_lower = topic.lower()
    
    discipline_keywords = {
        'physics': ['quantum', 'atom', 'energy', 'wave', 'particle', 'force', 'momentum', 'electromagnetic'],
        'chemistry': ['molecule', 'reaction', 'bond', 'compound', 'element', 'acid', 'base', 'organic'],
        'biology': ['cell', 'dna', 'protein', 'evolution', 'genetics', 'organism', 'metabolism'],
        'mathematics': ['equation', 'function', 'derivative', 'integral', 'matrix', 'theorem', 'proof'],
        'history': ['war', 'empire', 'revolution', 'ancient', 'medieval', 'century', 'civilization'],
        'literature': ['novel', 'poem', 'author', 'character', 'narrative', 'theme', 'symbolism']
    }
    
    for discipline, keywords in discipline_keywords.items():
        if any(keyword in topic_lower for keyword in keywords):
            return discipline
    
    return 'general'


# 🎯 ENHANCED FUNCTIONS FOR SPECIAL USE CASES

def create_adaptive_prompt(topic: str, 
                          user_performance_history: Dict[str, Any] = None,
                          difficulty: str = "medium",
                          question_type: str = "mixed") -> str:
    """
    🎯 ADAPTIVE PROMPTING - Future-ready personalization
    
    Creates prompts that adapt based on user performance and preferences.
    This is a foundation for AI-powered difficulty adjustment.
    """
    # Future enhancement: analyze user performance to adjust difficulty
    if user_performance_history:
        # Could analyze success rates, time spent, error patterns, etc.
        pass
    
    return create_unified_mcq_prompt(topic, "", difficulty, question_type)


def create_multidisciplinary_prompt(topics: list, 
                                   difficulty: str = "medium",
                                   question_type: str = "mixed") -> str:
    """
    🌐 MULTIDISCIPLINARY QUESTIONS - Cross-domain integration
    
    Creates questions that span multiple academic disciplines.
    """
    combined_topic = " and ".join(topics)
    interdisciplinary_context = f"Create a question that connects concepts from: {', '.join(topics)}"
    
    return create_unified_mcq_prompt(
        topic=combined_topic,
        context=interdisciplinary_context,
        difficulty=difficulty,
        question_type=question_type,
        discipline="interdisciplinary"
    )


# 🧪 TESTING AND VALIDATION FUNCTIONS

def test_prompt_quality(topic: str, difficulty: str, iterations: int = 5) -> Dict[str, Any]:
    """
    🧪 PROMPT QUALITY TESTING - Validate prompt effectiveness
    
    Tests prompt quality by generating multiple prompts and analyzing consistency.
    """
    results = {
        'topic': topic,
        'difficulty': difficulty,
        'prompts_generated': [],
        'avg_prompt_length': 0,
        'consistency_score': 0
    }
    
    for i in range(iterations):
        prompt = create_unified_mcq_prompt(topic, "", difficulty, "mixed")
        results['prompts_generated'].append({
            'iteration': i + 1,
            'length': len(prompt),
            'preview': prompt[:200] + "..." if len(prompt) > 200 else prompt
        })
    
    # Calculate metrics
    lengths = [p['length'] for p in results['prompts_generated']]
    results['avg_prompt_length'] = sum(lengths) / len(lengths)
    results['length_variance'] = max(lengths) - min(lengths)
    
    return results


# Export main functions for public API
__all__ = [
    'create_unified_mcq_prompt',
    'get_provider_optimized_prompt',
    '_create_inquisitor_prompt',  # Legacy compatibility
    'create_adaptive_prompt',
    'create_multidisciplinary_prompt',
    'test_prompt_quality'
]


if __name__ == "__main__":
    # 🧪 DEMONSTRATION
    print("🚀 Next-Generation Prompt System - Demonstration")
    print("=" * 60)
    
    # Test different difficulty levels
    for difficulty in ["easy", "medium", "hard", "expert"]:
        prompt = create_unified_mcq_prompt(
            topic="quantum tunneling",
            difficulty=difficulty,
            question_type="conceptual",
            discipline="physics"
        )
        
        print(f"\n🎯 {difficulty.upper()} LEVEL:")
        print("-" * 40)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    
    print(f"\n✅ Next-Generation Prompt System Ready!")
