"""
✨ BALANCED PROMPT SYSTEM - "Supportive Excellence"
==================================================================

This module applies balanced educational philosophy: maintaining academic rigor
while providing supportive guidance that builds student confidence.

CORE PRINCIPLES:
1. Build Understanding - Focus on learning, not just testing
2. Support Growth - Challenge students while building confidence  
3. Quality Guidance - Clear direction without harsh constraints
4. Practical Excellence - Real-world applicable knowledge
5. Adaptive Support - Match guidance to student readiness

THE TRANSFORMATION:
==================
Old Constraint System → New Supportive System
🚫 "Don't use X"       → ✅ "Here's how to excel"
🚫 "Avoid Y"          → ✅ "Consider this approach"
🚫 "Never Z"          → ✅ "Best practices include"
🚫 "You will fail if" → ✅ "You can succeed by"

EDUCATIONAL PHILOSOPHY:
======================
Based on supportive learning environments that build confidence while
maintaining rigorous academic standards.

This eliminates:
❌ Harsh constraints and forbidden patterns
❌ Fear-based "you will fail if..." messaging  
❌ Rigid rule-following over understanding
❌ Academic pressure without support

This enables:
✅ Confident exploration of knowledge
✅ Balanced challenge with support
✅ Clear guidance for quality work
✅ Trust in student capabilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import random

logger = logging.getLogger(__name__)


class UnifiedPromptBuilder:
    """
    ✨ THE BALANCED PROMPT SYSTEM
    
    Creates questions that balance academic rigor with supportive guidance,
    helping students discover their capabilities while maintaining quality standards.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with config file or fallback configuration"""
        self.config_path = config_path or str(Path(__file__).parent / "prompt_config.json")
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from JSON file with graceful fallback"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"✅ Loaded balanced prompt configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Config file not found at {self.config_path}, using fallback")
            return self._get_fallback_config()
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in config file: {e}, using fallback")
            return self._get_fallback_config()
        except Exception as e:
            logger.error(f"❌ Unexpected error loading config: {e}, using fallback")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Minimal fallback configuration with balanced approach if file loading fails"""
        return {
            "personas": {
                "expert": "You are a distinguished educator who recognizes this advanced student's expertise in {topic}. Guide them toward profound insights with respect and high expectations.",
                "hard": "You are an experienced teacher who sees this student's growing mastery of {topic}. Create a challenge that builds their confidence while expanding their understanding.",
                "medium": "You are a supportive instructor who appreciates this student's progress in {topic}. Design a question that feels achievable yet enriching.",
                "easy": "You are an encouraging mentor who sees unlimited potential in this beginning explorer of {topic}. Make learning feel like a joyful discovery."
            },
            "difficulty_guidance": {
                "expert": "This capable mind is ready for advanced concepts. Honor their sophisticated understanding.",
                "hard": "This dedicated student has demonstrated readiness for greater challenges. Stretch their abilities thoughtfully.",
                "medium": "This learner is developing beautifully. Build their understanding step by step.", 
                "easy": "This curious student is beginning their journey. Make it accessible and engaging."
            },
            "examples": {"general": {}},
            "base_prompt_template": "{persona}\n\nSubject Area: {topic}\nLearning Objective: {difficulty_guidance}\n\nCreate a well-crafted {difficulty} level question as JSON."
        }
    
    def build_unified_prompt(self, 
                           topic: str, 
                           discipline: str = "general",
                           difficulty: str = "medium", 
                           question_type: str = "mixed",
                           context: str = "") -> str:
        """
        ✨ THE BALANCED CORE - Creates questions through supportive excellence
        
        Balances academic rigor with supportive guidance, helping students
        discover their capabilities while maintaining quality standards.
        
        Args:
            topic: The subject matter for exploration
            discipline: Field of knowledge (physics, chemistry, biology, etc.)
            difficulty: Learning level (easy, medium, hard, expert)
            question_type: Question style (numerical, conceptual, mixed)
            context: Additional guidance to include
            
        Returns:
            A thoughtfully crafted prompt that inspires growth and maintains standards
        """
        try:
            # Auto-detect discipline if not specified or if "general"
            if discipline == "general":
                discipline = self._auto_detect_discipline(topic)
            
            # 1. ✨ SUPPORTIVE GUIDANCE - The AI becomes an encouraging educator
            persona = self._get_nurturing_persona(difficulty, topic)
            
            # 2. 🎯 BALANCED DIRECTION - Clear guidance with encouragement
            guidance = self._get_wisdom_guidance(difficulty)
            
            # 3. 📚 INSPIRING EXAMPLES - Show excellence as aspiration, not constraint
            example = self._get_inspiring_example(discipline, difficulty, question_type)
            
            # 4. 🌟 THOUGHTFUL ASSEMBLY - Craft with care and academic rigor
            prompt = self._assemble_nurturing_prompt(persona, guidance, example, topic, context, difficulty)
            
            logger.debug(f"✨ Crafted balanced prompt for {topic} ({discipline}/{difficulty}/{question_type})")
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Prompt creation challenge: {e}")
            return self._get_gentle_fallback_prompt(topic, difficulty, question_type)
    
    def _auto_detect_discipline(self, topic: str) -> str:
        """Auto-detect discipline from topic keywords for comprehensive coverage"""
        topic_lower = topic.lower()
        
        # STEM disciplines
        if any(word in topic_lower for word in ['physics', 'quantum', 'mechanics', 'energy', 'force', 'wave', 'particle', 'thermodynamics', 'electromagnetism']):
            return "physics"
        elif any(word in topic_lower for word in ['chemistry', 'chemical', 'molecule', 'atom', 'reaction', 'element', 'compound', 'organic', 'inorganic']):
            return "chemistry"
        elif any(word in topic_lower for word in ['biology', 'cell', 'organism', 'evolution', 'genetics', 'ecology', 'physiology', 'anatomy', 'dna', 'rna']):
            return "biology"
        elif any(word in topic_lower for word in ['mathematics', 'math', 'algebra', 'calculus', 'geometry', 'statistics', 'equation', 'formula', 'theorem']):
            return "mathematics"
        
        # Humanities and Social Sciences
        elif any(word in topic_lower for word in ['history', 'historical', 'ancient', 'medieval', 'renaissance', 'war', 'empire', 'civilization', 'revolution']):
            return "history"
        elif any(word in topic_lower for word in ['literature', 'novel', 'poetry', 'shakespeare', 'author', 'writing', 'literary', 'narrative', 'prose']):
            return "literature"
        elif any(word in topic_lower for word in ['philosophy', 'philosophical', 'ethics', 'metaphysics', 'epistemology', 'logic', 'moral', 'kant', 'aristotle']):
            return "philosophy"
        elif any(word in topic_lower for word in ['psychology', 'psychological', 'cognitive', 'behavior', 'mental', 'therapy', 'freud', 'jung', 'perception']):
            return "psychology"
        elif any(word in topic_lower for word in ['economics', 'economic', 'market', 'finance', 'trade', 'supply', 'demand', 'capitalism', 'gdp']):
            return "economics"
        elif any(word in topic_lower for word in ['sociology', 'social', 'society', 'culture', 'community', 'inequality', 'class', 'race', 'gender']):
            return "sociology"
        elif any(word in topic_lower for word in ['political', 'politics', 'government', 'democracy', 'election', 'policy', 'law', 'constitution', 'state']):
            return "political_science"
        
        # Arts and Languages
        elif any(word in topic_lower for word in ['art', 'painting', 'sculpture', 'artist', 'renaissance', 'museum', 'gallery', 'aesthetic', 'visual']):
            return "art"
        elif any(word in topic_lower for word in ['music', 'musical', 'symphony', 'opera', 'composer', 'instrument', 'melody', 'harmony', 'rhythm']):
            return "music"
        elif any(word in topic_lower for word in ['language', 'linguistic', 'grammar', 'syntax', 'semantics', 'phonetic', 'dialect', 'translation']):
            return "linguistics"
        elif any(word in topic_lower for word in ['english', 'french', 'spanish', 'german', 'chinese', 'japanese', 'latin', 'vocabulary', 'pronunciation']):
            return "language"
        
        # Technology and Applied Sciences
        elif any(word in topic_lower for word in ['computer', 'programming', 'software', 'algorithm', 'data', 'coding', 'technology', 'digital']):
            return "computer_science"
        elif any(word in topic_lower for word in ['engineering', 'mechanical', 'electrical', 'civil', 'design', 'construction', 'technical']):
            return "engineering"
        
        return "general"
    
    def _get_nurturing_persona(self, difficulty: str, topic: str) -> str:
        """Get the supportive educator persona that guides through encouragement"""
        persona_template = self.config["personas"].get(
            difficulty, 
            self.config["personas"].get("medium", "You are a supportive teacher who sees potential in every student.")
        )
        return persona_template.format(topic=topic)
    
    def _get_wisdom_guidance(self, difficulty: str) -> str:
        """Get gentle, encouraging guidance instead of harsh constraints"""
        return self.config["difficulty_guidance"].get(
            difficulty,
            self.config["difficulty_guidance"].get("medium", "This student is ready to grow. Guide them thoughtfully.")
        )
    
    def _get_inspiring_example(self, discipline: str, difficulty: str, question_type: str) -> Dict[str, Any]:
        """
        ✨ INSPIRATION THROUGH EXAMPLE
        
        Instead of constraining with rules, we inspire with beautiful examples.
        Like showing a student a master's work to inspire their own excellence.
        """
        # Try to find discipline-specific example
        discipline_examples = self.config["examples"].get(discipline, {})
        
        # Look for exact match first
        example_key = f"{difficulty}_{question_type}"
        if example_key in discipline_examples:
            return discipline_examples[example_key]
        
        # Fall back to difficulty-only match
        difficulty_examples = {k: v for k, v in discipline_examples.items() if difficulty in k}
        if difficulty_examples:
            return random.choice(list(difficulty_examples.values()))
        
        # Fall back to any example from this discipline
        if discipline_examples:
            return random.choice(list(discipline_examples.values()))
        
        # Ultimate fallback to general examples
        general_examples = self.config["examples"].get("general", {})
        if general_examples:
            return random.choice(list(general_examples.values()))
        
        # If all else fails, return empty example
        return {}
    
    def _assemble_nurturing_prompt(self, persona: str, guidance: str, example: Dict[str, Any], 
                                 topic: str, context: str, difficulty: str) -> str:
        """
        🌟 THOUGHTFUL ASSEMBLY - Weave together wisdom and guidance
        
        Creates a unified prompt that maintains academic standards while
        providing supportive guidance for student growth.
        """
        base_template = self.config.get("base_prompt_template", 
            "{persona}\n\nSubject: {topic}\nGuidance: {difficulty_guidance}\n\nCreate a {difficulty} question as JSON.")
        
        # Format the base template with all required parameters
        try:
            assembled_prompt = base_template.format(
                persona=persona,
                topic=topic,
                difficulty_guidance=guidance,
                difficulty=difficulty,
                context=context if context else "",
                example_json=json.dumps(example, indent=2) if example else "{}"
            )
        except KeyError as e:
            # Fallback if template format is incompatible
            assembled_prompt = f"{persona}\n\nSubject: {topic}\nGuidance: {guidance}\n\nCreate a {difficulty} question as JSON."
        
        # Add context if provided (only if not already in template)
        if context.strip() and "context" not in base_template.lower():
            assembled_prompt += f"\n\nAdditional Context: {context.strip()}"
        
        # Add inspiring example if available (only if not already in template)
        if example and "example" not in base_template.lower():
            assembled_prompt += f"\n\nInspiring Example Format:\n{json.dumps(example, indent=2)}"
        
        # Add the JSON format requirement (only if not already in template)
        if "json" not in assembled_prompt.lower():
            assembled_prompt += "\n\nPlease provide your response as a properly formatted JSON object."
        
        return assembled_prompt
    
    def _get_gentle_fallback_prompt(self, topic: str, difficulty: str, question_type: str) -> str:
        """
        💝 GENTLE FALLBACK - When things go wrong, stay supportive
        
        Even in error conditions, maintain the supportive tone that helps
        students feel confident and capable.
        """
        return f"""You are a supportive educator who helps students explore {topic} with confidence.

Create a {difficulty} level {question_type} question that:
- Builds understanding rather than just testing
- Provides clear guidance for success
- Maintains appropriate academic standards
- Helps the student feel capable and supported

Please provide your response as a properly formatted JSON object with:
- "question": the question text
- "answer": the correct answer
- "explanation": supportive explanation
- "difficulty": "{difficulty}"
- "topic": "{topic}"
"""

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration for debugging"""
        return {
            "config_path": self.config_path,
            "has_config": bool(self.config),
            "personas_count": len(self.config.get("personas", {})),
            "difficulty_levels": list(self.config.get("difficulty_guidance", {}).keys()),
            "example_disciplines": list(self.config.get("examples", {}).keys()),
            "template_available": "base_prompt_template" in self.config
        }
