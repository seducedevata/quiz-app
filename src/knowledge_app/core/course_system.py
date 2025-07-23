"""
🎓 Advanced Course System for Knowledge App
Manages expert-level PhD questions, prevents UI freezing, and integrates with LaTeX rendering
"""

import logging
import asyncio
import threading
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)

class CourseLevel(Enum):
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate" 
    PHD = "phd"
    EXPERT = "expert"
    RESEARCH = "research"

class QuestionComplexity(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    RESEARCH_LEVEL = "research_level"

@dataclass
class CourseModule:
    """Represents a course module with specific learning objectives"""
    id: str
    title: str
    level: CourseLevel
    topics: List[str]
    prerequisites: List[str]
    learning_objectives: List[str]
    complexity: QuestionComplexity
    estimated_duration: int  # minutes
    latex_heavy: bool = False

@dataclass
class ExpertQuestion:
    """Represents an expert-level question with metadata"""
    question_id: str
    question_text: str
    options: List[str]
    correct_answer: str
    explanation: str
    topic: str
    level: CourseLevel
    complexity: QuestionComplexity
    latex_content: bool
    cognitive_load: int  # 1-10 scale
    research_area: Optional[str] = None
    source_paper: Optional[str] = None

class NonBlockingCourseSystem:
    """
    🚀 NON-BLOCKING Course System
    Prevents UI freezing while generating expert-level content
    """
    
    def __init__(self):
        self.courses: Dict[str, Dict[str, CourseModule]] = {}
        self.question_cache: Dict[str, List[ExpertQuestion]] = {}
        # 🔧 BUG FIX 37: Proper thread pool lifecycle management
        self.generation_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="CourseSystem"
        )
        self.active_generations = set()
        self._shutdown_requested = False
        
        # Load predefined course structures
        self._initialize_default_courses()
        
        # LaTeX-aware question templates
        self.expert_templates = {
            CourseLevel.PHD: {
                "mathematics": [
                    "Advanced analysis of {topic} using {method}. Given the constraints {constraint}, derive {target}.",
                    "Prove or disprove: {statement} holds for all {domain} under {conditions}.",
                    "Research problem: Investigate {phenomenon} in {context}. What are the implications for {field}?"
                ],
                "physics": [
                    "Quantum mechanical treatment of {system}. Calculate {observable} using {formalism}.",
                    "Field theory approach to {problem}. Derive {equation} from first principles.",
                    "Experimental design: How would you measure {quantity} in {environment}?"
                ],
                "chemistry": [
                    "Mechanistic analysis of {reaction} under {conditions}. Predict {outcome}.",
                    "Computational study of {molecule}. What is the {property} and why?",
                    "Synthetic strategy for {target_compound} from {starting_materials}."
                ]
            },
            CourseLevel.EXPERT: {
                "general": [
                    "Critical evaluation of {theory} in light of {recent_findings}.",
                    "Design an experiment to test {hypothesis} considering {limitations}.",
                    "Compare and contrast {approach_a} vs {approach_b} for {application}."
                ]
            }
        }
        
    def _initialize_default_courses(self):
        """Initialize predefined course structures"""
        
        # Advanced Mathematics Course
        math_modules = {
            "real_analysis": CourseModule(
                id="real_analysis",
                title="Advanced Real Analysis",
                level=CourseLevel.PHD,
                topics=["measure theory", "functional analysis", "topology"],
                prerequisites=["undergraduate_analysis", "linear_algebra"],
                learning_objectives=[
                    "Master measure-theoretic foundations",
                    "Understand Banach and Hilbert spaces",
                    "Apply topological methods"
                ],
                complexity=QuestionComplexity.EXPERT,
                estimated_duration=180,
                latex_heavy=True
            ),
            "algebraic_topology": CourseModule(
                id="algebraic_topology", 
                title="Algebraic Topology",
                level=CourseLevel.PHD,
                topics=["homology", "homotopy", "cohomology"],
                prerequisites=["abstract_algebra", "point_set_topology"],
                learning_objectives=[
                    "Compute homology groups",
                    "Understand homotopy theory",
                    "Apply spectral sequences"
                ],
                complexity=QuestionComplexity.RESEARCH_LEVEL,
                estimated_duration=240,
                latex_heavy=True
            )
        }
        
        # Advanced Physics Course
        physics_modules = {
            "quantum_field_theory": CourseModule(
                id="quantum_field_theory",
                title="Quantum Field Theory",
                level=CourseLevel.PHD,
                topics=["path integrals", "gauge theory", "renormalization"],
                prerequisites=["quantum_mechanics", "statistical_mechanics"],
                learning_objectives=[
                    "Master path integral formalism",
                    "Understand gauge invariance",
                    "Perform renormalization calculations"
                ],
                complexity=QuestionComplexity.EXPERT,
                estimated_duration=300,
                latex_heavy=True
            ),
            "general_relativity": CourseModule(
                id="general_relativity",
                title="General Relativity",
                level=CourseLevel.PHD, 
                topics=["spacetime geometry", "Einstein equations", "black holes"],
                prerequisites=["special_relativity", "differential_geometry"],
                learning_objectives=[
                    "Solve Einstein field equations",
                    "Analyze spacetime metrics",
                    "Understand black hole physics"
                ],
                complexity=QuestionComplexity.EXPERT,
                estimated_duration=240,
                latex_heavy=True
            )
        }
        
        self.courses = {
            "advanced_mathematics": math_modules,
            "theoretical_physics": physics_modules
        }
        
    async def generate_expert_questions_async(self, topic: str, level: CourseLevel, 
                                            count: int = 5, 
                                            mcq_manager=None) -> List[ExpertQuestion]:
        """
        🚀 NON-BLOCKING expert question generation
        Prevents UI freezing during complex question generation
        """
        if not mcq_manager:
            logger.error("❌ MCQ manager required for expert question generation")
            return []
            
        generation_id = f"{topic}_{level.value}_{int(time.time())}"
        
        if generation_id in self.active_generations:
            logger.warning("⚠️ Generation already active for this request")
            return []
            
        self.active_generations.add(generation_id)
        
        try:
            logger.info(f"🎓 Generating {count} expert-level questions for '{topic}' at {level.value} level")
            
            # Use thread pool to prevent UI blocking
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._generate_expert_questions_sync,
                    topic, level, count, mcq_manager
                )
                
                # Wait with timeout to prevent hanging
                questions = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=120.0  # 2 minute timeout
                )
                
            logger.info(f"✅ Successfully generated {len(questions)} expert questions")
            return questions
            
        except asyncio.TimeoutError:
            logger.error("⏰ Expert question generation timed out")
            return []
        except Exception as e:
            logger.error(f"❌ Expert question generation failed: {e}")
            return []
        finally:
            self.active_generations.discard(generation_id)
    
    def _generate_expert_questions_sync(self, topic: str, level: CourseLevel, 
                                      count: int, mcq_manager) -> List[ExpertQuestion]:
        """Synchronous expert question generation with enhanced prompting"""
        questions = []
        
        for i in range(count):
            try:
                # Create expert-level prompt
                expert_prompt = self._create_expert_prompt(topic, level, i)
                
                # Enhanced parameters for expert generation
                quiz_params = {
                    "topic": expert_prompt,
                    "difficulty": "expert",
                    "game_mode": "serious", 
                    "submode": "mixed",
                    "num_questions": 1,
                    "expert_mode": True,
                    "latex_enabled": True,
                    "cognitive_complexity": "high"
                }
                
                # Generate with enhanced validation
                mcq_result = mcq_manager.generate_quiz(quiz_params)
                
                if mcq_result and len(mcq_result) > 0:
                    # Convert to expert question format
                    expert_q = self._convert_to_expert_question(
                        mcq_result[0], topic, level, i
                    )
                    if expert_q:
                        questions.append(expert_q)
                        logger.info(f"✅ Expert question {i+1}/{count} created")
                else:
                    logger.warning(f"⚠️ Expert question {i+1} generation returned no result")
                    
            except Exception as e:
                logger.error(f"❌ Expert question {i+1} generation failed: {e}")
                continue
                
        return questions
    
    def _create_expert_prompt(self, topic: str, level: CourseLevel, index: int) -> str:
        """Create sophisticated expert-level prompts"""
        
        base_topic = topic.lower().strip()
        
        # PhD-level complexity indicators
        complexity_phrases = [
            "advanced theoretical framework",
            "cutting-edge research methodology", 
            "interdisciplinary analysis",
            "novel experimental approach",
            "computational modeling",
            "theoretical derivation",
            "critical evaluation",
            "comparative analysis"
        ]
        
        # Mathematical sophistication
        math_elements = [
            "rigorous proof techniques",
            "mathematical formalism",
            "quantitative analysis", 
            "statistical modeling",
            "differential equations",
            "linear algebra applications",
            "optimization theory",
            "stochastic processes"
        ]
        
        # Research context
        research_contexts = [
            "current literature",
            "recent experimental findings",
            "theoretical developments",
            "open research problems",
            "methodological innovations",
            "interdisciplinary connections"
        ]
        
        complexity_phrase = complexity_phrases[index % len(complexity_phrases)]
        math_element = math_elements[index % len(math_elements)]
        research_context = research_contexts[index % len(research_contexts)]
        
        if level == CourseLevel.PHD:
            prompt = f"""Advanced graduate-level question on {topic}:
            
            Context: A PhD student is studying {topic} using {complexity_phrase} and {math_element}. 
            They need to demonstrate mastery of {research_context} in this field.
            
            Create a challenging question that requires:
            - Deep theoretical understanding
            - Advanced mathematical reasoning
            - Critical analysis of complex concepts
            - Integration of multiple theoretical frameworks
            
            The question should be appropriate for someone with:
            - Advanced undergraduate preparation
            - Graduate-level coursework
            - Research experience in {topic}
            
            Topic: {topic}
            Cognitive Level: Expert analysis and synthesis
            Mathematical Content: Include relevant equations, formulas, or mathematical notation where appropriate
            """
        elif level == CourseLevel.EXPERT:
            prompt = f"""Expert-level question on {topic}:
            
            Create a research-level question that would challenge an expert in {topic}.
            The question should require sophisticated understanding of {complexity_phrase} 
            and demonstrate mastery of {math_element}.
            
            Context: {research_context} in {topic}
            Level: Post-doctoral research level
            Complexity: Requires original thinking and deep expertise
            
            Topic: {topic}
            """
        else:
            prompt = f"""Graduate-level question on {topic} requiring {complexity_phrase} and {math_element}."""
            
        return prompt
    
    def _convert_to_expert_question(self, mcq_result, topic: str, 
                                  level: CourseLevel, index: int) -> Optional[ExpertQuestion]:
        """Convert MCQ result to expert question format"""
        try:
            if hasattr(mcq_result, 'question'):
                question_text = mcq_result.question
                options = mcq_result.options
                correct_answer = mcq_result.correct_answer
                explanation = getattr(mcq_result, 'explanation', 'No explanation provided.')
            elif isinstance(mcq_result, dict):
                question_text = mcq_result.get('question', '')
                options = list(mcq_result.get('options', {}).values()) if 'options' in mcq_result else []
                correct_answer = mcq_result.get('options', {}).get(mcq_result.get('correct', 'A'), '')
                explanation = mcq_result.get('explanation', 'No explanation provided.')
            else:
                logger.error("❌ Invalid MCQ result format")
                return None
                
            # Detect LaTeX content
            latex_indicators = ['$', '\\', 'frac', 'sum', 'int', 'alpha', 'beta', 'gamma']
            has_latex = any(indicator in question_text + explanation for indicator in latex_indicators)
            
            # Calculate cognitive load (1-10 scale)
            cognitive_load = self._calculate_cognitive_load(question_text, options, explanation, level)
            
            return ExpertQuestion(
                question_id=f"{topic}_{level.value}_{index}_{int(time.time())}",
                question_text=question_text,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
                topic=topic,
                level=level,
                complexity=QuestionComplexity.EXPERT if level == CourseLevel.PHD else QuestionComplexity.ADVANCED,
                latex_content=has_latex,
                cognitive_load=cognitive_load,
                research_area=topic,
                source_paper=None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to convert MCQ result to expert question: {e}")
            return None
    
    def _calculate_cognitive_load(self, question: str, options: List[str], 
                                explanation: str, level: CourseLevel) -> int:
        """Calculate cognitive complexity on 1-10 scale"""
        base_score = {
            CourseLevel.UNDERGRADUATE: 4,
            CourseLevel.GRADUATE: 6,
            CourseLevel.PHD: 8,
            CourseLevel.EXPERT: 9,
            CourseLevel.RESEARCH: 10
        }.get(level, 5)
        
        # Adjust based on content complexity
        complexity_indicators = [
            'proof', 'derive', 'analyze', 'synthesize', 'evaluate', 'compare',
            'theorem', 'lemma', 'corollary', 'research', 'novel', 'innovative'
        ]
        
        text_to_analyze = (question + ' ' + explanation).lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in text_to_analyze)
        
        # Mathematical content increases load
        math_indicators = ['equation', 'formula', '$', '\\', 'integral', 'derivative']
        math_count = sum(1 for indicator in math_indicators if indicator in text_to_analyze)
        
        final_score = min(10, base_score + complexity_count + math_count)
        return final_score
    
    def get_course_progression(self, subject: str) -> List[CourseModule]:
        """Get recommended course progression for a subject"""
        if subject in self.courses:
            modules = list(self.courses[subject].values())
            # Sort by complexity and prerequisites
            return sorted(modules, key=lambda m: (m.complexity.value, len(m.prerequisites)))
        return []
    
    def recommend_next_module(self, completed_modules: List[str], subject: str) -> Optional[CourseModule]:
        """Recommend next module based on completed prerequisites"""
        available_modules = self.courses.get(subject, {})
        
        for module in available_modules.values():
            if module.id not in completed_modules:
                # Check if prerequisites are met
                if all(prereq in completed_modules for prereq in module.prerequisites):
                    return module
        return None

    def shutdown(self):
        """🔧 BUG FIX 37: Proper thread pool shutdown and resource cleanup"""
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info("🔄 Shutting down NonBlockingCourseSystem...")

        try:
            # Cancel active generations
            for generation_id in list(self.active_generations):
                logger.info(f"🔄 Cancelling active generation: {generation_id}")

            # Shutdown thread pool gracefully (timeout not supported in older Python versions)
            self.generation_pool.shutdown(wait=True)
            logger.info("✅ Thread pool shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            # Force shutdown if graceful shutdown fails
            try:
                self.generation_pool.shutdown(wait=False)
            except:
                pass

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup

# Global instance
_course_system = None

def get_course_system() -> NonBlockingCourseSystem:
    """Get or create global course system instance"""
    global _course_system
    if _course_system is None:
        _course_system = NonBlockingCourseSystem()
    return _course_system