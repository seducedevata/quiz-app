"""
Advanced RAG MCQ Generator - The Grounded Scholar

This module implements the Grounded Scholar approach using RAG to create
highly specific, factually accurate MCQs grounded in user's source material.
"""

import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .mcq_validator import MCQValidator, ValidationResult, fix_common_mcq_issues

logger = logging.getLogger(__name__)


class CognitiveLevel(Enum):
    """Bloom's Taxonomy cognitive levels"""

    REMEMBERING = "remembering"
    UNDERSTANDING = "understanding"
    APPLYING = "applying"
    ANALYZING = "analyzing"
    EVALUATING = "evaluating"
    CREATING = "creating"


@dataclass
class MCQGenerationContext:
    """Context for MCQ generation"""

    topic: str
    difficulty: str
    cognitive_level: CognitiveLevel
    context_passages: List[str]
    source_documents: List[str]
    target_audience: str


class AdvancedRAGMCQGenerator:
    """
    The Grounded Scholar - Advanced RAG-based MCQ generator

    This generator uses RAG to retrieve specific, relevant passages from
    the user's uploaded books and creates questions grounded in that content.
    """

    def __init__(self, rag_engine=None, model_interface=None):
        self.rag_engine = rag_engine
        self.model_interface = model_interface
        self.validator = MCQValidator()

        # Cognitive level prompts
        self.cognitive_prompts = {
            CognitiveLevel.REMEMBERING: "Create a question that tests recall of facts and basic concepts from the context.",
            CognitiveLevel.UNDERSTANDING: "Create a question that tests explanation and comprehension of ideas from the context.",
            CognitiveLevel.APPLYING: "Create a question that tests application of information to new situations based on the context.",
            CognitiveLevel.ANALYZING: "Create a question that tests analysis of relationships and connections in the context.",
            CognitiveLevel.EVALUATING: "Create a question that tests evaluation and judgment based on the context.",
            CognitiveLevel.CREATING: "Create a question that tests synthesis and creation of new ideas from the context.",
        }

    async def generate_grounded_mcq(
        self,
        topic: str,
        difficulty: str = "medium",
        cognitive_level: CognitiveLevel = CognitiveLevel.UNDERSTANDING,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a high-quality MCQ using the Grounded Scholar approach

        Args:
            topic: The topic for the question
            difficulty: Question difficulty level
            cognitive_level: Bloom's taxonomy level
            max_retries: Maximum number of generation attempts

        Returns:
            Dictionary containing the generated MCQ
        """
        try:
            logger.info(f"🔍 Generating grounded MCQ for topic: {topic}")

            # Step 1: Retrieve relevant context using RAG
            context_passages = await self._retrieve_context(topic)

            if not context_passages:
                logger.error("❌ No relevant context found for RAG generation")
                raise Exception(f"No relevant context found for topic '{topic}' - RAG generation requires context")

            # Step 2: Create generation context
            generation_context = MCQGenerationContext(
                topic=topic,
                difficulty=difficulty,
                cognitive_level=cognitive_level,
                context_passages=context_passages,
                source_documents=[],  # Will be populated by RAG engine
                target_audience=self._get_target_audience(difficulty),
            )

            # Step 3: Generate MCQ with validation loop
            for attempt in range(max_retries):
                try:
                    mcq = await self._generate_mcq_with_context(generation_context)

                    # Step 4: Validate the generated MCQ
                    validation_result = self.validator.validate_mcq(mcq)

                    if validation_result.is_valid:
                        logger.info(f"✅ Generated valid grounded MCQ (attempt {attempt + 1})")
                        mcq["validation_score"] = validation_result.score
                        mcq["grounded"] = True
                        mcq["source_passages"] = len(context_passages)
                        return mcq
                    else:
                        logger.warning(
                            f"Generated MCQ failed validation (attempt {attempt + 1}): {validation_result.issues}"
                        )

                        # Try to fix common issues
                        if attempt < max_retries - 1:
                            mcq = fix_common_mcq_issues(mcq)
                            validation_result = self.validator.validate_mcq(mcq)
                            if validation_result.is_valid:
                                logger.info(
                                    f"✅ Fixed MCQ validation issues (attempt {attempt + 1})"
                                )
                                mcq["validation_score"] = validation_result.score
                                mcq["grounded"] = True
                                mcq["source_passages"] = len(context_passages)
                                return mcq

                except Exception as e:
                    logger.error(f"Error in MCQ generation attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise

            # If all attempts failed, raise error
            logger.error("❌ All MCQ generation attempts failed")
            raise Exception(f"All MCQ generation attempts failed for topic '{topic}'")

        except Exception as e:
            logger.error(f"❌ Error in grounded MCQ generation: {e}")
            raise Exception(f"Grounded MCQ generation failed for '{topic}': {str(e)}")

    async def _retrieve_context(self, topic: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context passages using RAG"""
        try:
            if not self.rag_engine:
                logger.warning("RAG engine not available")
                return []

            # Use RAG engine to retrieve context (synchronous method)
            if hasattr(self.rag_engine, "retrieve_context"):
                contexts = self.rag_engine.retrieve_context(topic, top_k=top_k)
                logger.info(f"Retrieved {len(contexts)} context passages for topic: {topic}")
                return contexts
            else:
                logger.warning("RAG engine doesn't support retrieve_context method")
                return []

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    async def _generate_mcq_with_context(self, context: MCQGenerationContext) -> Dict[str, Any]:
        """Generate MCQ using the provided context"""

        # Combine context passages
        combined_context = "\n\n".join(context.context_passages)

        # Create the Inquisitor's Mandate prompt
        prompt = self._create_inquisitor_prompt(context, combined_context)

        # Generate using model interface
        if self.model_interface:
            # 🚀 CRITICAL FIX: model_interface.generate_text is already async, so await it directly
            response = await self.model_interface.generate_text(prompt)
        else:
            # No model interface available
            raise Exception(f"No model interface available for MCQ generation for topic '{context.topic}'")

        # Parse JSON response
        try:
            # Debug logging to see what we're getting
            logger.info(f"Raw response from model (first 500 chars): {response[:500] if response else 'None'}")
            
            # Clean the response (remove markdown formatting if present)
            cleaned_response = self._clean_json_response(response)
            mcq_data = json.loads(cleaned_response)

            # Add metadata
            mcq_data["generation_method"] = "grounded_scholar"
            mcq_data["cognitive_level"] = context.cognitive_level.value
            mcq_data["context_length"] = len(combined_context)

            return mcq_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from model: {e}")

    def _create_inquisitor_prompt(
        self, context: MCQGenerationContext, combined_context: str
    ) -> str:
        """Create the Inquisitor's Mandate prompt for grounded generation"""
        from .inquisitor_prompt import _create_inquisitor_prompt

        return _create_inquisitor_prompt(combined_context, context.topic, context.difficulty)

    def _clean_json_response(self, response: str) -> str:
        """
        Clean the model response to extract valid JSON.
        Now handles both old format (correct_answer) and new format (correct).
        """
        try:
            # Log raw response for debugging
            logger.info(f"[DEBUG] Raw response from model: {response[:500]}...")
            
            # First, try to find a clean JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                potential_json = match.group(0)
                logger.info(f"[DEBUG] Found potential JSON block: {potential_json[:200]}...")
                try:
                    # Try to parse it directly
                    parsed = json.loads(potential_json)
                    # Normalize the format
                    if "correct" in parsed and "correct_answer" not in parsed:
                        parsed["correct_answer"] = parsed["correct"]
                    elif "correct_answer" in parsed and "correct" not in parsed:
                        parsed["correct"] = parsed["correct_answer"]
                    # Handle options format (dict vs list)
                    if "options" in parsed and isinstance(parsed["options"], dict):
                        # Convert dict to list
                        parsed["options"] = [parsed["options"][k] for k in sorted(parsed["options"].keys())]
                    return json.dumps(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(f"[DEBUG] JSON decode error: {e}")
                    # Fallback to manual extraction if parsing fails
                    pass
            else:
                logger.warning("[DEBUG] No JSON block found with regex")

            # Manual extraction using regex if JSON is completely broken
            question_match = re.search(r'"question":\s*"(.*?)"', response, re.DOTALL)
            # Try both formats
            correct_match = re.search(r'"correct":\s*"(.*?)"', response, re.DOTALL) or \
                           re.search(r'"correct_answer":\s*"(.*?)"', response, re.DOTALL)
            explanation_match = re.search(r'"explanation":\s*"(.*?)"', response, re.DOTALL)
            
            # Handle options in different formats
            options_dict_match = re.search(r'"options":\s*\{(.*?)\}', response, re.DOTALL)
            options_list_match = re.search(r'"options":\s*\[(.*?)\]', response, re.DOTALL)
            
            if question_match and correct_match and (options_dict_match or options_list_match):
                question = question_match.group(1).strip()
                correct_answer = correct_match.group(1).strip()
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                
                # Parse options
                if options_dict_match:
                    # Parse dict format {"A": "...", "B": "..."}
                    options_str = options_dict_match.group(1)
                    options = []
                    for opt_match in re.finditer(r'"[A-D]":\s*"(.*?)"', options_str):
                        options.append(opt_match.group(1))
                else:
                    # Parse list format ["...", "..."]
                    options_str = options_list_match.group(1)
                    options = [opt.strip().strip('"') for opt in options_str.split(',')]

                # Reconstruct a valid JSON object
                reconstructed_json = {
                    "question": question,
                    "options": options,
                    "correct": correct_answer,
                    "correct_answer": correct_answer,
                    "explanation": explanation,
                }
                logger.info(f"[DEBUG] Successfully reconstructed JSON: {reconstructed_json}")
                return json.dumps(reconstructed_json, indent=2)

            # 🛡️ CRITICAL FIX: Raise explicit error instead of returning empty JSON
            error_msg = f"Could not parse or reconstruct JSON from response. Raw response: {response[:200] if response else 'None'}..."
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            # 🛡️ CRITICAL FIX: Raise explicit error with context instead of returning empty JSON
            error_msg = f"Critical error in JSON cleaning: {str(e)}. Raw response: {response[:200] if response else 'None'}..."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _get_target_audience(self, difficulty: str) -> str:
        """Get target audience description based on difficulty"""
        audience_map = {
            "easy": "a high-school student",
            "medium": "an undergraduate university student",
            "hard": "a graduate student specializing in the field",
            "expert": "a post-doctoral researcher",
        }
        return audience_map.get(difficulty.lower(), "an undergraduate university student")



    async def generate_with_two_step_distractors(
        self,
        topic: str,
        difficulty: str = "medium",
        cognitive_level: CognitiveLevel = CognitiveLevel.UNDERSTANDING,
    ) -> Dict[str, Any]:
        """
        Generate MCQ using two-step process for better distractors

        Step 1: Generate question and correct answer
        Step 2: Generate high-quality distractors
        """
        try:
            logger.info(f"🎯 Generating MCQ with two-step distractor process for: {topic}")

            # Step 1: Retrieve context
            context_passages = await self._retrieve_context(topic)
            if not context_passages:
                raise Exception(f"No context passages found for two-step generation for topic '{topic}'")

            combined_context = "\n\n".join(context_passages)

            # Step 2: Generate question and correct answer only
            question_prompt = self._create_question_only_prompt(
                topic, combined_context, difficulty, cognitive_level
            )
            loop = asyncio.get_event_loop()
            question_response = await loop.run_in_executor(
                None, self.model_interface.generate_text, question_prompt
            )

            try:
                question_data = json.loads(self._clean_json_response(question_response))
            except json.JSONDecodeError:
                logger.error("Failed to parse question-only response")
                raise Exception(f"Failed to parse question-only response for topic '{topic}'")

            # Step 3: Generate distractors specifically
            distractor_prompt = self._create_distractor_prompt(
                combined_context, question_data["question"], question_data["correct_answer"], topic
            )
            distractor_response = await loop.run_in_executor(
                None, self.model_interface.generate_text, distractor_prompt
            )

            try:
                distractors = json.loads(self._clean_json_response(distractor_response))
            except json.JSONDecodeError:
                logger.error("Failed to parse distractor response")
                # Use basic distractors
                distractors = [
                    f"Alternative concept related to {topic}",
                    f"Common misconception about {topic}",
                    f"Related but incorrect principle",
                ]

            # Step 4: Combine into final MCQ
            final_mcq = {
                "question": question_data["question"],
                "options": {
                    "A": question_data["correct_answer"],
                    "B": distractors[0] if len(distractors) > 0 else "Alternative option",
                    "C": distractors[1] if len(distractors) > 1 else "Different approach",
                    "D": distractors[2] if len(distractors) > 2 else "Other method",
                },
                "correct": "A",
                "explanation": question_data.get(
                    "explanation", f"This is the correct answer based on the context about {topic}."
                ),
                "generation_method": "two_step_distractors",
                "grounded": True,
                "cognitive_level": cognitive_level.value,
            }

            # Validate the final result
            validation_result = self.validator.validate_mcq(final_mcq)
            if validation_result.is_valid:
                final_mcq["validation_score"] = validation_result.score
                logger.info("✅ Two-step MCQ generation successful")
                return final_mcq
            else:
                logger.warning("Two-step MCQ failed validation, applying fixes")
                return fix_common_mcq_issues(final_mcq)

        except Exception as e:
            logger.error(f"Error in two-step generation: {e}")
            raise Exception(f"Two-step generation failed for topic '{topic}': {str(e)}")

    def _create_question_only_prompt(
        self, topic: str, context: str, difficulty: str, cognitive_level: CognitiveLevel
    ) -> str:
        """Create prompt for generating question and correct answer only"""

        cognitive_instruction = self.cognitive_prompts.get(
            cognitive_level, self.cognitive_prompts[CognitiveLevel.UNDERSTANDING]
        )
        target_audience = self._get_target_audience(difficulty)

        return f"""### ROLE & GOAL ###
You are a university professor creating an exam question. Generate ONLY a question and its correct answer based on the provided context about '{topic}'.

### COGNITIVE LEVEL ###
{cognitive_instruction}

### CONTEXT ###
{context}

### TASK ###
1. Identify a key concept from the context related to '{topic}'
2. Create a clear, specific question about this concept
3. Provide the correct answer based on the context
4. Write a brief explanation

### TARGET AUDIENCE ###
{target_audience}

### OUTPUT FORMAT ###
Respond with ONLY this JSON structure:
{{
  "question": "Your question here",
  "correct_answer": "The correct answer here",
  "explanation": "Brief explanation of why this is correct"
}}"""

    def _create_distractor_prompt(
        self, context: str, question: str, correct_answer: str, topic: str
    ) -> str:
        """Create prompt for generating high-quality distractors"""

        return f"""### ROLE & GOAL ###
You are an expert test designer specializing in creating challenging distractors. Based on the context, question, and correct answer provided, generate three incorrect options that target common student misconceptions about '{topic}'.

### CONTEXT ###
{context}

### QUESTION ###
{question}

### CORRECT ANSWER ###
{correct_answer}

### TASK ###
Generate three plausible but incorrect options. They should:
1. Be similar in length and style to the correct answer
2. Target common misconceptions about the topic
3. Be factually incorrect according to the context
4. Sound plausible to someone who hasn't fully understood the concept

### OUTPUT FORMAT ###
Respond with ONLY a JSON array:
["Distractor A", "Distractor B", "Distractor C"]"""

    async def generate_multiple_grounded_mcqs(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        cognitive_levels: Optional[List[CognitiveLevel]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple grounded MCQs with varied cognitive levels"""

        if cognitive_levels is None:
            # Use a mix of cognitive levels
            cognitive_levels = [
                CognitiveLevel.UNDERSTANDING,
                CognitiveLevel.APPLYING,
                CognitiveLevel.ANALYZING,
                CognitiveLevel.REMEMBERING,
                CognitiveLevel.EVALUATING,
            ]

        questions = []

        for i in range(num_questions):
            cognitive_level = cognitive_levels[i % len(cognitive_levels)]

            try:
                # Alternate between regular and two-step generation
                if i % 2 == 0:
                    mcq = await self.generate_grounded_mcq(
                        topic=f"{topic} (aspect {i+1})",
                        difficulty=difficulty,
                        cognitive_level=cognitive_level,
                    )
                else:
                    mcq = await self.generate_with_two_step_distractors(
                        topic=f"{topic} (aspect {i+1})",
                        difficulty=difficulty,
                        cognitive_level=cognitive_level,
                    )

                if mcq:
                    questions.append(mcq)
                    logger.info(f"Generated grounded MCQ {i+1}/{num_questions}")

            except Exception as e:
                logger.error(f"Failed to generate MCQ {i+1}: {e}")

        logger.info(f"✅ Generated {len(questions)} grounded MCQs for topic: {topic}")
        return questions

    async def generate_quiz_async(
        self,
        pure_content: str,
        topic: str,
        difficulty: str = "medium",
        cognitive_level: str = "understanding",
    ) -> Dict[str, Any]:
        """
        Generate MCQ using pure factual content (fixes prompt leakage!)

        Args:
            pure_content: Factual content about the topic (from RAG, not instructions)
            topic: The topic name for context
            difficulty: Difficulty level
            cognitive_level: Cognitive level for question complexity
        """
        try:
            logger.info(f"🎯 Generating MCQ from pure content (length: {len(pure_content)})")

            # Convert string cognitive level to enum
            cognitive_level_enum = self._string_to_cognitive_level(cognitive_level)

            # Create MCQ generation context with pure content
            context = MCQGenerationContext(
                topic=topic,
                context_passages=[pure_content],  # Pure content, not instructions!
                difficulty=difficulty,
                cognitive_level=cognitive_level_enum,
                source_documents=[],  # Empty source documents
                target_audience=self._get_target_audience(difficulty)
            )

            # Generate using the existing context-based method
            result = await self._generate_mcq_with_context(context)

            if result:
                # Mark as content-based generation
                result["generation_method"] = "pure_content_based"
                result["grounded"] = True
                logger.info("✅ Successfully generated MCQ from pure content")
            else:
                logger.error("❌ Pure content generation failed")
                raise Exception(f"Pure content generation failed for topic '{topic}'")

            return result

        except Exception as e:
            logger.error(f"❌ Pure content MCQ generation failed: {e}")
            raise Exception(f"Pure content MCQ generation failed for '{topic}': {str(e)}")

    def _string_to_cognitive_level(self, cognitive_level: str) -> CognitiveLevel:
        """Convert string cognitive level to enum"""
        mapping = {
            "understanding": CognitiveLevel.UNDERSTANDING,
            "applying": CognitiveLevel.APPLYING,
            "analyzing": CognitiveLevel.ANALYZING,
            "evaluating": CognitiveLevel.EVALUATING,
            "remembering": CognitiveLevel.REMEMBERING,
            "creating": CognitiveLevel.CREATING,
        }
        return mapping.get(cognitive_level.lower(), CognitiveLevel.UNDERSTANDING)
