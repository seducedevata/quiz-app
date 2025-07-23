"""
Intelligent MCQ Generator - The Fire Method Implementation

This module implements the "Inquisitor's Mandate" prompting technique and RAG-based
question generation to produce high-quality, specific MCQs instead of generic garbage.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCQContext:
    """Container for MCQ generation context"""

    topic: str
    context_text: str
    difficulty: str
    question_type: str


class InquisitorPromptTemplate:
    """The Inquisitor's Mandate - Rigid prompting for quality MCQs"""

    @staticmethod
    def create_mandate_prompt(context: MCQContext) -> str:
        """Create the rigid, non-negotiable prompt that forces quality"""
        from .inquisitor_prompt import _create_inquisitor_prompt

        return _create_inquisitor_prompt(context.context_text, context.topic, context.difficulty, context.question_type)

    @staticmethod
    def create_distractor_prompt(context: MCQContext, question: str, correct_answer: str) -> str:
        """Create prompt specifically for generating better distractors"""

        return f"""### ROLE & GOAL ###
You are an expert in creating challenging multiple-choice distractors. Your job is to generate three highly plausible but incorrect options that will challenge students.

### QUALITY REQUIREMENTS ###
- Each distractor must be substantive and non-empty
- Include domain-specific terminology
- Make distractors plausible but clearly incorrect
- Ensure all options are of similar length and complexity

### CONTEXT ###
{context.context_text}

### QUESTION ###
{question}

### CORRECT ANSWER ###
{correct_answer}

### THE TASK ###
Generate three plausible but incorrect distractor answers based on:
1. Common misconceptions related to this topic
2. Partial understanding that leads to wrong conclusions
3. Confusion with related but distinct concepts
4. Logical-sounding but factually incorrect statements

### CONSTRAINTS ###
- Each distractor must be clearly wrong according to the CONTEXT
- Distractors should be similar in length and complexity to the correct answer
- Avoid obviously wrong answers that any student would immediately eliminate
- Base distractors on actual misconceptions, not random information

### OUTPUT FORMAT ###
Respond with ONLY a JSON array of three distractor options:
["Distractor 1", "Distractor 2", "Distractor 3"]"""


class SocraticGauntlet:
    """Multi-step chain-of-thought MCQ generation with self-critique"""

    def __init__(self, model_interface):
        self.model = model_interface

    async def generate_with_critique(self, context: MCQContext) -> Dict[str, Any]:
        """Run the full Socratic Gauntlet process"""

        # Step 1: Initial Draft
        draft_mcq = await self._generate_initial_draft(context)
        if not draft_mcq:
            raise ValueError("Failed to generate initial draft")

        # Step 2: Justification (Chain-of-Thought)
        rationale = await self._generate_rationale(context, draft_mcq)

        # Step 3: Critique
        critique = await self._generate_critique(context, draft_mcq, rationale)

        # Step 4: Final Polish
        final_mcq = await self._generate_final_version(context, draft_mcq, rationale, critique)

        return final_mcq

    async def _generate_initial_draft(self, context: MCQContext) -> Dict[str, Any]:
        """Step 1: Generate initial draft"""
        prompt = f"""Based on this context, draft one question, four options, and identify the correct one.

CONTEXT:
{context.context_text}

Create a {context.difficulty} level {context.question_type} question. Respond in JSON format:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_answer": "..."}}"""

        response = await self.model.generate_text(prompt)
        return self._parse_json_response(response)

    async def _generate_rationale(self, context: MCQContext, mcq: Dict[str, Any]) -> str:
        """Step 2: Generate step-by-step rationale"""
        prompt = f"""For the question you just drafted, provide a step-by-step rationale for why the correct answer is correct and why each of the three distractors is incorrect. Cite specific phrases from the context for each point.

CONTEXT:
{context.context_text}

QUESTION: {mcq['question']}
OPTIONS: {mcq['options']}
CORRECT: {mcq['correct_answer']}

Provide detailed reasoning for each option."""

        return await self.model.generate_text(prompt)

    async def _generate_critique(
        self, context: MCQContext, mcq: Dict[str, Any], rationale: str
    ) -> str:
        """Step 3: Generate critique of the question"""
        prompt = f"""You are a quality assurance inspector. Based on the provided question and rationale, identify any flaws. Is the question ambiguous? Are the distractors too easy? Is the rationale weak? Suggest one specific improvement.

CONTEXT:
{context.context_text}

QUESTION: {mcq['question']}
RATIONALE: {rationale}

Provide specific critique and improvement suggestions."""

        return await self.model.generate_text(prompt)

    async def _generate_final_version(
        self, context: MCQContext, mcq: Dict[str, Any], rationale: str, critique: str
    ) -> Dict[str, Any]:
        """Step 4: Generate final polished version"""
        prompt = f"""Using the critique you just provided, generate the final, improved version of the MCQ in the required JSON format.

ORIGINAL MCQ: {mcq}
RATIONALE: {rationale}
CRITIQUE: {critique}

Generate the improved final version in JSON format:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_answer": "...", "explanation": "..."}}"""

        response = await self.model.generate_text(prompt)
        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response using the enhanced parser"""
        from ..utils.enhanced_mcq_parser import EnhancedMCQParser

        parser = EnhancedMCQParser()
        parse_result = parser.parse_mcq(response)

        if parse_result.success:
            return parse_result.mcq_data
        else:
            logger.error(f"Failed to parse intelligent MCQ response: {parse_result.issues}")
            raise ValueError(f"Failed to parse model response: {parse_result.issues}")


class IntelligentMCQGenerator:
    """Main intelligent MCQ generator using Fire Methods"""

    def __init__(self, model_interface, rag_engine=None):
        self.model = model_interface
        self.rag_engine = rag_engine
        self.socratic_gauntlet = SocraticGauntlet(model_interface)

    async def generate_mcq(
        self,
        topic: str,
        difficulty: str = "medium",
        question_type: str = "Multiple Choice",
        use_rag: bool = True,
        use_gauntlet: bool = False,
    ) -> Dict[str, Any]:
        """Generate high-quality MCQ using Fire Methods"""

        try:
            # Step 1: Get context (RAG or fallback)
            context_text = await self._get_context(topic, use_rag)

            # Create context object
            context = MCQContext(
                topic=topic,
                context_text=context_text,
                difficulty=difficulty,
                question_type=question_type,
            )

            # Step 2: Generate MCQ
            if use_gauntlet:
                # Use Socratic Gauntlet for highest quality
                mcq = await self.socratic_gauntlet.generate_with_critique(context)
            else:
                # Use Inquisitor's Mandate for fast, high-quality generation
                mcq = await self._generate_with_mandate(context)

            # Step 3: Validate and enhance
            mcq = self._validate_and_enhance_mcq(mcq, context)

            logger.info(f"✅ Generated intelligent MCQ for topic: {topic}")
            return mcq

        except Exception as e:
            logger.error(f"❌ Intelligent MCQ generation failed: {e}")
            # No fallback - raise error for pure AI generation
            raise Exception(f"Intelligent MCQ generation failed for '{topic}': {str(e)}")

    async def _get_context(self, topic: str, use_rag: bool) -> str:
        """Get relevant context for the topic"""

        if use_rag and self.rag_engine:
            try:
                # Use RAG to find relevant context
                context_chunks = await self.rag_engine.retrieve_context(topic, top_k=3)
                context_text = "\n\n".join(context_chunks)

                if len(context_text.strip()) > 100:  # Minimum viable context
                    logger.info(f"✅ Retrieved RAG context: {len(context_text)} characters")
                    return context_text
                else:
                    logger.warning("⚠️ RAG context too short, using fallback")

            except Exception as e:
                logger.error(f"❌ RAG retrieval failed: {e}")

        # Fallback: Use curated context based on topic
        return self._get_curated_context(topic)

    def _get_curated_context(self, topic: str) -> str:
        """Get curated context for common topics"""

        # High-quality curated contexts for common topics
        contexts = {
            "magnetism": """
Magnetism is a fundamental force of nature that arises from the motion of electric charges. When electric charges move, they create magnetic fields. The strength and direction of a magnetic field can be represented by magnetic field lines, which form closed loops from the north pole to the south pole of a magnet.

The magnetic force on a moving charged particle is given by the Lorentz force equation: F = q(v × B), where q is the charge, v is the velocity vector, and B is the magnetic field vector. This force is always perpendicular to both the velocity of the particle and the magnetic field direction.

Faraday's law of electromagnetic induction states that a changing magnetic flux through a conductor induces an electromotive force (EMF). The magnitude of the induced EMF is proportional to the rate of change of magnetic flux: EMF = -dΦ/dt, where Φ is the magnetic flux.

Magnetic materials can be classified into three main categories: ferromagnetic (strongly attracted to magnets, like iron), paramagnetic (weakly attracted to magnets), and diamagnetic (weakly repelled by magnets). Ferromagnetic materials can become permanently magnetized due to the alignment of magnetic domains within the material.
""",
            "physics": """
Physics is the fundamental science that seeks to understand how the universe works. It studies matter, energy, and their interactions across all scales, from subatomic particles to galaxies.

Classical mechanics, developed by Newton, describes the motion of objects from projectiles to planets. Newton's three laws of motion form the foundation: (1) an object at rest stays at rest unless acted upon by a force, (2) force equals mass times acceleration (F=ma), and (3) for every action there is an equal and opposite reaction.

Thermodynamics deals with heat, temperature, and energy transfer. The first law states that energy cannot be created or destroyed, only converted from one form to another. The second law introduces the concept of entropy, stating that the entropy of an isolated system always increases over time.

Quantum mechanics revolutionized our understanding of the atomic and subatomic world. It reveals that energy comes in discrete packets called quanta, and that particles exhibit both wave-like and particle-like properties depending on how they are observed.
""",
            "chemistry": """
Chemistry is the science that studies the composition, structure, properties, and behavior of matter at the atomic and molecular level. All matter is composed of atoms, which are the basic building blocks of elements.

The periodic table organizes elements by their atomic number (number of protons). Elements in the same group (vertical columns) have similar chemical properties due to having the same number of valence electrons. The periodic trends include atomic radius, ionization energy, and electronegativity.

Chemical bonding occurs when atoms interact to form compounds. Ionic bonds form between metals and nonmetals through electron transfer, creating charged ions. Covalent bonds form when atoms share electrons, typically between nonmetals. Metallic bonding occurs in metals where electrons are delocalized in a "sea" of electrons.

Chemical reactions involve the breaking and forming of bonds. The law of conservation of mass states that matter cannot be created or destroyed in chemical reactions, only rearranged. Reaction rates depend on factors such as temperature, concentration, surface area, and the presence of catalysts.
""",
            "biology": """
Biology is the study of living organisms and their interactions with each other and their environment. All living things share certain characteristics: they are made of cells, grow and develop, reproduce, respond to stimuli, maintain homeostasis, and evolve over time.

The cell is the basic unit of life. Prokaryotic cells (bacteria and archaea) lack a membrane-bound nucleus, while eukaryotic cells (plants, animals, fungi, protists) have a nucleus and other membrane-bound organelles. Cell membranes control what enters and exits the cell through selective permeability.

DNA (deoxyribonucleic acid) carries genetic information in all living organisms. The structure of DNA is a double helix composed of nucleotides containing four bases: adenine (A), thymine (T), guanine (G), and cytosine (C). Base pairing rules state that A pairs with T and G pairs with C.

Evolution by natural selection explains the diversity of life. Organisms with favorable traits are more likely to survive and reproduce, passing these traits to their offspring. Over time, this leads to changes in populations and the emergence of new species.
""",
        }

        # Get context for topic or use general context
        context = contexts.get(topic.lower(), contexts.get("physics", ""))

        if not context:
            # Generate basic context if topic not found
            context = f"""
{topic.title()} is an important area of study that involves understanding fundamental concepts, principles, and applications. Students studying {topic} need to grasp both theoretical foundations and practical implications.

Key aspects of {topic} include understanding the underlying mechanisms, recognizing patterns and relationships, and applying knowledge to solve problems. Mastery requires both conceptual understanding and the ability to analyze and synthesize information.

The study of {topic} builds upon foundational knowledge and connects to other related fields. Students must develop critical thinking skills to evaluate information, draw conclusions, and make predictions based on evidence and established principles.
"""

        return context.strip()

    async def _generate_with_mandate(self, context: MCQContext) -> Dict[str, Any]:
        """Generate MCQ using Inquisitor's Mandate"""

        # Create the rigid mandate prompt
        prompt = InquisitorPromptTemplate.create_mandate_prompt(context)

        # Generate response
        response = await self.model.generate_text(prompt)

        # Parse JSON response
        mcq = self._parse_json_response(response)

        # Enhance distractors if needed
        if self._distractors_need_improvement(mcq):
            mcq = await self._improve_distractors(context, mcq)

        return mcq

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model"""
        try:
            # Clean up response - remove markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise

    def _distractors_need_improvement(self, mcq: Dict[str, Any]) -> bool:
        """Check if distractors need improvement"""
        options = mcq.get("options", {})
        if len(options) != 4:
            return True

        # Check for obviously bad distractors
        for key, option in options.items():
            if len(option.strip()) < 10:  # Too short
                return True
            if "important" in option.lower() and "understand" in option.lower():  # Generic
                return True

        return False

    async def _improve_distractors(
        self, context: MCQContext, mcq: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Improve distractors using specialized prompt"""

        correct_key = mcq.get("correct_answer", "A")
        correct_answer = mcq["options"][correct_key]

        # Generate better distractors
        distractor_prompt = InquisitorPromptTemplate.create_distractor_prompt(
            context, mcq["question"], correct_answer
        )

        response = await self.model.generate_text(distractor_prompt)

        try:
            distractors = json.loads(response.strip())

            # Replace distractors while keeping correct answer
            options = {"A": "", "B": "", "C": "", "D": ""}
            options[correct_key] = correct_answer

            distractor_keys = [k for k in options.keys() if k != correct_key]
            for i, distractor in enumerate(distractors[:3]):
                if i < len(distractor_keys):
                    options[distractor_keys[i]] = distractor

            mcq["options"] = options

        except Exception as e:
            logger.error(f"Failed to improve distractors: {e}")

        return mcq

    def _validate_and_enhance_mcq(self, mcq: Dict[str, Any], context: MCQContext) -> Dict[str, Any]:
        """Validate and enhance the generated MCQ"""

        # Ensure all required fields exist
        required_fields = ["question", "options", "correct_answer", "explanation"]
        for field in required_fields:
            if field not in mcq:
                raise ValueError(f"Missing required field: {field}")

        # Validate options
        if len(mcq["options"]) != 4:
            raise ValueError("MCQ must have exactly 4 options")

        # Ensure correct answer is valid
        if mcq["correct_answer"] not in mcq["options"]:
            raise ValueError("Correct answer key not found in options")

        # Add metadata
        mcq["topic"] = context.topic
        mcq["difficulty"] = context.difficulty
        mcq["question_type"] = context.question_type
        mcq["generation_method"] = "intelligent_fire_method"

        return mcq


