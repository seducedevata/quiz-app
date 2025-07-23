#!/usr/bin/env python3
"""
🚀 TRAINING DATA GENERATOR
Advanced training data generation pipeline that processes documents through local AI models
to create high-quality JSON training datasets for model fine-tuning.
"""

import json
import logging
import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainingDataConfig:
    """Configuration for training data generation"""
    source_model: str  # Model used for data generation (DeepSeek, Llama, etc.)
    target_model: str  # Model to be trained
    question_types: List[str]  # ["numerical", "conceptual", "mixed"]
    difficulty_levels: List[str]  # ["easy", "medium", "hard", "expert"]
    questions_per_chunk: int = 5
    chunk_size: int = 1000
    output_format: str = "json"  # "json", "csv", "xml"
    quality_threshold: float = 0.8
    use_anti_vague_enforcement: bool = True
    include_explanations: bool = True

@dataclass
class GeneratedQuestion:
    """Structure for a generated training question"""
    id: str
    question: str
    options: List[str]
    correct_answer: str
    correct_index: int
    explanation: str
    topic: str
    difficulty: str
    question_type: str
    source_chunk: str
    quality_score: float
    metadata: Dict[str, Any]

class TrainingDataGenerator:
    """
    🚀 ADVANCED TRAINING DATA GENERATOR
    
    Processes documents through local AI models to generate high-quality
    training datasets in JSON/CSV/XML format for model fine-tuning.
    """
    
    def __init__(self, config: TrainingDataConfig):
        self.config = config
        self.generated_questions: List[GeneratedQuestion] = []
        self.processing_stats = {
            "chunks_processed": 0,
            "questions_generated": 0,
            "questions_accepted": 0,
            "questions_rejected": 0,
            "average_quality_score": 0.0,
            "processing_time": 0.0
        }
        
        # Initialize AI model interfaces
        self._init_model_interfaces()
    
    def _init_model_interfaces(self):
        """Initialize the AI model interfaces for data generation"""
        try:
            # Import existing model managers
            from .mcq_manager import MCQManager
            from .unified_inference_manager import UnifiedInferenceManager
            
            self.mcq_manager = MCQManager()
            self.inference_manager = UnifiedInferenceManager()
            
            logger.info(f"🚀 Training data generator initialized with source model: {self.config.source_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model interfaces: {e}")
            raise
    
    async def generate_training_data_from_documents(
        self, 
        processed_chunks: List[Dict[str, Any]], 
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        🚀 MAIN PIPELINE: Generate training data from processed document chunks
        
        Args:
            processed_chunks: List of processed document chunks from AdvancedDocumentProcessor
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing generated training data and statistics
        """
        start_time = time.time()
        logger.info(f"🚀 Starting training data generation from {len(processed_chunks)} chunks")
        
        try:
            # Reset stats
            self.processing_stats = {
                "chunks_processed": 0,
                "questions_generated": 0,
                "questions_accepted": 0,
                "questions_rejected": 0,
                "average_quality_score": 0.0,
                "processing_time": 0.0
            }
            
            total_chunks = len(processed_chunks)
            
            for i, chunk in enumerate(processed_chunks):
                if progress_callback:
                    progress_callback(f"Processing chunk {i+1}/{total_chunks}")
                
                # Generate questions from this chunk
                chunk_questions = await self._generate_questions_from_chunk(chunk)
                
                # Quality filter and add to collection
                for question in chunk_questions:
                    if question.quality_score >= self.config.quality_threshold:
                        self.generated_questions.append(question)
                        self.processing_stats["questions_accepted"] += 1
                    else:
                        self.processing_stats["questions_rejected"] += 1
                        logger.debug(f"❌ Rejected question with quality score: {question.quality_score}")
                
                self.processing_stats["chunks_processed"] += 1
                self.processing_stats["questions_generated"] += len(chunk_questions)
                
                # Update progress
                progress = (i + 1) / total_chunks * 100
                if progress_callback:
                    progress_callback(f"Processed {i+1}/{total_chunks} chunks ({progress:.1f}%)")
            
            # Calculate final statistics
            self.processing_stats["processing_time"] = time.time() - start_time
            if self.generated_questions:
                self.processing_stats["average_quality_score"] = sum(
                    q.quality_score for q in self.generated_questions
                ) / len(self.generated_questions)
            
            logger.info(f"✅ Training data generation completed: {len(self.generated_questions)} high-quality questions")
            
            return {
                "success": True,
                "questions": self.generated_questions,
                "statistics": self.processing_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"❌ Training data generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": self.processing_stats
            }
    
    async def _generate_questions_from_chunk(self, chunk: Dict[str, Any]) -> List[GeneratedQuestion]:
        """Generate questions from a single document chunk"""
        try:
            chunk_text = chunk.get("text", "")
            chunk_topic = chunk.get("topic", "General Knowledge")
            chunk_metadata = chunk.get("metadata", {})
            
            if len(chunk_text.strip()) < 100:  # Skip very short chunks
                return []
            
            questions = []
            
            # Generate questions for each configured difficulty and type
            for difficulty in self.config.difficulty_levels:
                for question_type in self.config.question_types:
                    
                    # Generate multiple questions per chunk/difficulty/type combination
                    for q_index in range(self.config.questions_per_chunk):
                        try:
                            question = await self._generate_single_question(
                                chunk_text, chunk_topic, difficulty, question_type, chunk_metadata
                            )
                            
                            if question:
                                questions.append(question)
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to generate question {q_index+1} for {difficulty}/{question_type}: {e}")
                            continue
            
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to process chunk: {e}")
            return []
    
    async def _generate_single_question(
        self, 
        context: str, 
        topic: str, 
        difficulty: str, 
        question_type: str,
        metadata: Dict[str, Any]
    ) -> Optional[GeneratedQuestion]:
        """Generate a single high-quality question using the configured source model"""
        try:
            # Prepare quiz parameters for the MCQ manager
            quiz_params = {
                "topic": topic,
                "difficulty": difficulty,
                "question_type": question_type,
                "game_mode": "training",  # Special mode for training data generation
                "context": context,
                "use_anti_vague": self.config.use_anti_vague_enforcement,
                "source_model": self.config.source_model
            }
            
            # Generate question using existing MCQ infrastructure
            result = await self.mcq_manager.generate_quiz_async(quiz_params)
            
            if not result or not result.get("success"):
                logger.warning(f"⚠️ MCQ generation failed for {topic}/{difficulty}/{question_type}")
                return None
            
            question_data = result.get("question", {})
            if not question_data:
                return None
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(question_data, context, difficulty, question_type)
            
            # Create GeneratedQuestion object
            generated_question = GeneratedQuestion(
                id=str(uuid.uuid4()),
                question=question_data.get("question", ""),
                options=question_data.get("options", []),
                correct_answer=question_data.get("correct_answer", ""),
                correct_index=question_data.get("correct_index", 0),
                explanation=question_data.get("explanation", ""),
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                source_chunk=context[:500] + "..." if len(context) > 500 else context,
                quality_score=quality_score,
                metadata={
                    **metadata,
                    "generated_at": time.time(),
                    "source_model": self.config.source_model,
                    "target_model": self.config.target_model
                }
            )
            
            return generated_question
            
        except Exception as e:
            logger.error(f"❌ Single question generation failed: {e}")
            return None
    
    def _calculate_quality_score(
        self, 
        question_data: Dict[str, Any], 
        context: str, 
        difficulty: str, 
        question_type: str
    ) -> float:
        """Calculate quality score for a generated question"""
        try:
            score = 0.0
            max_score = 10.0
            
            question_text = question_data.get("question", "")
            options = question_data.get("options", [])
            explanation = question_data.get("explanation", "")
            
            # Basic completeness check (2 points)
            if question_text and len(options) >= 4 and explanation:
                score += 2.0
            
            # Question length and complexity (2 points)
            if len(question_text) >= 50:
                score += 1.0
            if len(question_text) >= 100:
                score += 1.0
            
            # Question type adherence (2 points)
            if question_type == "numerical":
                if any(word in question_text.lower() for word in ["calculate", "compute", "determine", "find"]):
                    score += 1.0
                if any(char.isdigit() for char in question_text):
                    score += 1.0
            elif question_type == "conceptual":
                if any(word in question_text.lower() for word in ["explain", "why", "how", "describe"]):
                    score += 1.0
                if not any(word in question_text.lower() for word in ["calculate", "compute"]):
                    score += 1.0
            else:  # mixed
                score += 2.0
            
            # Difficulty appropriateness (2 points)
            if difficulty == "expert":
                if len(question_text) >= 120 and len(explanation) >= 100:
                    score += 2.0
            elif difficulty == "hard":
                if len(question_text) >= 80 and len(explanation) >= 80:
                    score += 2.0
            else:
                if len(question_text) >= 50 and len(explanation) >= 50:
                    score += 2.0
            
            # Anti-vague enforcement (2 points)
            vague_words = ["what is", "what are", "main purpose", "primary function"]
            if not any(vague in question_text.lower() for vague in vague_words):
                score += 1.0
            if "mechanism" in question_text.lower() or "process" in question_text.lower():
                score += 1.0
            
            return min(score / max_score, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.0
    
    def export_training_data(self, output_path: str) -> bool:
        """Export generated training data to specified format"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.output_format.lower() == "json":
                return self._export_json(output_path)
            elif self.config.output_format.lower() == "csv":
                return self._export_csv(output_path)
            elif self.config.output_format.lower() == "xml":
                return self._export_xml(output_path)
            else:
                logger.error(f"❌ Unsupported output format: {self.config.output_format}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False
    
    def _export_json(self, output_path: Path) -> bool:
        """Export training data as JSON"""
        try:
            training_data = {
                "metadata": {
                    "generated_at": time.time(),
                    "source_model": self.config.source_model,
                    "target_model": self.config.target_model,
                    "total_questions": len(self.generated_questions),
                    "statistics": self.processing_stats,
                    "config": {
                        "question_types": self.config.question_types,
                        "difficulty_levels": self.config.difficulty_levels,
                        "quality_threshold": self.config.quality_threshold
                    }
                },
                "training_examples": []
            }
            
            for question in self.generated_questions:
                training_example = {
                    "id": question.id,
                    "input": {
                        "context": question.source_chunk,
                        "topic": question.topic,
                        "difficulty": question.difficulty,
                        "question_type": question.question_type
                    },
                    "output": {
                        "question": question.question,
                        "options": question.options,
                        "correct_answer": question.correct_answer,
                        "correct_index": question.correct_index,
                        "explanation": question.explanation
                    },
                    "metadata": question.metadata,
                    "quality_score": question.quality_score
                }
                training_data["training_examples"].append(training_example)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Training data exported to JSON: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            return False

    def _export_csv(self, output_path: Path) -> bool:
        """Export training data as CSV"""
        try:
            import csv

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'id', 'question', 'option_a', 'option_b', 'option_c', 'option_d',
                    'correct_answer', 'correct_index', 'explanation', 'topic',
                    'difficulty', 'question_type', 'source_chunk', 'quality_score',
                    'generated_at', 'source_model', 'target_model'
                ])

                # Write data rows
                for question in self.generated_questions:
                    options = question.options + [''] * (4 - len(question.options))  # Pad to 4 options
                    writer.writerow([
                        question.id,
                        question.question,
                        options[0] if len(options) > 0 else '',
                        options[1] if len(options) > 1 else '',
                        options[2] if len(options) > 2 else '',
                        options[3] if len(options) > 3 else '',
                        question.correct_answer,
                        question.correct_index,
                        question.explanation,
                        question.topic,
                        question.difficulty,
                        question.question_type,
                        question.source_chunk,
                        question.quality_score,
                        question.metadata.get('generated_at', ''),
                        question.metadata.get('source_model', ''),
                        question.metadata.get('target_model', '')
                    ])

            logger.info(f"✅ Training data exported to CSV: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            return False

    def _export_xml(self, output_path: Path) -> bool:
        """Export training data as XML"""
        try:
            import xml.etree.ElementTree as ET

            root = ET.Element("training_data")

            # Add metadata
            metadata = ET.SubElement(root, "metadata")
            ET.SubElement(metadata, "generated_at").text = str(time.time())
            ET.SubElement(metadata, "source_model").text = self.config.source_model
            ET.SubElement(metadata, "target_model").text = self.config.target_model
            ET.SubElement(metadata, "total_questions").text = str(len(self.generated_questions))

            # Add statistics
            stats = ET.SubElement(metadata, "statistics")
            for key, value in self.processing_stats.items():
                ET.SubElement(stats, key).text = str(value)

            # Add training examples
            examples = ET.SubElement(root, "training_examples")

            for question in self.generated_questions:
                example = ET.SubElement(examples, "example")
                example.set("id", question.id)

                # Input section
                input_elem = ET.SubElement(example, "input")
                ET.SubElement(input_elem, "context").text = question.source_chunk
                ET.SubElement(input_elem, "topic").text = question.topic
                ET.SubElement(input_elem, "difficulty").text = question.difficulty
                ET.SubElement(input_elem, "question_type").text = question.question_type

                # Output section
                output_elem = ET.SubElement(example, "output")
                ET.SubElement(output_elem, "question").text = question.question

                options_elem = ET.SubElement(output_elem, "options")
                for i, option in enumerate(question.options):
                    option_elem = ET.SubElement(options_elem, "option")
                    option_elem.set("index", str(i))
                    option_elem.text = option

                ET.SubElement(output_elem, "correct_answer").text = question.correct_answer
                ET.SubElement(output_elem, "correct_index").text = str(question.correct_index)
                ET.SubElement(output_elem, "explanation").text = question.explanation

                # Metadata
                meta_elem = ET.SubElement(example, "metadata")
                ET.SubElement(meta_elem, "quality_score").text = str(question.quality_score)
                for key, value in question.metadata.items():
                    ET.SubElement(meta_elem, key).text = str(value)

            # Write to file
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)

            logger.info(f"✅ Training data exported to XML: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ XML export failed: {e}")
            return False
