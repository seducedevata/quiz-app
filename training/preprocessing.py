"""
Data Preprocessing Module for AI Training

This module handles preprocessing of training data for the AI model fine-tuning pipeline.
"""

import logging
import json
import os
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_data(domain_name: str, training_params: Dict[str, Any], device) -> Dict[str, Any]:
    """
    Preprocess training data for the specified domain
    
    Args:
        domain_name: Name of the training domain
        training_params: Training configuration parameters
        device: Training device (cuda/cpu)
        
    Returns:
        Dictionary containing preprocessed training data
    """
    try:
        logger.info(f"🔄 Preprocessing data for domain: {domain_name}")
        
        # Get data directory
        # processed_docs moved to training components - not part of main app
        data_dir = Path("data/cache")  # Use cache directory instead
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for training data files
        training_files = []
        
        # Check for domain-specific training data
        domain_file = data_dir / f"{domain_name}_training_data.jsonl"
        if domain_file.exists():
            training_files.append(domain_file)
        
        # Check for general training data
        general_files = [
            data_dir / "combined_training_dataset.txt",
            data_dir / "training_data.jsonl"
        ]
        
        for file_path in general_files:
            if file_path.exists():
                training_files.append(file_path)
        
        if not training_files:
            # Create sample training data
            logger.warning("⚠️ No training data found, creating sample data...")
            training_samples = _create_sample_training_data(domain_name)
        else:
            # Load existing training data
            training_samples = _load_training_data(training_files)
        
        # Filter and validate samples
        valid_samples = _validate_training_samples(training_samples)
        
        # Apply preprocessing based on training parameters
        processed_samples = _apply_preprocessing(valid_samples, training_params)
        
        logger.info(f"✅ Preprocessing complete:")
        logger.info(f"   📊 Total samples: {len(processed_samples)}")
        logger.info(f"   📁 Files processed: {len(training_files)}")
        
        return {
            "training_samples": processed_samples,
            "domain_name": domain_name,
            "total_samples": len(processed_samples),
            "files_processed": [str(f) for f in training_files]
        }
        
    except Exception as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        raise


def _load_training_data(training_files: List[Path]) -> List[Dict[str, Any]]:
    """Load training data from files"""
    all_samples = []
    
    for file_path in training_files:
        try:
            logger.info(f"📄 Loading training data from: {file_path}")
            
            if file_path.suffix == '.jsonl':
                # Load JSONL format
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            if line.strip():
                                sample = json.loads(line.strip())
                                all_samples.append(sample)
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Invalid JSON on line {line_num}: {e}")
            
            elif file_path.suffix == '.txt':
                # Load text format and convert to instruction format
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Split content into chunks and create instruction-response pairs
                chunks = content.split('\n\n')
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Only use substantial chunks
                        sample = {
                            "instruction": f"Explain the following concept:",
                            "input": "",
                            "output": chunk.strip()
                        }
                        all_samples.append(sample)
            
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
    
    return all_samples


def _validate_training_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and filter training samples"""
    valid_samples = []
    
    required_fields = ["instruction", "output"]
    
    for sample in samples:
        # Check required fields
        if all(field in sample for field in required_fields):
            # Check minimum content length
            if len(sample["instruction"]) > 10 and len(sample["output"]) > 20:
                # Ensure input field exists
                if "input" not in sample:
                    sample["input"] = ""
                
                valid_samples.append(sample)
    
    logger.info(f"✅ Validated {len(valid_samples)} out of {len(samples)} samples")
    return valid_samples


def _apply_preprocessing(samples: List[Dict[str, Any]], training_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply preprocessing based on training parameters"""
    processed_samples = []
    
    max_samples = training_params.get("max_samples", len(samples))
    min_instruction_length = training_params.get("min_instruction_length", 10)
    min_output_length = training_params.get("min_output_length", 20)
    
    for sample in samples[:max_samples]:
        # Filter by length requirements
        if (len(sample["instruction"]) >= min_instruction_length and 
            len(sample["output"]) >= min_output_length):
            
            # Clean up text
            sample["instruction"] = sample["instruction"].strip()
            sample["output"] = sample["output"].strip()
            sample["input"] = sample.get("input", "").strip()
            
            processed_samples.append(sample)
    
    return processed_samples


def _create_sample_training_data(domain_name: str) -> List[Dict[str, Any]]:
    """Create sample training data for the domain"""
    sample_data = [
        {
            "instruction": "Explain the concept of artificial intelligence",
            "input": "",
            "output": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses various techniques including machine learning, neural networks, and deep learning to enable computers to perform tasks that typically require human intelligence."
        },
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on the learned patterns."
        },
        {
            "instruction": "Describe the importance of data in AI",
            "input": "",
            "output": "Data is the foundation of artificial intelligence systems. Quality, relevant, and sufficient data is crucial for training AI models effectively. The performance of AI systems directly depends on the data they are trained on - better data leads to better AI performance."
        },
        {
            "instruction": f"Explain a concept related to {domain_name}",
            "input": "",
            "output": f"This is a sample explanation related to {domain_name}. In a real implementation, this would contain domain-specific content that helps train the AI model for specialized tasks in the {domain_name} field."
        }
    ]
    
    logger.info(f"📝 Created {len(sample_data)} sample training examples")
    return sample_data 